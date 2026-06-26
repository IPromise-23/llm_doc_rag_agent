from __future__ import annotations 

from pathlib import Path
from typing import Any

from llm_doc_rag_agent.agents import build_rag_graph
from llm_doc_rag_agent.agents.llm_quality import RuleBasedQualityGrader, build_quality_grader
from llm_doc_rag_agent.chunking import SimpleTextSplitter
from llm_doc_rag_agent.config import Settings
from llm_doc_rag_agent.embeddings import SentenceTransformerEmbeddingProvider
from llm_doc_rag_agent.generation import QAService
from llm_doc_rag_agent.loaders import LocalDocumentLoader
from llm_doc_rag_agent.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    NoOpReranker,
)
from llm_doc_rag_agent.schemas import Answer, Chunk
from llm_doc_rag_agent.vectorstores import QdrantVectorStore


class _ConfiguredRetriever: # 适配 graph.py
    """Late-bound retriever adapter so graph routing can skip retrieval entirely."""

    def __init__(self, service: "RagService", retriever_type: str, candidate_k: int | None) -> None:
        self.service = service
        self.retriever_type = retriever_type
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5): # 给 graph 调用的统一接口，把请求转交给 RagService.retrieve_only() , 并附带当前配置的 retriever_type && candidate_k
        return self.service.retrieve_only(
            question=query,
            top_k=top_k,
            retriever_type=self.retriever_type,
            candidate_k=self.candidate_k,
        )


class RagService:
    """Application service shared by CLI, API, tests, and evaluation."""

    def __init__(self, settings: Settings, collection: str | None = None) -> None:
        self.settings = settings
        self.collection = collection or settings.default_collection
        self.embeddings = SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        self.loader = LocalDocumentLoader(ignore_roots=[settings.resolved_project_root])
        self.splitter = SimpleTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._store: QdrantVectorStore | None = None
        self._retriever: DenseRetriever | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._qa: QAService | None = None
        self._quality_grader: RuleBasedQualityGrader | None = None

    @property   # 懒加载语法，让方法像属性一样使用，外部可以写 service.store
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(    # 第一次访问时创建 Qdrant store ,之后重复使用同一个对象
                path=self.settings.resolved_qdrant_path,
                collection=self.collection,
            )
        return self._store

    @property
    def retriever(self) -> DenseRetriever:
        if self._retriever is None:
            self._retriever = DenseRetriever(embeddings=self.embeddings, store=self.store)
        return self._retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(store=self.store)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            reranker = (
                CrossEncoderReranker(self.settings.reranker_model)
                if self.settings.reranker_model
                else NoOpReranker()
            )
            self._hybrid_retriever = HybridRetriever(
                dense=self.retriever,
                bm25=self.bm25_retriever,
                reranker=reranker,
            )
        return self._hybrid_retriever

    @property
    def qa(self) -> QAService:
        if self._qa is None:
            self._qa = QAService(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
            )
        return self._qa

    @property
    def quality_grader(self) -> RuleBasedQualityGrader:
        if self._quality_grader is None:
            self._quality_grader = build_quality_grader(
                mode=self.settings.quality_grader,
                api_key=self.settings.quality_api_key or self.settings.deepseek_api_key,
                base_url=self.settings.quality_base_url or self.settings.llm_base_url,
                model=self.settings.quality_model or self.settings.llm_model,
                max_tokens=self.settings.quality_max_tokens,
                top_p=self.settings.quality_top_p,
                disable_thinking=self.settings.quality_disable_thinking,
            )
        return self._quality_grader

    def ingest_path(self, path: str | Path, recreate: bool = False, batch_size: int = 64) -> dict[str, Any]:    # 把本地文件或者目录入库    # 增量索引，每次 ingest 只处理新增或发生变化的文档；按照源文件 source 级别实现的，并非按照单个 chunk 级别实现
        documents = self.loader.load_path(path)     # 加载文档，得到 Docunment 列表
        if not documents:
            return {
                "collection": self.collection,
                "documents": 0,
                "changed_documents": 0,
                "skipped_documents": 0,
                "chunks": 0,
                "upserted": 0,
                "deleted_chunks": 0,
                "qdrant_path": str(self.settings.resolved_qdrant_path),
            }
        if recreate or not self.store.collection_exists():          # 如果要求重建或者 collection 不存在，就确保 Qdrant collection 存在
            self.store.vector_size = self.embeddings.vector_size    # 需要设置 vector_size ，因为 Qdrant collection 需要知道向量维度
            self.store.ensure_collection(recreate=recreate)
        changed_documents = []  # 需要重新入库的 document 列表
        skipped_documents = 0
        deleted_chunks = 0
        for document in documents:
            document_hash = str(document.metadata.get("document_hash", ""))                                 # 拿到每个文档的 hash ，表示当前本地文档内容的 hash ，是 loader 读取当前文件时根据该文件内容算出来的哈希值
            existing_hashes = set() if recreate else self.store.source_content_hashes(document.source_path) # 查 collection 中这个 source 已有的 hash   source 代表原始文档来源（路径），一般是源文件路径
            if existing_hashes == {document_hash}:                                                          # 这里主要是想判断：这个文件这次读取到的内容，是否和以前入库时的内容是一样的？
                skipped_documents += 1                                                                      # 如果库里这个 source 的 hash 正好等于当前文件内容 hash ，说明该文件没有变化，不需要重新 chunk / embedding / rewrite into Qdrant
                continue                                                                                    # hash 相等就直接进入下一个循环，只有不等才会在循环内继续
            if existing_hashes:                                                                             # 不相等且库中已有当前 source 的 hash ，就要先删除旧的 chunks ，再把新文档加入待重新入库列表中
                deleted_chunks += self.store.delete_source(document.source_path)
            changed_documents.append(document)
        chunks = self.splitter.split_documents(changed_documents)
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        inserted = self.store.upsert_chunks(chunks, vectors, batch_size=batch_size)
        return {
            "collection": self.collection,
            "documents": len(documents),
            "changed_documents": len(changed_documents),
            "skipped_documents": skipped_documents,
            "chunks": len(chunks),
            "upserted": inserted,
            "deleted_chunks": deleted_chunks,
            "qdrant_path": str(self.settings.resolved_qdrant_path),
        }

    def reindex_source(self, path: str | Path, batch_size: int = 64) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()                                  # 路径标准化，expanduser() 处理 ~ ，resolve() 转成绝对路径
        deleted = self.delete_source(str(source))                                   # 删除这个 source 在 Qdrant 中已有的 chunks
        result = self.ingest_path(source, recreate=False, batch_size=batch_size)    # 重新读取这个路径下的文件并入库
        result["deleted_before_reindex"] = deleted
        return result

    def delete_source(self, source_path: str) -> int:
        return self.store.delete_source(source_path)

    def inspect_collection(self) -> dict[str, Any]:     # 返回 collection 状态，用于诊断
        return self.store.inspect_collection()

    def query(                                          # 问答入口，用户查询的主入口
        self,
        question: str,
        top_k: int | None = None,
        use_graph: bool = True,
        retriever_type: str | None = None,
        candidate_k: int | None = None,
    ) -> Answer:
        effective_top_k = top_k or self.settings.top_k
        effective_retriever = retriever_type or self.settings.retriever_type
        effective_candidate_k = candidate_k if candidate_k is not None else self.settings.candidate_k
        if use_graph:
            graph = build_rag_graph(                                        # graph = build_rag_graph() 把当前服务能力注入到 graph
                _ConfiguredRetriever(self, effective_retriever, effective_candidate_k),
                self.qa,
                list_sources=self.list_sources,                             # 列出有哪些源文件入库了，返回源文件路径列表
                chunks_for_source=self.chunks_for_source,                   # 按某个源文件路径取回它的 chunks
                quality_grader=self.quality_grader,
                max_rewrites=self.settings.max_rewrites,
                max_generation_retries=self.settings.max_generation_retries,
                min_relevance_score=self.settings.min_relevance_score,
                min_relevant_chunks=self.settings.min_relevant_chunks,
                min_grounded_overlap=self.settings.min_grounded_overlap,
            )
            state = graph.invoke(                                           # 执行图，graph.invoke(...) 返回最终 state
                {
                    "question": question,
                    "top_k": effective_top_k,
                    "retriever_type": effective_retriever,
                    "candidate_k": effective_candidate_k,
                }
            )
            answer = state["answer"]
            trace = dict(answer.trace)
            trace.setdefault("configured_retriever_type", effective_retriever)
            trace.setdefault("top_k", effective_top_k)
            trace.setdefault("candidate_k", effective_candidate_k)
            return Answer(
                question=answer.question,
                answer=answer.answer,
                citations=answer.citations,
                contexts=answer.contexts,
                trace=trace,
            )
        retrieved = self.retrieve_only(         # 不使用 graph ，只是用简单链路
            question,
            top_k=effective_top_k,
            retriever_type=effective_retriever,
            candidate_k=effective_candidate_k,
        )
        answer = self.qa.answer(question, retrieved)
        trace = dict(answer.trace)
        trace.update({"retriever_type": effective_retriever, "candidate_k": effective_candidate_k})
        return Answer(
            question=answer.question,
            answer=answer.answer,
            citations=answer.citations,
            contexts=answer.contexts,
            trace=trace,
        )

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
        retriever_type: str | None = None,
        candidate_k: int | None = None,
    ):
        return self.hybrid_retriever.retrieve(
            query=question,
            top_k=top_k or self.settings.top_k,
            retriever_type=retriever_type or self.settings.retriever_type,
            candidate_k=candidate_k if candidate_k is not None else self.settings.candidate_k,
        )

    def list_sources(self, limit: int | None = 200) -> list[str]:   # 列出有哪些源文件入库了，返回源文件路径列表
        return self.store.list_sources(limit=limit)

    def chunks_for_source(self, source_path: str, limit: int | None = 100) -> list[Chunk]:  # 按某个源文件路径取回它的 chunks
        return self.store.chunks_for_source(source_path=source_path, limit=limit)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()

"""
增量索引是基于 source_path + document_hash 做的 source 级增量更新。
每次 ingest 时先比较当前文档 hash 和 Qdrant 中已保存的 hash；没变就跳过，变了就删除该 source 的旧 chunks 并重新切块、embedding、upsert。
它避免了每次都重建整个索引，节省 embedding 成本和时间。
"""