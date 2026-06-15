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


class _ConfiguredRetriever:
    """Late-bound retriever adapter so graph routing can skip retrieval entirely."""

    def __init__(self, service: "RagService", retriever_type: str, candidate_k: int | None) -> None:
        self.service = service
        self.retriever_type = retriever_type
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5):
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

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
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

    def ingest_path(self, path: str | Path, recreate: bool = False, batch_size: int = 64) -> dict[str, Any]:
        documents = self.loader.load_path(path)
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
        if recreate or not self.store.collection_exists():
            self.store.vector_size = self.embeddings.vector_size
            self.store.ensure_collection(recreate=recreate)
        changed_documents = []
        skipped_documents = 0
        deleted_chunks = 0
        for document in documents:
            document_hash = str(document.metadata.get("document_hash", ""))
            existing_hashes = set() if recreate else self.store.source_content_hashes(document.source_path)
            if existing_hashes == {document_hash}:
                skipped_documents += 1
                continue
            if existing_hashes:
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
        source = Path(path).expanduser().resolve()
        deleted = self.delete_source(str(source))
        result = self.ingest_path(source, recreate=False, batch_size=batch_size)
        result["deleted_before_reindex"] = deleted
        return result

    def delete_source(self, source_path: str) -> int:
        return self.store.delete_source(source_path)

    def inspect_collection(self) -> dict[str, Any]:
        return self.store.inspect_collection()

    def query(
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
            graph = build_rag_graph(
                _ConfiguredRetriever(self, effective_retriever, effective_candidate_k),
                self.qa,
                list_sources=self.list_sources,
                chunks_for_source=self.chunks_for_source,
                quality_grader=self.quality_grader,
                max_rewrites=self.settings.max_rewrites,
                min_relevance_score=self.settings.min_relevance_score,
                min_relevant_chunks=self.settings.min_relevant_chunks,
                min_grounded_overlap=self.settings.min_grounded_overlap,
            )
            state = graph.invoke(
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
        retrieved = self.retrieve_only(
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

    def list_sources(self, limit: int | None = 200) -> list[str]:
        return self.store.list_sources(limit=limit)

    def chunks_for_source(self, source_path: str, limit: int | None = 100) -> list[Chunk]:
        return self.store.chunks_for_source(source_path=source_path, limit=limit)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
