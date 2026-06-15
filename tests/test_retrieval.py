from llm_doc_rag_agent.retrieval import BM25Retriever, HybridRetriever
from llm_doc_rag_agent.schemas import Chunk, RetrievedChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source_path=f"/tmp/{chunk_id}.md",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
    )


class FakeStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def list_chunks(self, limit: int | None = None) -> list[Chunk]:
        return self.chunks[:limit] if limit is not None else list(self.chunks)


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.results[:top_k]


class ReverseReranker:
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(reversed(candidates))[:top_k]


def test_bm25_retriever_ranks_exact_terms_first():
    chunks = [
        _chunk("a", "semantic vector embedding search"),
        _chunk("b", "delete-source command removes one source"),
    ]
    retriever = BM25Retriever(store=FakeStore(chunks))

    results = retriever.retrieve("delete source", top_k=2)

    assert results[0].chunk.id == "b"
    assert results[0].retriever_type == "bm25"


def test_bm25_retriever_handles_chinese_terms():
    chunks = [
        _chunk("a", "这个模块负责向量检索"),
        _chunk("b", "这个命令用于删除文档来源"),
    ]
    retriever = BM25Retriever(store=FakeStore(chunks))

    results = retriever.retrieve("删除来源", top_k=1)

    assert results[0].chunk.id == "b"


def test_hybrid_retriever_fuses_dense_and_bm25_rankings():
    dense_first = _chunk("dense", "dense semantic match")
    bm25_first = _chunk("lexical", "inspect collection command")
    dense = FakeDenseRetriever([RetrievedChunk(dense_first, 0.9), RetrievedChunk(bm25_first, 0.1)])
    bm25 = BM25Retriever(store=FakeStore([bm25_first, dense_first]))
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    results = retriever.retrieve("inspect collection", top_k=2, retriever_type="hybrid_rrf", candidate_k=2)

    assert {item.chunk.id for item in results} == {"dense", "lexical"}
    assert all(item.retriever_type == "hybrid_rrf" for item in results)


def test_hybrid_retriever_dense_rerank_uses_candidate_set():
    first = _chunk("first", "first dense result")
    second = _chunk("second", "second dense result")
    dense = FakeDenseRetriever([RetrievedChunk(first, 0.9), RetrievedChunk(second, 0.8)])
    bm25 = BM25Retriever(store=FakeStore([]))
    retriever = HybridRetriever(dense=dense, bm25=bm25, reranker=ReverseReranker())

    results = retriever.retrieve("anything", top_k=1, retriever_type="dense_rerank", candidate_k=2)

    assert results[0].chunk.id == "second"
    assert results[0].retriever_type == "dense_rerank"


def test_hybrid_retriever_hybrid_rerank_uses_rrf_candidates():
    dense_first = _chunk("dense", "dense semantic match")
    bm25_first = _chunk("lexical", "inspect collection command")
    dense = FakeDenseRetriever([RetrievedChunk(dense_first, 0.9), RetrievedChunk(bm25_first, 0.1)])
    bm25 = BM25Retriever(store=FakeStore([bm25_first, dense_first]))
    retriever = HybridRetriever(dense=dense, bm25=bm25, reranker=ReverseReranker())

    results = retriever.retrieve("inspect collection", top_k=2, retriever_type="hybrid_rerank", candidate_k=2)

    assert {item.chunk.id for item in results} == {"dense", "lexical"}
    assert all(item.retriever_type == "hybrid_rerank" for item in results)
