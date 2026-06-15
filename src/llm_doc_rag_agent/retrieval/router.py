"""
根据 retriever_type 的不同来针对性选择 retrieve 方式

RRF     是融合候选，如果一个 chunk 同时被 dense 和 BM25 排得比较靠前，它会被奖励
RERANK  是利用 CrossEncoder 对 query 和检索得到的 chunk text 进行相关性打分
"""
from __future__ import annotations

from llm_doc_rag_agent.retrieval.bm25 import BM25Retriever
from llm_doc_rag_agent.retrieval.dense import DenseRetriever
from llm_doc_rag_agent.retrieval.reranker import NoOpReranker, Reranker
from llm_doc_rag_agent.schemas import RetrievedChunk


class HybridRetriever:
    """Route retrieval across dense, BM25, and simple RRF hybrid strategies."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker or NoOpReranker()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retriever_type: str = "dense",
        candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized = retriever_type.lower() 
        if normalized == "dense":
            return self.dense.retrieve(query=query, top_k=top_k)
        if normalized == "bm25":
            return self.bm25.retrieve(query=query, top_k=top_k, candidate_k=candidate_k)
        if normalized in {"hybrid", "hybrid_rrf"}:
            return self._hybrid_rrf(query=query, top_k=top_k, candidate_k=candidate_k or max(top_k * 4, 20))
        if normalized == "dense_rerank":
            candidates = self.dense.retrieve(query=query, top_k=candidate_k or max(top_k * 4, 20))
            return self._rerank(query=query, candidates=candidates, top_k=top_k, retriever_type="dense_rerank")
        if normalized == "hybrid_rerank":
            candidates = self._hybrid_rrf(query=query, top_k=candidate_k or max(top_k * 4, 20), candidate_k=candidate_k or max(top_k * 4, 20))
            return self._rerank(query=query, candidates=candidates, top_k=top_k, retriever_type="hybrid_rerank")
        raise ValueError(f"Unknown retriever_type: {retriever_type}")

    def _hybrid_rrf(self, query: str, top_k: int, candidate_k: int) -> list[RetrievedChunk]:    # 将 dense 检索结果和 BM25 检索结果 融合成一份最终结果
        dense_results = self.dense.retrieve(query=query, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(query=query, top_k=candidate_k, candidate_k=candidate_k)
        by_id: dict[str, RetrievedChunk] = {}   # 同一个 chunk 可能同时被 dense 和 BM25 检索得到，因此用 chunk_id --> RetrievedChunk or RRF_score 来去重
        scores: dict[str, float] = {}

        for results in (dense_results, bm25_results):   # 二元 tuple ，循环会跑两轮，第一轮处理 dense_results   第二轮处理 bm25_results
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.id
                by_id.setdefault(chunk_id, item)    # 如果 hy_id 中已有这个 chunk_id 就不覆盖，如果没有就保存 item
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)    # 如果该 chunk 之前没有分数就从 0 开始，如果之前已经在另一种检索结果中出现过就在已有分数上继续加

        fused = [
            RetrievedChunk(chunk=by_id[chunk_id].chunk, score=score, retriever_type="hybrid_rrf")
            for chunk_id, score in scores.items()
        ]
        return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
        retriever_type: str,
    ) -> list[RetrievedChunk]:
        results = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return [
            RetrievedChunk(chunk=item.chunk, score=item.score, retriever_type=retriever_type)
            for item in results
        ]
