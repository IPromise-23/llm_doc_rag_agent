"""
二阶排序，dense BM25 hybrid 已经先找出一批候选结果，reranker 不负责从全库中检索，只负责对这些候选 chunk 重新打分、排序  调用 CrossEncoder 模型来预测 query 和 chunk text 的相关性分数
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from llm_doc_rag_agent.schemas import RetrievedChunk


class Reranker(ABC):
    @abstractmethod # 子类必须实现它
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


class CrossEncoderReranker(Reranker):
    """Lazy CrossEncoder reranker. Only instantiated when rerank strategies are used."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name    # 保存模型名，比如 "BAAI/bag-reranker-base"
        self._model = None              # 懒加载

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]   # 将 query 与每一个候选 chunk 的正文内容匹配
        scores = self.model.predict(pairs)
        reranked = [    # 需要重新创建 RetrievedChunk   保留原 chunk    换掉分数    标记来源
            RetrievedChunk(chunk=item.chunk, score=float(score), retriever_type=f"{item.retriever_type}_rerank")
            for item, score in zip(candidates, scores, strict=True) # 每个 item 是一个检索结果对象 RetrievedChunk ，里面包括 chunk score retriever_type
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
