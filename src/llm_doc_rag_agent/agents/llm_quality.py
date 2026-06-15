from __future__ import annotations

import json
from typing import Any

from llm_doc_rag_agent.agents.quality import (
    AnswerGrade,
    DocumentGrade,
    grade_answer,
    grade_retrieved_documents,
    rewrite_query,
)
from llm_doc_rag_agent.schemas import RetrievedChunk
from llm_doc_rag_agent.utils import safe_snippet


class RuleBasedQualityGrader:
    """Deterministic quality grader used as the default and fallback."""

    name = "rule"

    def grade_documents(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        min_relevance_score: float = 0.05,
        min_relevant_chunks: int = 1,
    ) -> DocumentGrade:
        return grade_retrieved_documents(
            query=query,
            retrieved=retrieved,
            min_relevance_score=min_relevance_score,
            min_relevant_chunks=min_relevant_chunks,
        )

    def rewrite_query(self, question: str, previous_query: str | None = None) -> str:
        return rewrite_query(question, previous_query=previous_query)

    def grade_answer(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        min_grounded_overlap: float = 0.2,
    ) -> AnswerGrade:
        return grade_answer(
            question=question,
            answer=answer,
            contexts=contexts,
            min_grounded_overlap=min_grounded_overlap,
        )


class HybridLLMQualityGrader(RuleBasedQualityGrader):
    """LLM-as-judge grader with deterministic rule fallback."""

    name = "hybrid"

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        max_tokens: int | None = 4096,  # 表示模型每次最多可以生成多少 token ，输出长度上线
        top_p: float | None = 0.1,      # 控制模型从多大概率范围内挑选下一个 token  越小表示从概率最高的一小撮 token 中选取，更加保守稳定
        disable_thinking: bool = True,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.disable_thinking = disable_thinking
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing API key for hybrid quality grader.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def grade_documents(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        min_relevance_score: float = 0.05,
        min_relevant_chunks: int = 1,
    ) -> DocumentGrade:
        fallback = super().grade_documents(         # 先调用父类 RuleBaseQualityGrader.grade_documents()
            query=query,
            retrieved=retrieved,
            min_relevance_score=min_relevance_score,
            min_relevant_chunks=min_relevant_chunks,
        )
        if not retrieved or not self.api_key:       # 如果没有检索结果或者没有 API key 就直接返回规则判断结果
            return fallback

        candidates = "\n\n".join(                   # 把检索结果整理成给 LLM 看的文本
            (
                f"Index: {index}\n"
                f"Source: {item.chunk.source_path}\n"
                f"Chunk: {item.chunk.chunk_index}\n"
                f"Retriever score: {item.score:.4f}\n"
                f"Text: {safe_snippet(item.chunk.text, limit=1200)}"
            )
            for index, item in enumerate(retrieved)
        )
        payload = self._call_json(      # 调用 LLM 并返回 JSON 信息
            system=(
                "You are a strict RAG document relevance judge. "
                "Return JSON only. Mark a chunk relevant only if it can help answer the query."
            ),
            user=(
                "Decide which retrieved chunks are relevant to the query.\n"
                'Return JSON shaped like {"relevant_indices":[0,2],"reason":"..."}.\n'
                "Use zero-based candidate indices.\n\n"
                f"Query:\n{query}\n\nCandidates:\n{candidates}"
            ),
        )
        if not payload:
            return fallback

        relevant_indices = self._relevant_indices(payload, len(retrieved))      # 从 LLM 返回的 JSON 里提取合法的相关 chunk 下标
        filtered = [retrieved[index] for index in relevant_indices]
        decision = "accept" if len(filtered) >= min_relevant_chunks else "rewrite"
        reason = str(
            payload.get("reason")
            or ("llm_relevant_context_found" if decision == "accept" else "llm_context_below_relevance_threshold")
        )
        return DocumentGrade(
            filtered_documents=filtered,
            relevant_count=len(filtered),
            retrieved_count=len(retrieved),
            max_score=fallback.max_score,
            query_terms=fallback.query_terms,
            decision=decision,
            reason=reason,
        )

    def rewrite_query(self, question: str, previous_query: str | None = None) -> str:
        fallback = super().rewrite_query(question, previous_query=previous_query)
        if not self.api_key:
            return fallback

        payload = self._call_json(
            system=(
                "You rewrite RAG retrieval queries for technical documentation. "
                "Return JSON only."
            ),
            user=(
                "Rewrite the question into a concise search query for retrieving code and documentation chunks.\n"
                'Return JSON shaped like {"query":"..."}.\n\n'
                f"Original question:\n{question}\n\nPrevious search query:\n{previous_query or question}"
            ),
        )
        if not payload:
            return fallback
        rewritten = str(payload.get("query") or payload.get("rewritten_query") or "").strip()
        return rewritten or fallback

    def grade_answer(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        min_grounded_overlap: float = 0.2,
    ) -> AnswerGrade:
        fallback = super().grade_answer(
            question=question,
            answer=answer,
            contexts=contexts,
            min_grounded_overlap=min_grounded_overlap,
        )
        if not answer.strip() or not contexts or not self.api_key:
            return fallback

        context = "\n\n".join(
            f"[{index}] {safe_snippet(text, limit=1200)}" for index, text in enumerate(contexts, start=1)
        )
        payload = self._call_json(
            system=(
                "You are a strict RAG answer judge. Return JSON only. "
                "Judge whether the answer is grounded in the context and answers the question."
            ),
            user=(
                'Return JSON shaped like {"grounded":true,"relevant":true,"reason":"..."}.\n\n'
                f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
            ),
        )
        if not payload:
            return fallback

        return AnswerGrade(
            grounded=self._coerce_bool(payload.get("grounded"), fallback.grounded),
            relevant=self._coerce_bool(payload.get("relevant"), fallback.relevant),
            grounded_overlap_ratio=fallback.grounded_overlap_ratio,
            answer_question_overlap_ratio=fallback.answer_question_overlap_ratio,
            answer_terms=fallback.answer_terms,
            context_terms=fallback.context_terms,
            question_terms=fallback.question_terms,
        )

    def _call_json(self, system: str, user: str) -> dict[str, Any] | None:  # 调用 LLM 并解析 JSON
        try:
            request: dict[str, Any] = {     # 构造请求
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,     # 希望输出稳定、减少随机性
            }
            if self.max_tokens is not None:
                request["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                request["top_p"] = self.top_p
            if self.disable_thinking:
                request["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self.client.chat.completions.create(**request)   # 发起 cha completion 请求
            content = response.choices[0].message.content or ""
            return _parse_json_object(content)  # 从 LLM 的返回内容中提取 JSON 信息
        except Exception:
            return None

    @staticmethod   # 静态方法，不依赖实例状态
    def _relevant_indices(payload: dict[str, Any], count: int) -> list[int]:    # 清洗 LLM 返回的 chunk 下标
        raw_indices = payload.get("relevant_indices") or payload.get("indices") or []   # 优先读取 LLM 返回的 relevant_indices ，也兼容 indices
        if not raw_indices and isinstance(payload.get("chunks"), list):
            raw_indices = [     # 如果 LLM 没有返回 relevant_indices ，但是返回类似 {"chunks":[{"index":0,"relevant":true},]} 的内容，就从 chunks 中提取 relevant = true 的 index
                item.get("index")
                for item in payload["chunks"]
                if isinstance(item, dict) and HybridLLMQualityGrader._coerce_bool(item.get("relevant"), False)
            ]
        indices: list[int] = []
        for value in raw_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < count and index not in indices: # 跳过越界 index 并去重，保持原顺序
                indices.append(index)
        return indices

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:    # 把 LLM 输出转为 bool value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y"}:
                return True
            if normalized in {"false", "no", "n"}:
                return False
        return default  # default 是规则 fallback 的结果


def build_quality_grader(   # 根据配置创建评分器    工厂函数，外部不用自己判断要实例化哪个类，只要传 mode
    mode: str,
    api_key: str | None,
    base_url: str,
    model: str,
    max_tokens: int | None = 4096,
    top_p: float | None = 0.1,
    disable_thinking: bool = True,
) -> RuleBasedQualityGrader:
    normalized = mode.strip().lower()
    if normalized in {"hybrid", "llm", "llm_hybrid"}:
        return HybridLLMQualityGrader(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            disable_thinking=disable_thinking,
        )
    return RuleBasedQualityGrader()


def _parse_json_object(text: str) -> dict[str, Any] | None:     # 从 LLM 文本中提取 JSON
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None   # 只接受 JSON object
