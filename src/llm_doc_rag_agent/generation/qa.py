from __future__ import annotations

from openai import OpenAI

from llm_doc_rag_agent.schemas import Answer, Citation, RetrievedChunk
from llm_doc_rag_agent.utils import safe_snippet


class QAService:
    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY. Fill .env before running LLM generation.")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        contexts = [item.chunk.text for item in retrieved]
        prompt = self._build_prompt(question, retrieved)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise technical-document RAG assistant. "
                        "Answer only from the supplied context. If the context is insufficient, say so."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        citations = [       # Citation 展示阶段的精简引用   RetrievedChunk 检索阶段的完整结果
            Citation(
                source_path=item.chunk.source_path,
                chunk_id=item.chunk.id,
                chunk_index=item.chunk.chunk_index,
                score=item.score,
                snippet=safe_snippet(item.chunk.text),  # 把完整的 chunk text 文本截成一小段摘要    citation 是展示用的，safe_snippet() 会压缩空白、截断长度，让引用列表更轻
            )
            for item in retrieved
        ]
        retriever_types = sorted({item.retriever_type for item in retrieved}) or ["unknown"]    # 防御性汇总，从 retrieved 的每个结果中收集 retriever_type  当下确实只有一种检索类型，但是如果想要保留原始来源或者需要测试、调试，那就可能直接把不同 retriever 的结果拼接到一起传给 QA
        return Answer(
            question=question,
            answer=text.strip(),
            citations=citations,    # 正式答案结构的一部分，给前端、CLI 展示引用来源
            contexts=contexts,
            trace={                 # 调试信息
                "model": self.model,
                "retriever_type": ",".join(retriever_types),
                "context_count": len(contexts),
            },
        )

    def _build_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str: # 建立提示词 
        context_blocks = []
        for index, item in enumerate(retrieved, start=1):   # index 是从 1 开始的数字   item 是 RetrievedChunk 的实例对象
            chunk = item.chunk
            context_blocks.append(
                f"[{index}] source={chunk.source_path} chunk={chunk.chunk_index} score={item.score:.4f}\n{chunk.text}"
            )
        context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
        return (
            "Use the following document chunks to answer the question.\n"
            "Cite sources by mentioning their source path and chunk number when useful.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
