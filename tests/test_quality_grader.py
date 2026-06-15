from llm_doc_rag_agent.agents.llm_quality import HybridLLMQualityGrader, RuleBasedQualityGrader
from llm_doc_rag_agent.schemas import Chunk, RetrievedChunk


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responses[self.calls]
        self.calls += 1
        return _FakeResponse(content)


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)


def _retrieved(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            source_path=f"/tmp/{chunk_id}.md",
            chunk_index=0,
            content_hash=f"hash-{chunk_id}",
        ),
        score=0.01,
        retriever_type="dense",
    )


def test_rule_based_quality_grader_rewrites_without_llm():
    grader = RuleBasedQualityGrader()

    rewritten = grader.rewrite_query("How does Qdrant collection configuration work?")

    assert "qdrant" in rewritten
    assert grader.name == "rule"


def test_hybrid_quality_grader_uses_llm_document_indices():
    grader = HybridLLMQualityGrader(api_key="test", base_url="https://example.test", model="judge")
    grader._client = _FakeClient(['{"relevant_indices":[1],"reason":"second chunk matches"}'])
    candidates = [_retrieved("a", "unrelated"), _retrieved("b", "collection configuration details")]

    grade = grader.grade_documents("How is collection configured?", candidates, min_relevant_chunks=1)

    assert grade.decision == "accept"
    assert grade.filtered_documents == [candidates[1]]
    assert grade.reason == "second chunk matches"
    request = grader._client.chat.completions.requests[0]
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 4096
    assert request["top_p"] == 0.1
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_hybrid_quality_grader_uses_llm_answer_grade():
    grader = HybridLLMQualityGrader(api_key="test", base_url="https://example.test", model="judge")
    grader._client = _FakeClient(['{"grounded":false,"relevant":true,"reason":"unsupported"}'])

    grade = grader.grade_answer(
        question="How is collection configured?",
        answer="Use an unsupported setting.",
        contexts=["Qdrant collections require vector size and distance."],
    )

    assert grade.grounded is False
    assert grade.relevant is True
