from llm_doc_rag_agent.agents import build_rag_graph, extract_source_hint, route_question
from llm_doc_rag_agent.schemas import Answer, Chunk, RetrievedChunk


def test_route_question_detects_source_lookup():
    assert route_question("请列出当前有哪些文档") == "source_lookup"
    assert route_question("show sources") == "source_lookup"
    assert route_question("这个项目如何做检索？") == "retrieve_rag"


def test_extract_source_hint_from_explicit_source():
    assert extract_source_hint("查看 source=/tmp/doc.md 的 chunks") == "/tmp/doc.md"
    assert extract_source_hint("这个文件 `notes/sample.ipynb` 讲了什么？") == "notes/sample.ipynb"


class FakeRetriever:
    def __init__(self, results_by_call: list[list[RetrievedChunk]]) -> None:
        self.results_by_call = results_by_call
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.queries.append(query)
        index = min(len(self.queries) - 1, len(self.results_by_call) - 1)
        return self.results_by_call[index][:top_k]


class FakeQA:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        self.calls += 1
        contexts = [item.chunk.text for item in retrieved]
        return Answer(
            question=question,
            answer=contexts[0] if contexts else "fake answer",
            citations=[],
            contexts=contexts,
            trace={"model": "fake"},
        )


class FakeQualityGrader:
    name = "fake_quality"

    def __init__(self) -> None:
        self.rewrite_calls = 0

    def grade_documents(self, query, retrieved, min_relevance_score=0.05, min_relevant_chunks=1):
        from llm_doc_rag_agent.agents.quality import DocumentGrade

        return DocumentGrade(
            filtered_documents=retrieved,
            relevant_count=len(retrieved),
            retrieved_count=len(retrieved),
            max_score=max((item.score for item in retrieved), default=0.0),
            query_terms=[query],
            decision="accept" if retrieved else "rewrite",
            reason="fake_quality_decision",
        )

    def rewrite_query(self, question, previous_query=None):
        self.rewrite_calls += 1
        return "fake rewritten query"

    def grade_answer(self, question, answer, contexts, min_grounded_overlap=0.2):
        from llm_doc_rag_agent.agents.quality import AnswerGrade

        return AnswerGrade(
            grounded=True,
            relevant=True,
            grounded_overlap_ratio=1.0,
            answer_question_overlap_ratio=1.0,
            answer_terms=[],
            context_terms=[],
            question_terms=[],
        )


def _retrieved(text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="chunk-1",
            text=text,
            source_path="/tmp/doc.md",
            chunk_index=0,
            content_hash="hash",
        ),
        score=score,
        retriever_type="dense",
    )


def test_graph_rewrites_once_when_documents_are_not_relevant():
    retriever = FakeRetriever([[], [_retrieved("missing work details")]])
    qa = FakeQA()
    graph = build_rag_graph(retriever, qa, max_rewrites=1)

    state = graph.invoke({"question": "How does missing work?", "top_k": 1})
    answer = state["answer"]

    assert retriever.queries == ["How does missing work?", "missing work"]
    assert qa.calls == 1
    assert answer.answer == "missing work details"
    assert answer.trace["rewrite_count"] == 1
    assert answer.trace["final_decision"] == "generated"
    assert answer.trace["graph_path"] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "generate",
        "grade_generation",
    ]


def test_graph_returns_insufficient_context_after_rewrite_budget():
    retriever = FakeRetriever([[]])
    qa = FakeQA()
    graph = build_rag_graph(retriever, qa, max_rewrites=0)

    state = graph.invoke({"question": "How does missing work?", "top_k": 1})
    answer = state["answer"]

    assert qa.calls == 0
    assert answer.trace["final_decision"] == "insufficient_context"
    assert answer.trace["graph_path"] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "insufficient_context",
    ]


def test_graph_uses_injected_quality_grader_for_rewrite_and_answer_grade():
    retriever = FakeRetriever([[], [_retrieved("fake rewritten context")]])
    qa = FakeQA()
    grader = FakeQualityGrader()
    graph = build_rag_graph(retriever, qa, quality_grader=grader, max_rewrites=1)

    state = graph.invoke({"question": "Original question?", "top_k": 1})
    answer = state["answer"]

    assert retriever.queries == ["Original question?", "fake rewritten query"]
    assert grader.rewrite_calls == 1
    assert answer.trace["quality_grader"] == "fake_quality"
    assert answer.trace["answer_grounded"] is True
