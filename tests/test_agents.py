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
        self.feedback: list[str | None] = []

    def answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        generation_feedback: str | None = None,
    ) -> Answer:
        self.calls += 1
        self.feedback.append(generation_feedback)
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
        self.answer_decisions = ["accept"]
        self.answer_grade_calls = 0

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

        index = min(self.answer_grade_calls, len(self.answer_decisions) - 1)
        decision = self.answer_decisions[index]
        self.answer_grade_calls += 1
        grounded = decision == "accept"
        relevant = decision != "rewrite_query"
        return AnswerGrade(
            grounded=grounded,
            relevant=relevant,
            grounded_overlap_ratio=1.0 if grounded else 0.0,
            answer_question_overlap_ratio=1.0 if relevant else 0.0,
            answer_terms=[],
            context_terms=[],
            question_terms=[],
            decision=decision,
            reason=f"fake_{decision}",
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
    assert answer.trace["generation_grade_decision"] == "accept"


def test_graph_regenerates_when_generation_judge_requests_answer_retry():
    retriever = FakeRetriever([[_retrieved("fake regenerated context")]])
    qa = FakeQA()
    grader = FakeQualityGrader()
    grader.answer_decisions = ["regenerate", "accept"]
    graph = build_rag_graph(
        retriever,
        qa,
        quality_grader=grader,
        max_rewrites=1,
        max_generation_retries=1,
    )

    state = graph.invoke({"question": "Original question?", "top_k": 1})
    answer = state["answer"]

    assert retriever.queries == ["Original question?"]
    assert qa.calls == 2
    assert qa.feedback[0] is None
    assert "fake_regenerate" in qa.feedback[1]
    assert answer.trace["generation_retry_count"] == 1
    assert answer.trace["generation_grade_decision"] == "accept"
    assert answer.trace["final_decision"] == "generated"
    assert answer.trace["graph_path"] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "generate",
        "grade_generation",
        "regenerate_answer",
        "generate",
        "grade_generation",
    ]


def test_graph_rewrites_query_when_generation_judge_requests_new_retrieval():
    retriever = FakeRetriever(
        [
            [_retrieved("first context")],
            [_retrieved("fake rewritten context")],
        ]
    )
    qa = FakeQA()
    grader = FakeQualityGrader()
    grader.answer_decisions = ["rewrite_query", "accept"]
    graph = build_rag_graph(
        retriever,
        qa,
        quality_grader=grader,
        max_rewrites=1,
        max_generation_retries=1,
    )

    state = graph.invoke({"question": "Original question?", "top_k": 1})
    answer = state["answer"]

    assert retriever.queries == ["Original question?", "fake rewritten query"]
    assert grader.rewrite_calls == 1
    assert qa.calls == 2
    assert answer.trace["rewrite_count"] == 1
    assert answer.trace["generation_grade_decision"] == "accept"
    assert answer.trace["final_decision"] == "generated"
    assert answer.trace["graph_path"] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "generate",
        "grade_generation",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "generate",
        "grade_generation",
    ]
