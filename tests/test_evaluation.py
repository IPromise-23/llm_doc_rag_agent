from pathlib import Path

from llm_doc_rag_agent.config import Settings
from llm_doc_rag_agent.evaluation import EvalRunner, RagasEvalRunner, RetrievalEvalRunner
from llm_doc_rag_agent.schemas import Answer, Chunk, RetrievedChunk


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeService:
    def __init__(self, tmp_path: Path) -> None:
        self.settings = Settings(
            LLM_DOC_RAG_PROJECT_ROOT=tmp_path,
            TOP_K=3,
            EVAL_RETRIEVERS=["dense"],
        )
        self.embeddings = FakeEmbeddings()
        self.calls: list[str] = []
        self.graph_calls: list[bool] = []
        self.retrieve_calls: list[str] = []

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_graph: bool = True,
        retriever_type: str | None = None,
        candidate_k: int | None = None,
    ) -> Answer:
        active_retriever = retriever_type or self.settings.retriever_type
        self.calls.append(active_retriever)
        self.graph_calls.append(use_graph)
        return Answer(
            question=question,
            answer=f"{active_retriever} answer",
            citations=[],
            contexts=[f"{active_retriever} context"],
            trace={"retriever_type": active_retriever},
        )

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
        retriever_type: str | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:
        active_retriever = retriever_type or self.settings.retriever_type
        self.retrieve_calls.append(active_retriever)
        chunk = Chunk(
            id=f"{active_retriever}-chunk",
            text=f"{active_retriever} context",
            source_path=f"/tmp/project/docs/{active_retriever}.md",
            chunk_index=0,
            content_hash="hash",
        )
        return [RetrievedChunk(chunk=chunk, score=0.9, retriever_type=active_retriever)]


def test_eval_runner_compares_multiple_retrievers(tmp_path: Path):
    dataset = tmp_path / "questions.csv"
    dataset.write_text("question,ground_truth\nHow?,Truth\n", encoding="utf-8")
    output = tmp_path / "runs" / "result.jsonl"
    service = FakeService(tmp_path)

    results = EvalRunner(service).run(
        dataset_path=dataset,
        output_path=output,
        retrievers=["dense", "bm25"],
        candidate_k=8,
    )

    assert service.calls == ["dense", "bm25"]
    assert service.graph_calls == [True, True]
    assert [result.trace["retriever_type"] for result in results] == ["dense", "bm25"]
    assert all(result.trace["candidate_k"] == 8 for result in results)
    assert all(result.trace["eval_layer"] == "rag" for result in results)
    assert all(result.trace["use_graph"] is True for result in results)
    assert output.exists()
    assert EvalRunner(service).default_report_path(output) == output.with_suffix(".md")


def test_eval_runner_can_bypass_graph_explicitly(tmp_path: Path):
    dataset = tmp_path / "questions.csv"
    dataset.write_text("question,ground_truth\nHow?,Truth\n", encoding="utf-8")
    service = FakeService(tmp_path)

    EvalRunner(service).run(dataset_path=dataset, output_path=tmp_path / "result.jsonl", use_graph=False)

    assert service.graph_calls == [False]


def test_retrieval_eval_runner_does_not_call_generation(tmp_path: Path):
    dataset = tmp_path / "questions.csv"
    dataset.write_text(
        "question,ground_truth,expected_sources,category,answerable\n"
        "How?,Truth,docs/bm25.md,lexical,true\n",
        encoding="utf-8",
    )
    output = tmp_path / "retrieval.csv"
    report = tmp_path / "retrieval.md"
    service = FakeService(tmp_path)
    runner = RetrievalEvalRunner(service)

    rows = runner.run(
        dataset_path=dataset,
        output_path=output,
        retrievers=["bm25"],
        candidate_k=8,
    )
    runner.write_report(rows, report)

    assert service.calls == []
    assert service.retrieve_calls == ["bm25"]
    assert rows[0]["eval_layer"] == "retrieval"
    assert rows[0]["hit"] is True
    assert rows[0]["first_hit_rank"] == 1
    assert output.exists()
    text = report.read_text(encoding="utf-8")
    assert "# Retrieval Evaluation Report" in text
    assert "does not call the LLM generation layer" in text


def test_eval_runner_writes_retriever_type_to_csv(tmp_path: Path):
    output = tmp_path / "result.csv"
    service = FakeService(tmp_path)
    answer = service.query("How?", retriever_type="bm25")

    EvalRunner(service).write_results(
        [
            service_result
            for service_result in [
                type(
                    "Result",
                    (),
                    {
                        "question": answer.question,
                        "answer": answer.answer,
                        "ground_truth": None,
                        "contexts": answer.contexts,
                        "citations": answer.citations,
                        "trace": answer.trace,
                    },
                )()
            ]
        ],
        output,
    )

    assert "retriever_type" in output.read_text(encoding="utf-8")
    assert "bm25" in output.read_text(encoding="utf-8")


def test_eval_runner_writes_markdown_report(tmp_path: Path):
    dataset = tmp_path / "questions.csv"
    dataset.write_text("question,ground_truth\nHow?,Truth\n", encoding="utf-8")
    output = tmp_path / "result.jsonl"
    report = tmp_path / "result.md"
    service = FakeService(tmp_path)
    runner = EvalRunner(service)
    results = runner.run(dataset_path=dataset, output_path=output, retrievers=["dense", "bm25"])

    runner.write_report(results, report)
    text = report.read_text(encoding="utf-8")

    assert "# RAG Evaluation Report" in text
    assert "| dense | 1 |" in text
    assert "| bm25 | 1 |" in text
    assert "dense answer" in text
    assert "bm25 answer" in text
    assert "## Quality Diagnostics" in text
    assert "These are deterministic lexical diagnostics" in text
    assert "Truth" in text
    assert "low_answer_ground_truth_coverage" in text


def test_ragas_eval_runner_scores_existing_eval_results(tmp_path: Path, monkeypatch):
    class FakeEvaluationDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeRagasResult:
        scores = [{"faithfulness": 0.9, "answer_relevancy": 0.8}]

    def fake_evaluate(dataset, metrics, llm, embeddings, raise_exceptions, show_progress):
        assert dataset[0]["user_input"] == "How?"
        assert metrics == ["faithfulness", "answer_relevancy"]
        assert llm == "fake-llm"
        assert raise_exceptions is False
        assert show_progress is False
        return FakeRagasResult()

    monkeypatch.setattr(
        RagasEvalRunner,
        "_load_ragas",
        lambda self, metric_names: (fake_evaluate, FakeEvaluationDataset, list(metric_names)),
    )
    monkeypatch.setattr(RagasEvalRunner, "_make_llm", lambda self: "fake-llm")

    dataset = tmp_path / "questions.csv"
    dataset.write_text("question,ground_truth\nHow?,Truth\n", encoding="utf-8")
    output = tmp_path / "ragas.csv"
    service = FakeService(tmp_path)
    result = RagasEvalRunner(service).run(
        dataset_path=dataset,
        output_path=output,
        retrievers=["dense"],
        metrics=["faithfulness", "answer_relevancy"],
    )

    assert result.examples == 1
    assert result.metrics == ["faithfulness", "answer_relevancy"]
    assert output.exists()
    assert result.raw_output_path and result.raw_output_path.exists()
    assert result.report_path and result.report_path.exists()
    text = output.read_text(encoding="utf-8")
    assert "faithfulness" in text
    assert "0.9" in text
    assert "# RAGAS Evaluation Report" in result.report_path.read_text(encoding="utf-8")
