from pathlib import Path

from llm_doc_rag_agent.config import Settings
from llm_doc_rag_agent.schemas import Answer
from llm_doc_rag_agent.service import RagService


class FakeEmbeddings:
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @property
    def vector_size(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += len(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0]


class FakeQA:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question, retrieved, generation_feedback=None):
        self.calls += 1
        retriever_types = sorted({item.retriever_type for item in retrieved}) or ["unknown"]
        contexts = [item.chunk.text for item in retrieved]
        return Answer(
            question=question,
            answer=contexts[0] if contexts else "fake answer",
            citations=[],
            contexts=contexts,
            trace={"model": "fake", "retriever_type": ",".join(retriever_types)},
        )


def _service(tmp_path: Path) -> RagService:
    settings = Settings(
        LLM_DOC_RAG_PROJECT_ROOT=tmp_path,
        LLM_DOC_RAG_QDRANT_PATH=tmp_path / "qdrant",
        LLM_DOC_RAG_COLLECTION="test",
        QUALITY_GRADER="rule",
    )
    service = RagService(settings=settings)
    service.embeddings = FakeEmbeddings()
    return service


def test_service_skips_unchanged_source_on_second_ingest(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("# Doc\n\nhello", encoding="utf-8")
    service = _service(tmp_path)

    first = service.ingest_path(source)
    second = service.ingest_path(source)

    assert first["changed_documents"] == 1
    assert second["changed_documents"] == 0
    assert second["skipped_documents"] == 1
    assert service.embeddings.document_calls == first["chunks"]


def test_service_reindexes_changed_source(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("hello", encoding="utf-8")
    service = _service(tmp_path)

    service.ingest_path(source)
    source.write_text("hello changed", encoding="utf-8")
    result = service.ingest_path(source)

    assert result["changed_documents"] == 1
    assert result["deleted_chunks"] >= 1
    assert service.delete_source(str(source.resolve())) >= 1


def test_service_retrieves_with_bm25(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("alpha semantic text\n\ninspect collection command", encoding="utf-8")
    service = _service(tmp_path)
    service.ingest_path(source)

    results = service.retrieve_only("inspect collection", retriever_type="bm25")

    assert results
    assert results[0].retriever_type == "bm25"
    assert "inspect collection" in results[0].chunk.text


def test_service_graph_lists_sources_without_embedding_or_llm(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("alpha semantic text\n\ninspect collection command", encoding="utf-8")
    service = _service(tmp_path)
    service.ingest_path(source)
    service.embeddings.query_calls = 0
    fake_qa = FakeQA()
    service._qa = fake_qa

    answer = service.query("请列出当前有哪些文档", use_graph=True)

    assert str(source.resolve()) in answer.answer
    assert answer.trace["route"] == "source_lookup"
    assert answer.trace["graph_path"] == ["route_question", "source_lookup"]
    assert answer.trace["retrieval_skipped"] is True
    assert service.embeddings.query_calls == 0
    assert fake_qa.calls == 0


def test_service_graph_retrieves_and_generates_for_normal_question(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("alpha semantic text\n\ninspect collection command", encoding="utf-8")
    service = _service(tmp_path)
    service.ingest_path(source)
    service.embeddings.query_calls = 0
    fake_qa = FakeQA()
    service._qa = fake_qa

    answer = service.query("How does inspect collection work?", use_graph=True)

    assert "inspect collection command" in answer.answer
    assert answer.trace["route"] == "retrieve_rag"
    assert answer.trace["graph_path"] == [
        "route_question",
        "retrieve",
        "grade_documents",
        "generate",
        "grade_generation",
    ]
    assert answer.trace["retriever_type"] == "dense"
    assert answer.trace["document_grade_decision"] == "accept"
    assert "answer_grounded" in answer.trace
    assert service.embeddings.query_calls == 1
    assert fake_qa.calls == 1


def test_service_graph_inspects_chunks_for_matched_source(tmp_path: Path):
    source = tmp_path / "doc.md"
    source.write_text("alpha semantic text\n\ninspect collection command", encoding="utf-8")
    service = _service(tmp_path)
    service.ingest_path(source)
    fake_qa = FakeQA()
    service._qa = fake_qa

    answer = service.query(f"查看 source={source.resolve()} 的 chunks", use_graph=True)

    assert answer.trace["route"] == "source_lookup"
    assert answer.trace["source_path"] == str(source.resolve())
    assert answer.trace["chunk_count"] >= 1
    assert answer.citations
    assert "chunk 0" in answer.answer
    assert fake_qa.calls == 0
