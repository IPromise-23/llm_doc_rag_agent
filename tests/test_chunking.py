from llm_doc_rag_agent.chunking import SimpleTextSplitter
from llm_doc_rag_agent.schemas import Document


def test_splitter_preserves_source_and_metadata():
    splitter = SimpleTextSplitter(chunk_size=40, chunk_overlap=5)
    doc = Document(
        text="alpha beta gamma\n\n" + "delta " * 20,
        source_path="/tmp/example.md",
        metadata={"file_type": "md"},
    )

    chunks = splitter.split_document(doc)

    assert chunks
    assert all(chunk.source_path == "/tmp/example.md" for chunk in chunks)
    assert all(chunk.metadata["file_type"] == "md" for chunk in chunks)
    assert chunks[0].id


def test_splitter_rejects_invalid_overlap():
    try:
        SimpleTextSplitter(chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")
