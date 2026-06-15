from pathlib import Path

from llm_doc_rag_agent.schemas import Chunk
from llm_doc_rag_agent.vectorstores import QdrantVectorStore


def _chunk(source: str, index: int) -> Chunk:
    return Chunk(
        id=f"{source}-{index}",
        text=f"text {source} {index}",
        source_path=source,
        chunk_index=index,
        content_hash=f"hash-{source}-{index}",
        metadata={"document_hash": f"doc-{source}"},
    )


def test_qdrant_store_paginates_sources_and_deletes_source(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", collection="test", vector_size=2)
    store.ensure_collection()
    chunks = [_chunk(f"/tmp/source-{index}.md", 0) for index in range(3)]
    store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], batch_size=2)

    assert len(store.list_sources(limit=None)) == 3
    assert store.count_source("/tmp/source-1.md") == 1
    assert store.delete_source("/tmp/source-1.md") == 1
    assert "/tmp/source-1.md" not in store.list_sources(limit=None)


def test_qdrant_store_inspects_collection(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", collection="test", vector_size=2)
    store.ensure_collection()
    store.upsert_chunks([_chunk("/tmp/a.md", 0)], [[1.0, 0.0]])

    info = store.inspect_collection()

    assert info["collection"] == "test"
    assert info["points"] == 1
    assert info["sources"] == 1
