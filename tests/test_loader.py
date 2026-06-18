from pathlib import Path

from llm_doc_rag_agent.loaders import LocalDocumentLoader


def test_loader_reads_explicit_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Title\n\nhello", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("ignore", encoding="utf-8")
    (tmp_path / "skip.bin").write_text("ignore", encoding="utf-8")

    docs = LocalDocumentLoader().load_path(tmp_path)

    assert len(docs) == 1
    assert docs[0].source_path.endswith("a.md")
    assert "hello" in docs[0].text
    assert docs[0].metadata["file_size"] > 0
    assert docs[0].metadata["document_hash"]


def test_loader_respects_ragignore(tmp_path: Path):
    (tmp_path / ".ragignore").write_text("ignored.md\nnested/\n", encoding="utf-8")
    (tmp_path / "keep.md").write_text("keep", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("ignore", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "skip.md").write_text("ignore", encoding="utf-8")

    docs = LocalDocumentLoader().load_path(tmp_path)

    assert len(docs) == 1
    assert docs[0].source_path.endswith("keep.md")


def test_loader_uses_project_root_ragignore_for_nested_ingest(tmp_path: Path):
    (tmp_path / ".ragignore").write_text("data/raw/ignored.md\n", encoding="utf-8")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "keep.md").write_text("keep", encoding="utf-8")
    (raw / "ignored.md").write_text("ignore", encoding="utf-8")

    docs = LocalDocumentLoader(ignore_roots=[tmp_path]).load_path(raw)

    assert len(docs) == 1
    assert docs[0].source_path.endswith("keep.md")


def test_loader_reads_explicit_yaml_file(tmp_path: Path):
    config = tmp_path / "default.yaml"
    config.write_text("chunk_size: 900\nchunk_overlap: 120\n", encoding="utf-8")

    docs = LocalDocumentLoader().load_path(config)

    assert len(docs) == 1
    assert docs[0].source_path.endswith("default.yaml")
    assert "chunk_size: 900" in docs[0].text
    assert docs[0].metadata["file_type"] == "yaml"


def test_loader_reads_notebook_cells(tmp_path: Path):
    notebook = tmp_path / "demo.ipynb"
    notebook.write_text(
        '{"cells":[{"cell_type":"markdown","source":["# Demo"]},{"cell_type":"code","source":["x = 1"]}]}',
        encoding="utf-8",
    )

    docs = LocalDocumentLoader().load_path(notebook)

    assert len(docs) == 1
    assert "[markdown cell 0]" in docs[0].text
    assert "x = 1" in docs[0].text
