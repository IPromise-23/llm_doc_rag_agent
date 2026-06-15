from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from llm_doc_rag_agent.config import get_settings
from llm_doc_rag_agent.evaluation import EvalRunner, RagasEvalRunner
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import to_jsonable

app = typer.Typer(help="Local technical-document RAG agent.")


def _service(collection: Optional[str], config: Optional[Path]) -> RagService:
    settings = get_settings().with_yaml(config)
    return RagService(settings=settings, collection=collection)


@app.command()
def ingest(
    path: Path = typer.Option(..., "--path", "-p", help="Explicit file or directory to ingest."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate the target collection before ingesting."),
) -> None:
    service = _service(collection, config)
    result = service.ingest_path(path=path, recreate=recreate)
    print(to_jsonable(result))


@app.command()
def query(
    question: str = typer.Argument(...),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retriever_type: Optional[str] = typer.Option(
        None,
        "--retriever",
        help="dense, bm25, hybrid_rrf, dense_rerank, or hybrid_rerank.",
    ),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Bypass LangGraph and call retrieve/generate directly."),
) -> None:
    service = _service(collection, config)
    answer = service.query(
        question=question,
        top_k=top_k,
        use_graph=not no_graph,
        retriever_type=retriever_type or service.settings.retriever_type,
        candidate_k=candidate_k,
    )
    print(to_jsonable(answer))


@app.command()
def retrieve(
    question: str = typer.Argument(...),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retriever_type: Optional[str] = typer.Option(
        None,
        "--retriever",
        help="dense, bm25, hybrid_rrf, dense_rerank, or hybrid_rerank.",
    ),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
) -> None:
    service = _service(collection, config)
    print(
        to_jsonable(
            service.retrieve_only(
                question=question,
                top_k=top_k,
                retriever_type=retriever_type or service.settings.retriever_type,
                candidate_k=candidate_k,
            )
        )
    )


@app.command()
def sources(
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    service = _service(collection, config)
    print(service.list_sources(limit=limit))


@app.command("chunks")
def chunks_command(
    source: str = typer.Option(..., "--source"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    service = _service(collection, config)
    print(to_jsonable(service.chunks_for_source(source_path=source, limit=limit)))


@app.command("delete-source")
def delete_source_command(
    source: str = typer.Option(..., "--source"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    service = _service(collection, config)
    deleted = service.delete_source(source)
    print({"collection": service.collection, "source": source, "deleted_chunks": deleted})


@app.command("reindex-source")
def reindex_source_command(
    path: Path = typer.Option(..., "--path", "-p", help="Explicit source file to reindex."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    service = _service(collection, config)
    print(to_jsonable(service.reindex_source(path)))


@app.command("inspect-collection")
def inspect_collection_command(
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    service = _service(collection, config)
    print(to_jsonable(service.inspect_collection()))


@app.command("eval")
def eval_command(
    dataset: Path = typer.Option(..., "--dataset"),
    output: Optional[Path] = typer.Option(None, "--output"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retrievers: Optional[list[str]] = typer.Option(None, "--retriever", help="Repeat for comparison, e.g. --retriever dense --retriever bm25."),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write a Markdown summary report. Defaults to output path with .md suffix."),
) -> None:
    service = _service(collection, config)
    runner = EvalRunner(service)
    output_path = output or runner.default_output_path()
    results = runner.run(
        dataset_path=dataset,
        output_path=output_path,
        top_k=top_k,
        retrievers=retrievers,
        candidate_k=candidate_k,
    )
    report_path = report or runner.default_report_path(output_path)
    runner.write_report(results, report_path)
    payload = {"examples": len(results), "output": str(output_path), "report": str(report_path)}
    if service.settings.run_ragas:
        ragas_runner = RagasEvalRunner(service)
        ragas_output = output_path.with_suffix(".ragas.csv")
        ragas_report = ragas_runner.default_report_path(ragas_output)
        ragas_result = ragas_runner.score_results(
            results,
            output_path=ragas_output,
            raw_output_path=output_path,
            report_path=ragas_report,
            metrics=service.settings.ragas_metrics,
        )
        payload.update(
            {
                "ragas_output": str(ragas_result.output_path),
                "ragas_report": str(ragas_result.report_path) if ragas_result.report_path else None,
                "ragas_metrics": ragas_result.metrics,
            }
        )
    print(payload)


@app.command("ragas-eval")
def ragas_eval_command(
    dataset: Path = typer.Option(..., "--dataset"),
    output: Optional[Path] = typer.Option(None, "--output"),
    raw_output: Optional[Path] = typer.Option(None, "--raw-output", help="Write raw RAG answers before RAGAS scoring."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retrievers: Optional[list[str]] = typer.Option(None, "--retriever", help="Repeat for comparison, e.g. --retriever dense --retriever hybrid_rrf."),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    metrics: Optional[list[str]] = typer.Option(None, "--metric", help="Repeat to select RAGAS metrics."),
    report: Optional[Path] = typer.Option(None, "--report", help="Write a Markdown RAGAS summary report."),
    show_progress: bool = typer.Option(False, "--show-progress", help="Show RAGAS progress bars."),
) -> None:
    service = _service(collection, config)
    runner = RagasEvalRunner(service)
    output_path = output or runner.default_output_path()
    result = runner.run(
        dataset_path=dataset,
        output_path=output_path,
        raw_output_path=raw_output,
        report_path=report,
        top_k=top_k,
        retrievers=retrievers,
        candidate_k=candidate_k,
        metrics=metrics,
        show_progress=show_progress,
    )
    print(
        {
            "examples": result.examples,
            "metrics": result.metrics,
            "output": str(result.output_path),
            "raw_output": str(result.raw_output_path) if result.raw_output_path else None,
            "report": str(result.report_path) if result.report_path else None,
        }
    )


if __name__ == "__main__":
    app()
