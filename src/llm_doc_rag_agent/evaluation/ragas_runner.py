from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings

from llm_doc_rag_agent.evaluation.runner import EvalRunner
from llm_doc_rag_agent.schemas import EvalResult
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import dump_jsonl, to_jsonable


DEFAULT_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

REFERENCE_REQUIRED_METRICS = {"context_precision", "context_recall", "answer_correctness"}

_METRIC_IMPORTS = {
    "answer_correctness": ("ragas.metrics._answer_correctness", "answer_correctness"),
    "answer_relevancy": ("ragas.metrics._answer_relevance", "answer_relevancy"),
    "answer_relevance": ("ragas.metrics._answer_relevance", "answer_relevancy"),
    "answer_similarity": ("ragas.metrics._answer_similarity", "answer_similarity"),
    "context_precision": ("ragas.metrics._context_precision", "context_precision"),
    "context_recall": ("ragas.metrics._context_recall", "context_recall"),
    "faithfulness": ("ragas.metrics._faithfulness", "faithfulness"),
}


@dataclass(frozen=True)
class RagasRunResult:
    examples: int
    metrics: list[str]
    output_path: Path
    raw_output_path: Path | None
    report_path: Path | None
    rows: list[dict[str, Any]]


class RagasEvalRunner:
    """Run offline RAGAS evaluation on answers produced by the existing service."""

    def __init__(self, service: RagService) -> None:
        self.service = service

    def default_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.service.settings.resolved_project_root / "experiments" / "ragas" / f"{timestamp}.csv"

    def default_raw_output_path(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser()
        return path.with_suffix(".raw.jsonl")

    def default_report_path(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser()
        return path.with_suffix(".md")

    def run(
        self,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        raw_output_path: str | Path | None = None,
        report_path: str | Path | None = None,
        top_k: int | None = None,
        retrievers: list[str] | None = None,
        candidate_k: int | None = None,
        metrics: Sequence[str] | None = None,
        show_progress: bool = False,
    ) -> RagasRunResult:
        output = Path(output_path).expanduser() if output_path else self.default_output_path()
        raw_output = Path(raw_output_path).expanduser() if raw_output_path else self.default_raw_output_path(output)
        report = Path(report_path).expanduser() if report_path else self.default_report_path(output)
        metric_names = _normalize_metrics(metrics or self.service.settings.ragas_metrics)

        base_runner = EvalRunner(self.service)
        started = time.perf_counter()
        eval_results = base_runner.run(
            dataset_path=dataset_path,
            output_path=raw_output,
            top_k=top_k,
            retrievers=retrievers,
            candidate_k=candidate_k,
        )
        return self.score_results(
            eval_results,
            output_path=output,
            raw_output_path=raw_output,
            report_path=report,
            metrics=metric_names,
            show_progress=show_progress,
            started=started,
        )

    def score_results(
        self,
        eval_results: list[EvalResult],
        output_path: str | Path,
        raw_output_path: str | Path | None = None,
        report_path: str | Path | None = None,
        metrics: Sequence[str] | None = None,
        show_progress: bool = False,
        started: float | None = None,
    ) -> RagasRunResult:
        output = Path(output_path).expanduser()
        raw_output = Path(raw_output_path).expanduser() if raw_output_path else None
        report = Path(report_path).expanduser() if report_path else self.default_report_path(output)
        metric_names = _normalize_metrics(metrics or self.service.settings.ragas_metrics)
        start_time = started if started is not None else time.perf_counter()

        ragas_rows = self.to_ragas_rows(eval_results)
        _validate_ragas_rows(ragas_rows, metric_names)
        evaluate, evaluation_dataset_cls, ragas_metrics = self._load_ragas(metric_names)
        ragas_dataset = evaluation_dataset_cls.from_list(ragas_rows)
        ragas_result = evaluate(
            ragas_dataset,
            metrics=ragas_metrics,
            llm=self._make_llm(),
            embeddings=_ProjectEmbeddings(self.service.embeddings),
            raise_exceptions=False,
            show_progress=show_progress,
        )
        score_rows = _score_rows(ragas_result)
        rows = _merge_scores(eval_results, score_rows, metric_names)
        for row in rows:
            row["ragas_latency_seconds"] = round(time.perf_counter() - start_time, 4)
        self.write_rows(rows, output)
        self.write_report(rows, report, metric_names)
        return RagasRunResult(
            examples=len(rows),
            metrics=list(metric_names),
            output_path=output,
            raw_output_path=raw_output,
            report_path=report,
            rows=rows,
        )

    def to_ragas_rows(self, results: list[EvalResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            row: dict[str, Any] = {
                "user_input": result.question,
                "response": result.answer,
                "retrieved_contexts": result.contexts,
            }
            if result.ground_truth:
                row["reference"] = result.ground_truth
            rows.append(row)
        return rows

    def write_rows(self, rows: list[dict[str, Any]], output_path: str | Path) -> None:
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".jsonl":
            dump_jsonl(path, rows)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _fieldnames(rows)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    def write_report(self, rows: list[dict[str, Any]], report_path: str | Path, metrics: Sequence[str]) -> None:
        path = Path(report_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_report(rows, metrics), encoding="utf-8")

    def build_report(self, rows: list[dict[str, Any]], metrics: Sequence[str]) -> str:
        by_retriever: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_retriever[str(row.get("retriever_type") or "unknown")].append(row)

        header = ["Retriever", "Examples", *metrics]
        lines = [
            "# RAGAS Evaluation Report",
            "",
            "## Summary",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---", "---:"] + ["---:"] * len(metrics)) + " |",
        ]
        for retriever, items in sorted(by_retriever.items()):
            values = [retriever, str(len(items))]
            values.extend(_format_average(items, metric) for metric in metrics)
            lines.append("| " + " | ".join(values) + " |")

        lines.extend(
            [
                "",
                "## Examples",
                "",
                "| Question | Retriever | " + " | ".join(metrics) + " | Answer Preview |",
                "| --- | --- | " + " | ".join(["---:"] * len(metrics)) + " | --- |",
            ]
        )
        for row in rows:
            metric_values = [_format_number(row.get(metric)) for metric in metrics]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table_cell(str(row.get("question") or "")),
                        _escape_table_cell(str(row.get("retriever_type") or "unknown")),
                        *metric_values,
                        _escape_table_cell(_preview(str(row.get("answer") or ""))),
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)

    def _make_llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        settings = self.service.settings
        api_key = settings.ragas_api_key or settings.quality_api_key or settings.deepseek_api_key
        if not api_key:
            raise RuntimeError("Missing RAGAS judge API key. Set RAGAS_API_KEY, QUALITY_API_KEY, or DEEPSEEK_API_KEY.")
        extra_body = {"thinking": {"type": "disabled"}} if settings.ragas_disable_thinking else None
        return ChatOpenAI(
            api_key=api_key,
            base_url=settings.ragas_base_url or settings.quality_base_url or settings.llm_base_url,
            model=settings.ragas_model or settings.quality_model or settings.llm_model,
            temperature=0.0,
            max_tokens=settings.ragas_max_tokens,
            top_p=settings.ragas_top_p,
            extra_body=extra_body,
        )

    def _load_ragas(self, metric_names: Sequence[str]) -> tuple[Any, Any, list[Any]]:
        try:
            _ensure_ragas_vertexai_compat()
            from ragas import EvaluationDataset, evaluate
        except Exception as exc:  # pragma: no cover - exercised in real envs
            raise RuntimeError(
                "RAGAS could not be imported. Install compatible eval dependencies in the "
                "`llm_doc_rag` environment, then retry `llm-doc-rag ragas-eval`."
            ) from exc
        return evaluate, EvaluationDataset, [_load_metric(name) for name in metric_names]


class _ProjectEmbeddings(Embeddings):
    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.provider.embed_query(text)


def _load_metric(name: str) -> Any:
    normalized = _normalize_metric_name(name)
    if normalized not in _METRIC_IMPORTS:
        choices = ", ".join(sorted(_METRIC_IMPORTS))
        raise ValueError(f"Unsupported RAGAS metric '{name}'. Supported metrics: {choices}.")
    module_name, attr = _METRIC_IMPORTS[normalized]
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _normalize_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_normalize_metric_name(metric) for metric in metrics if str(metric).strip())
    return normalized or DEFAULT_RAGAS_METRICS


def _normalize_metric_name(metric: str) -> str:
    return str(metric).strip().lower().replace("-", "_")


def _validate_ragas_rows(rows: list[dict[str, Any]], metrics: Sequence[str]) -> None:
    if not rows:
        raise ValueError("RAGAS evaluation needs at least one example.")
    needs_reference = any(metric in REFERENCE_REQUIRED_METRICS for metric in metrics)
    if needs_reference and any(not row.get("reference") for row in rows):
        needed = ", ".join(sorted(REFERENCE_REQUIRED_METRICS & set(metrics)))
        raise ValueError(f"RAGAS metrics [{needed}] require a ground_truth/reference column for every example.")


def _score_rows(ragas_result: Any) -> list[dict[str, Any]]:
    scores = getattr(ragas_result, "scores", None)
    if isinstance(scores, list):
        return [dict(score) for score in scores]
    if hasattr(ragas_result, "to_pandas"):
        return ragas_result.to_pandas().to_dict(orient="records")
    raise TypeError("Unsupported RAGAS result object: expected .scores or .to_pandas().")


def _merge_scores(
    results: list[EvalResult],
    score_rows: list[dict[str, Any]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result, score in zip(results, score_rows, strict=True):
        rows.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth or "",
                "retriever_type": result.trace.get("retriever_type", ""),
                "context_count": len(result.contexts),
                "contexts": result.contexts,
                "trace": result.trace,
                **{metric: score.get(metric) for metric in metrics},
            }
        )
    return rows


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["question", "answer", "ground_truth", "retriever_type", "context_count"]
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    return value


def _format_average(rows: list[dict[str, Any]], metric: str) -> str:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _ensure_ragas_vertexai_compat() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or importlib.util.find_spec(module_name):
        return
    from langchain_community.llms import VertexAI

    module = types.ModuleType(module_name)

    class ChatVertexAI(VertexAI):
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
