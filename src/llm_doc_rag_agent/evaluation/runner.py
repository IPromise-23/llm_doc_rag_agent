from __future__ import annotations

import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_doc_rag_agent.schemas import EvalExample, EvalResult
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import dump_jsonl, to_jsonable


class EvalRunner:
    def __init__(self, service: RagService) -> None:
        self.service = service

    def load_dataset(self, path: str | Path) -> list[EvalExample]:
        dataset_path = Path(path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Eval dataset does not exist: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        examples = []
        for row in rows:
            question = (row.get("question") or "").strip()
            if not question:
                continue
            examples.append(
                EvalExample(
                    question=question,
                    ground_truth=(row.get("ground_truth") or row.get("answer") or "").strip() or None,
                    metadata={k: v for k, v in row.items() if k not in {"question", "ground_truth", "answer"}},
                )
            )
        return examples

    def default_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.service.settings.resolved_project_root / "experiments" / "runs" / f"{timestamp}.jsonl"

    def default_report_path(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser()
        return path.with_suffix(".md")

    def run(
        self,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        top_k: int | None = None,
        retrievers: list[str] | None = None,
        candidate_k: int | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        retriever_types = retrievers or self.service.settings.eval_retrievers
        for example in self.load_dataset(dataset_path):
            for retriever_type in retriever_types:
                started = time.perf_counter()
                answer = self.service.query(
                    example.question,
                    top_k=top_k,
                    use_graph=retriever_type == "dense",
                    retriever_type=retriever_type,
                    candidate_k=candidate_k,
                )
                elapsed = time.perf_counter() - started
                trace: dict[str, Any] = dict(answer.trace)
                trace.update(
                    {
                        "latency_seconds": round(elapsed, 4),
                        "top_k": top_k or self.service.settings.top_k,
                        "retriever_type": retriever_type,
                        "candidate_k": candidate_k if candidate_k is not None else self.service.settings.candidate_k,
                    }
                )
                results.append(
                    EvalResult(
                        question=example.question,
                        answer=answer.answer,
                        ground_truth=example.ground_truth,
                        contexts=answer.contexts,
                        citations=answer.citations,
                        trace=trace,
                    )
                )
        self.write_results(results, output_path or self.default_output_path())
        return results

    def write_results(self, results: list[EvalResult], output_path: str | Path) -> None:
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".jsonl":
            dump_jsonl(path, results)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "question",
                    "answer",
                    "ground_truth",
                    "retriever_type",
                    "context_count",
                    "citations",
                    "trace",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "question": result.question,
                        "answer": result.answer,
                        "ground_truth": result.ground_truth or "",
                        "context_count": len(result.contexts),
                        "citations": to_jsonable(result.citations),
                        "retriever_type": result.trace.get("retriever_type", ""),
                        "trace": to_jsonable(result.trace),
                    }
                )

    def write_report(self, results: list[EvalResult], report_path: str | Path) -> None:
        path = Path(report_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_report(results), encoding="utf-8")

    def build_report(self, results: list[EvalResult]) -> str:
        by_retriever: dict[str, list[EvalResult]] = defaultdict(list)
        for result in results:
            by_retriever[str(result.trace.get("retriever_type", "unknown"))].append(result)

        lines = [
            "# RAG Evaluation Report",
            "",
            "## Summary",
            "",
            "| Retriever | Examples | Avg Contexts | Avg Latency (s) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for retriever, items in sorted(by_retriever.items()):
            avg_contexts = sum(len(item.contexts) for item in items) / max(len(items), 1)
            avg_latency = sum(float(item.trace.get("latency_seconds", 0.0)) for item in items) / max(len(items), 1)
            lines.append(f"| {retriever} | {len(items)} | {avg_contexts:.2f} | {avg_latency:.4f} |")

        lines.extend(
            [
                "",
                "## Examples",
                "",
                "| Question | Retriever | Contexts | Answer Preview |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for result in results:
            retriever = str(result.trace.get("retriever_type", "unknown"))
            question = _escape_table_cell(result.question)
            answer = _escape_table_cell(_preview(result.answer))
            lines.append(f"| {question} | {retriever} | {len(result.contexts)} | {answer} |")
        lines.append("")
        return "\n".join(lines)


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
