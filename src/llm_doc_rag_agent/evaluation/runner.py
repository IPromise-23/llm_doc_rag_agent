"""
读取评估数据集 CSV
-> 对每个问题都调用 RagService.query()
-> 记录答案、上下文、引用、耗时、retriever type
-> write JSONL or CSV result
-> 生成 MarkDown 简报

批量跑问答并落盘结果
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict # 按 retriever 类型分组统计报告
from datetime import datetime   # 生成默认输出文件名里的时间戳
from pathlib import Path
from typing import Any

from llm_doc_rag_agent.agents.quality import meaningful_terms
from llm_doc_rag_agent.schemas import EvalExample, EvalResult   # 评估输入 & 评估输出
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import dump_jsonl, to_jsonable


def load_eval_dataset(path: str | Path) -> list[EvalExample]:
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


class EvalRunner:   # 封装基础评估流程
    def __init__(self, service: RagService) -> None:
        self.service = service  # 评估时需要重复调用 self.service.query(...) ，每条评估样本都走项目的正式问答链路

    def load_dataset(self, path: str | Path) -> list[EvalExample]:  # 读取评估数据集
        return load_eval_dataset(path)

    def default_output_path(self) -> Path:  # 默认结果路径
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")    # 产生类似 20260616-151230 的结果
        return self.service.settings.resolved_project_root / "experiments" / "runs" / f"{timestamp}.jsonl"  # 最终默认输出到 项目根目录/experiments/runs/时间戳.jsonl

    def default_report_path(self, output_path: str | Path) -> Path: # 默认报告路径
        path = Path(output_path).expanduser()
        return path.with_suffix(".md")  # 把输出结果路径改为 .md 后缀

    def run(    # 核心评估流程
        self,
        dataset_path: str | Path,                       # 评估 CSV path
        output_path: str | Path | None = None,          # 结果输出 path
        top_k: int | None = None,                       # 每次检索取多少条
        retrievers: list[str] | None = None,            # 要比较哪些检索器
        candidate_k: int | None = None,                 # 候选召回数量，给 hybrid/rerank 用
        use_graph: bool = True,                         # 默认评估完整 LangGraph/RAG 链路；只检索请用 RetrievalEvalRunner
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        retriever_types = retrievers or self.service.settings.eval_retrievers
        for example in self.load_dataset(dataset_path): # 每次循环取出 评估数据集 中的一条数据
            for retriever_type in retriever_types:      # 每次循环选择一种检索器
                started = time.perf_counter()           # 记录开始时间，perf_counter() 适合测耗时，比 time.time() 更加适合性能计时
                answer = self.service.query(            # 调用真实问答服务
                    example.question,
                    top_k=top_k,
                    use_graph=use_graph,
                    retriever_type=retriever_type,
                    candidate_k=candidate_k,
                )
                elapsed = time.perf_counter() - started # 计算耗时
                trace: dict[str, Any] = dict(answer.trace)  # 复制一份 answer 中的 trace ，避免直接修改原来 answer.trace 对象
                trace.update(
                    {
                        "latency_seconds": round(elapsed, 4),
                        "top_k": top_k or self.service.settings.top_k,
                        "retriever_type": retriever_type,
                        "candidate_k": candidate_k if candidate_k is not None else self.service.settings.candidate_k,
                        "use_graph": use_graph,
                        "eval_layer": "rag",
                        "category": example.metadata.get("category", ""),
                        "answerable": example.metadata.get("answerable", ""),
                        "expected_route": example.metadata.get("expected_route", ""),
                        "expected_sources": _expected_sources(example.metadata),
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
        self.write_results(results, output_path or self.default_output_path())  # 把结果写到文件，然后返回结果列表
        return results

    def write_results(self, results: list[EvalResult], output_path: str | Path) -> None:    # 负责落盘
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".jsonl":
            dump_jsonl(path, results)   # 如果输出文件后缀是 .jsonl ，就用这个函数写 JSONL ，然后直接返回（避免继续执行 CSV 写入逻辑）
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:    # 打开文件写入
            writer = csv.DictWriter(    # 按字典写 CSV ，fieldnames 是 CSV 的列名
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
            writer.writeheader()    # 写表头
            for result in results:
                writer.writerow(    # 逐行写结果
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

    def write_report(self, results: list[EvalResult], report_path: str | Path) -> None: # 写 MD 报告
        path = Path(report_path).expanduser()                                           # 标准化路径
        path.parent.mkdir(parents=True, exist_ok=True)                                  # 创建目录
        path.write_text(self.build_report(results), encoding="utf-8")                   # 调用 build_report() 生成 MD text 并写入文件

    def build_report(self, results: list[EvalResult]) -> str:                           # 生成 MD 简报  评估结果 -> MD str
        by_retriever: dict[str, list[EvalResult]] = defaultdict(list)                   # 如果访问一个不存在的 key ，会自动创建空列表   by_retriever["dense"].append(result) --> 初始化 {"dense":[]}，再 append(result)
        for result in results:
            by_retriever[str(result.trace.get("retriever_type", "unknown"))].append(result) # 把结果按照 retriever 类型分组（按照检索器分组）

        lines = [
            "# RAG Evaluation Report",
            "",
            "## Summary",
            "",
            "| Retriever | Examples | Avg Contexts | Avg Latency (s) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for retriever, items in sorted(by_retriever.items()):
            avg_contexts = sum(len(item.contexts) for item in items) / max(len(items), 1)   # 计算对应于评估问题的检索回来的平均上下文数量
            avg_latency = sum(float(item.trace.get("latency_seconds", 0.0)) for item in items) / max(len(items), 1) # 计算平均耗时
            lines.append(f"| {retriever} | {len(items)} | {avg_contexts:.2f} | {avg_latency:.4f} |")    # 把统计结果追加为 MD 表格行

        lines.extend(   # 这些指标是确定性词项诊断
            [
                "",
                "## Quality Diagnostics",
                "",
                "These are deterministic lexical diagnostics, not RAGAS or LLM-judge scores.",
                "",
                "| Retriever | With Ground Truth | Avg Answer/Context | Avg Answer/Ground Truth | Avg Context/Ground Truth | Insufficient Answers |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for retriever, items in sorted(by_retriever.items()):
            metrics = [_quality_metrics(item) for item in items]    # 对每个检索器下的每条结果调用 _quality_metrics() ，得到每条样本的本地质量指标
            answer_context = _average_metric(metrics, "answer_context_overlap")
            answer_ground_truth = _average_metric(metrics, "answer_ground_truth_coverage")
            context_ground_truth = _average_metric(metrics, "context_ground_truth_coverage")
            with_ground_truth = sum(1 for metric in metrics if metric["has_ground_truth"])  # 有 ground truth 的样本数
            insufficient = sum(1 for metric in metrics if metric["insufficient_answer"])    # 被识别为拒答/上下文不足的样本数
            lines.append(
                f"| {retriever} | {with_ground_truth} | {_format_ratio(answer_context)} | "
                f"{_format_ratio(answer_ground_truth)} | {_format_ratio(context_ground_truth)} | {insufficient} |"
            )

        lines.extend(   # 添加 examples 表格
            [
                "",
                "## Examples",
                "",
                "| Question | Retriever | Contexts | Answer/Context | Answer/Ground Truth | Context/Ground Truth | Ground Truth Preview | Answer Preview |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for result in results:
            retriever = str(result.trace.get("retriever_type", "unknown"))
            question = _escape_table_cell(result.question)
            ground_truth = _escape_table_cell(_preview(result.ground_truth or ""))
            answer = _escape_table_cell(_preview(result.answer))
            metrics = _quality_metrics(result)  # 计算该样本的质量指标
            lines.append(
                f"| {question} | {retriever} | {len(result.contexts)} | "
                f"{_format_ratio(metrics['answer_context_overlap'])} | "        # 答案是否贴近上下文
                f"{_format_ratio(metrics['answer_ground_truth_coverage'])} | "  # 答案是否覆盖标准答案
                f"{_format_ratio(metrics['context_ground_truth_coverage'])} | " # 检索上下文是否覆盖标准答案
                f"{ground_truth} | {answer} |"
            )
        issue_rows = _quality_issue_rows(results)       # 找可疑样本，遍历所有 EvalResult ，给每条样本打问题标签
        lines.extend(["", "## Potential Issues", ""])   # 生成潜在问题列表
        if issue_rows:                                  # 如果存在问题样本
            lines.extend(
                [
                    "| Question | Retriever | Signals | Answer Preview |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for result, signals in issue_rows:
                question = _escape_table_cell(result.question)
                retriever = _escape_table_cell(str(result.trace.get("retriever_type", "unknown")))
                answer = _escape_table_cell(_preview(result.answer))
                lines.append(f"| {question} | {retriever} | {', '.join(signals)} | {answer} |")
        else:
            lines.append("No obvious deterministic quality issues detected.")   # 没发现问题，但也并非说明质量好
        lines.append("")
        return "\n".join(lines) # 把所有行用换行拼成一个 MD str


class RetrievalEvalRunner:
    """Evaluate retrieval quality without calling the LLM generation layer."""

    def __init__(self, service: RagService) -> None:
        self.service = service

    def load_dataset(self, path: str | Path) -> list[EvalExample]:
        return load_eval_dataset(path)

    def default_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.service.settings.resolved_project_root / "experiments" / "retrieval" / f"{timestamp}.csv"

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
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        retriever_types = retrievers or self.service.settings.eval_retrievers
        effective_top_k = top_k or self.service.settings.top_k
        effective_candidate_k = candidate_k if candidate_k is not None else self.service.settings.candidate_k
        for example in self.load_dataset(dataset_path):
            expected_sources = _expected_sources(example.metadata)
            for retriever_type in retriever_types:
                started = time.perf_counter()
                retrieved = self.service.retrieve_only(
                    question=example.question,
                    top_k=effective_top_k,
                    retriever_type=retriever_type,
                    candidate_k=candidate_k,
                )
                elapsed = time.perf_counter() - started
                retrieved_sources = [item.chunk.source_path for item in retrieved]
                first_hit_rank = _first_expected_source_rank(retrieved_sources, expected_sources)
                rows.append(
                    {
                        "question": example.question,
                        "ground_truth": example.ground_truth or "",
                        "category": example.metadata.get("category", ""),
                        "answerable": example.metadata.get("answerable", ""),
                        "expected_sources": expected_sources,
                        "retriever_type": retriever_type,
                        "top_k": effective_top_k,
                        "candidate_k": effective_candidate_k,
                        "context_count": len(retrieved),
                        "retrieved_sources": retrieved_sources,
                        "top_score": retrieved[0].score if retrieved else None,
                        "hit": first_hit_rank is not None if expected_sources else None,
                        "first_hit_rank": first_hit_rank,
                        "reciprocal_rank": (1.0 / first_hit_rank) if first_hit_rank else None,
                        "latency_seconds": round(elapsed, 4),
                        "eval_layer": "retrieval",
                    }
                )
        self.write_rows(rows, output_path or self.default_output_path())
        return rows

    def write_rows(self, rows: list[dict[str, Any]], output_path: str | Path) -> None:
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".jsonl":
            dump_jsonl(path, rows)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _retrieval_fieldnames(rows)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    def write_report(self, rows: list[dict[str, Any]], report_path: str | Path) -> None:
        path = Path(report_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_report(rows), encoding="utf-8")

    def build_report(self, rows: list[dict[str, Any]]) -> str:
        by_retriever: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_retriever[str(row.get("retriever_type") or "unknown")].append(row)

        lines = [
            "# Retrieval Evaluation Report",
            "",
            "This report evaluates retrieval only. It does not call the LLM generation layer.",
            "",
            "## Summary",
            "",
            "| Retriever | Examples | With Expected Sources | Hit Rate | MRR | Avg Contexts | Avg Latency (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for retriever, items in sorted(by_retriever.items()):
            expected_items = [item for item in items if item.get("expected_sources")]
            hits = [item for item in expected_items if item.get("hit") is True]
            hit_rate = len(hits) / len(expected_items) if expected_items else None
            mrr = _average_row_value(expected_items, "reciprocal_rank")
            avg_contexts = _average_row_value(items, "context_count")
            avg_latency = _average_row_value(items, "latency_seconds")
            lines.append(
                f"| {retriever} | {len(items)} | {len(expected_items)} | {_format_ratio(hit_rate)} | "
                f"{_format_ratio(mrr)} | {_format_ratio(avg_contexts)} | {_format_latency(avg_latency)} |"
            )

        lines.extend(
            [
                "",
                "## Examples",
                "",
                "| Question | Retriever | Expected Sources | Hit | First Hit Rank | Retrieved Sources |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table_cell(str(row.get("question") or "")),
                        _escape_table_cell(str(row.get("retriever_type") or "")),
                        _escape_table_cell("; ".join(row.get("expected_sources") or [])),
                        _escape_table_cell(_format_bool(row.get("hit"))),
                        str(row.get("first_hit_rank") or ""),
                        _escape_table_cell("; ".join(_short_source(source) for source in row.get("retrieved_sources") or [])),
                    ]
                )
                + " |"
            )

        missed = [row for row in rows if row.get("expected_sources") and row.get("hit") is not True]
        lines.extend(["", "## Potential Issues", ""])
        if missed:
            lines.extend(["| Question | Retriever | Expected Sources | Retrieved Sources |", "| --- | --- | --- | --- |"])
            for row in missed:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape_table_cell(str(row.get("question") or "")),
                            _escape_table_cell(str(row.get("retriever_type") or "")),
                            _escape_table_cell("; ".join(row.get("expected_sources") or [])),
                            _escape_table_cell("; ".join(_short_source(source) for source in row.get("retrieved_sources") or [])),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No expected-source misses detected.")
        lines.append("")
        return "\n".join(lines)


def _preview(text: str, limit: int = 120) -> str:   # 压缩答案预览
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _escape_table_cell(text: str) -> str:   # 转义 MD 表格单元格
    return text.replace("|", "\\|").replace("\n", " ")  # MD cell 中 | 是列分隔符，如果答案里本身有 | ，表格会乱，所以要替换为 \|   换行符 -> 空格


def _quality_metrics(result: EvalResult) -> dict[str, Any]:                 # 每条结果的本地质量指标
    answer_terms = set(meaningful_terms(result.answer))
    context_terms = set(meaningful_terms(" ".join(result.contexts)))
    ground_truth_terms = set(meaningful_terms(result.ground_truth or ""))   # 把答案、上下文、标准答案分别转成有意义词项集合
    insufficient_answer = any(
        phrase in result.answer.lower()
        for phrase in ("不足以回答", "无法回答", "insufficient", "not enough context")  # 检测答案是否像是拒绝回答或者上下文不足回答
    )
    return {
        "has_ground_truth": bool(ground_truth_terms),
        "insufficient_answer": insufficient_answer,
        "answer_context_overlap": _overlap_ratio(answer_terms, context_terms),                                              # 答案词项中有多少出现在上下文中
        "answer_ground_truth_coverage": _overlap_ratio(ground_truth_terms, answer_terms) if ground_truth_terms else None,   # 标准答案词项中有多少被答案覆盖
        "context_ground_truth_coverage": _overlap_ratio(ground_truth_terms, context_terms) if ground_truth_terms else None, # 标准答案词项中有多少被检索上下文覆盖
    }


def _quality_issue_rows(results: list[EvalResult]) -> list[tuple[EvalResult, list[str]]]:   # 找可疑样本，遍历所有 EvalResult ，给每条样本打问题标签
    rows: list[tuple[EvalResult, list[str]]] = []
    for result in results:
        metrics = _quality_metrics(result)  # 对应 eval result 的本地质量指标
        signals: list[str] = []
        if metrics["insufficient_answer"]:
            signals.append("insufficient_answer")
        if result.contexts and metrics["answer_context_overlap"] < 0.2:
            signals.append("low_answer_context_overlap")
        if metrics["answer_ground_truth_coverage"] is not None and metrics["answer_ground_truth_coverage"] < 0.2:
            signals.append("low_answer_ground_truth_coverage")
        if metrics["context_ground_truth_coverage"] is not None and metrics["context_ground_truth_coverage"] < 0.2:
            signals.append("low_context_ground_truth_coverage")
        if signals:
            rows.append((result, signals))
    return rows


def _overlap_ratio(source_terms: set[str], target_terms: set[str]) -> float:
    if not source_terms:
        return 0.0
    return len(source_terms & target_terms) / len(source_terms)


def _average_metric(metrics: list[dict[str, Any]], key: str) -> float | None:   # 平均指标
    values = [float(metric[key]) for metric in metrics if isinstance(metric.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _format_ratio(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return ""


def _format_latency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _expected_sources(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("expected_sources") or metadata.get("expected_source") or ""
    if isinstance(raw, list):
        values = raw
    else:
        values = str(raw).replace("|", ";").split(";")
    return [str(value).strip() for value in values if str(value).strip()]


def _first_expected_source_rank(retrieved_sources: list[str], expected_sources: list[str]) -> int | None:
    if not expected_sources:
        return None
    for rank, source in enumerate(retrieved_sources, start=1):
        if any(_source_matches(source, expected) for expected in expected_sources):
            return rank
    return None


def _source_matches(source_path: str, expected_source: str) -> bool:
    source = source_path.replace("\\", "/")
    expected = expected_source.replace("\\", "/").lstrip("./")
    return source == expected or source.endswith(f"/{expected}") or source.endswith(expected)


def _short_source(source_path: str) -> str:
    source = source_path.replace("\\", "/")
    markers = ("/docs/", "/data/", "/src/", "/tests/")
    for marker in markers:
        if marker in source:
            return source[source.index(marker) + 1 :]
    return Path(source).name


def _retrieval_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "question",
        "ground_truth",
        "category",
        "answerable",
        "expected_sources",
        "retriever_type",
        "top_k",
        "candidate_k",
        "context_count",
        "retrieved_sources",
        "top_score",
        "hit",
        "first_hit_rank",
        "reciprocal_rank",
        "latency_seconds",
        "eval_layer",
    ]
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _average_row_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    if value is None:
        return ""
    return value
