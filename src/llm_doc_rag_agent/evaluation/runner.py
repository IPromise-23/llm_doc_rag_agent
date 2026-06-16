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
import time
from collections import defaultdict # 按 retriever 类型分组统计报告
from datetime import datetime   # 生成默认输出文件名里的时间戳
from pathlib import Path
from typing import Any

from llm_doc_rag_agent.agents.quality import meaningful_terms
from llm_doc_rag_agent.schemas import EvalExample, EvalResult   # 评估输入 & 评估输出
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import dump_jsonl, to_jsonable


class EvalRunner:   # 封装基础评估流程
    def __init__(self, service: RagService) -> None:
        self.service = service  # 评估时需要重复调用 self.service.query(...) ，每条评估样本都走项目的正式问答链路

    def load_dataset(self, path: str | Path) -> list[EvalExample]:  # 读取评估数据集
        dataset_path = Path(path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Eval dataset does not exist: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as fh:    # with ... as fh 是上下文管理器，文件读完后自动关闭
            rows = list(csv.DictReader(fh))     # 把 CSV 每一行读成字典     list 再把迭代器一次性转成列表   rows 中包含多个待评估的问题，每一个问题（包括标注答案、tag等信息）组成一个 row
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
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        retriever_types = retrievers or self.service.settings.eval_retrievers
        for example in self.load_dataset(dataset_path): # 每次循环取出 评估数据集 中的一条数据
            for retriever_type in retriever_types:      # 每次循环选择一种检索器
                started = time.perf_counter()           # 记录开始时间，perf_counter() 适合测耗时，比 time.time() 更加适合性能计时
                answer = self.service.query(            # 调用真实问答服务
                    example.question,
                    top_k=top_k,
                    use_graph=retriever_type == "dense",
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
            by_retriever[str(result.trace.get("retriever_type", "unknown"))].append(result) # 按照 retriever 类型分组

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

        lines.extend(
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
            metrics = [_quality_metrics(item) for item in items]
            answer_context = _average_metric(metrics, "answer_context_overlap")
            answer_ground_truth = _average_metric(metrics, "answer_ground_truth_coverage")
            context_ground_truth = _average_metric(metrics, "context_ground_truth_coverage")
            with_ground_truth = sum(1 for metric in metrics if metric["has_ground_truth"])
            insufficient = sum(1 for metric in metrics if metric["insufficient_answer"])
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
            metrics = _quality_metrics(result)
            lines.append(
                f"| {question} | {retriever} | {len(result.contexts)} | "
                f"{_format_ratio(metrics['answer_context_overlap'])} | "
                f"{_format_ratio(metrics['answer_ground_truth_coverage'])} | "
                f"{_format_ratio(metrics['context_ground_truth_coverage'])} | "
                f"{ground_truth} | {answer} |"
            )
        issue_rows = _quality_issue_rows(results)
        lines.extend(["", "## Potential Issues", ""])
        if issue_rows:
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
            lines.append("No obvious deterministic quality issues detected.")
        lines.append("")
        return "\n".join(lines) # 把所有行用换行拼成一个 MD str


def _preview(text: str, limit: int = 120) -> str:   # 压缩答案预览
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _escape_table_cell(text: str) -> str:   # 转义 MD 表格单元格
    return text.replace("|", "\\|").replace("\n", " ")  # MD cell 中 | 是列分隔符，如果答案里本身有 | ，表格会乱，所以要替换为 \|   换行符 -> 空格


def _quality_metrics(result: EvalResult) -> dict[str, Any]:
    answer_terms = set(meaningful_terms(result.answer))
    context_terms = set(meaningful_terms(" ".join(result.contexts)))
    ground_truth_terms = set(meaningful_terms(result.ground_truth or ""))
    insufficient_answer = any(
        phrase in result.answer.lower()
        for phrase in ("不足以回答", "无法回答", "insufficient", "not enough context")
    )
    return {
        "has_ground_truth": bool(ground_truth_terms),
        "insufficient_answer": insufficient_answer,
        "answer_context_overlap": _overlap_ratio(answer_terms, context_terms),
        "answer_ground_truth_coverage": _overlap_ratio(ground_truth_terms, answer_terms) if ground_truth_terms else None,
        "context_ground_truth_coverage": _overlap_ratio(ground_truth_terms, context_terms) if ground_truth_terms else None,
    }


def _quality_issue_rows(results: list[EvalResult]) -> list[tuple[EvalResult, list[str]]]:
    rows: list[tuple[EvalResult, list[str]]] = []
    for result in results:
        metrics = _quality_metrics(result)
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


def _average_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(metric[key]) for metric in metrics if isinstance(metric.get(key), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _format_ratio(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return ""
