""""
EvalRunner 负责跑项目本身的 RAG 问答，并产出 EvalResult
RagasEvalRunner 负责把 EvalResult 转成 RAGAS 格式，调用 RAGAS + judge LLM + embdding ，产出相应的指标
"""
from __future__ import annotations

import csv                                                      # 写 CSV 评分
import copy
import importlib                                                # 动态导入 RAGAS metric
import importlib.util                                           # 检查某个模块是否存在
import json                                                     # 把 list / dict 写进 CSV 单元格时转成 JSON str
import sys
import time
import types                                                    # 创建动态模块
from collections import defaultdict                             # 按 retriever 类型分组报告
from dataclasses import dataclass                               # 定义结果对象 RagasRunResult
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence                                # Sequence 表示序列类型，比 list 更宽，可以接收 list、ruple 等

from langchain_core.embeddings import Embeddings                # RAGAS 需要一个 LangChain 风格的 embedding 对象，要用 _ProjectEmbedding 包一层形成 LangChain 接口

from llm_doc_rag_agent.evaluation.runner import EvalRunner
from llm_doc_rag_agent.schemas import EvalResult
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import dump_jsonl, to_jsonable


DEFAULT_RAGAS_METRICS = (   # 默认 RAGAS metrics（即 RAGAS 指标）
    "faithfulness",         # 答案是否忠于 retrieved contexts
    "answer_relevancy",     # 答案是否回答了问题
    "context_precision",    # 检索上下文中排在前面的是否更有用
    "context_recall",       # 上下文是否覆盖标准答案所需信息
)

REFERENCE_REQUIRED_METRICS = {"context_precision", "context_recall", "answer_correctness"}  # 这些指标需要 reference ，即标准答案 ground truth

_METRIC_IMPORTS = {         # 映射表，key 是项目允许用户传的 metric 名称，value 是 (模块路径，模块中的变量名)
    "answer_correctness": ("ragas.metrics._answer_correctness", "answer_correctness"),
    "answer_relevancy": ("ragas.metrics._answer_relevance", "answer_relevancy"),
    "answer_relevance": ("ragas.metrics._answer_relevance", "answer_relevancy"),
    "answer_similarity": ("ragas.metrics._answer_similarity", "answer_similarity"),
    "context_precision": ("ragas.metrics._context_precision", "context_precision"),
    "context_recall": ("ragas.metrics._context_recall", "context_recall"),
    "faithfulness": ("ragas.metrics._faithfulness", "faithfulness"),    # 比如这里，后面 _load_metric() 会根据这个表动态导入 module = importlib.import_module("ragas.metrics._faithfulness")    return getattr(module,"faithfulness")
}


@dataclass(frozen=True) # 不可变 dataclass ，用来表示 RAGAS 运行结果，frozen=True 表示对象创建后字段不能再更改
class RagasRunResult:
    examples: int                       # 评分样本数量
    metrics: list[str]                  # 本次使用的指标
    output_path: Path                   # RAGAS CSV/JSONL 输出路径
    raw_output_path: Path | None        # 基础 RAG 问答结果路径
    report_path: Path | None            # MD 报告路径
    rows: list[dict[str, Any]]          # 最终合并后的每行评分结果


class RagasEvalRunner:  # 对已有 serivce 生成的答案作离线评估
    """Run offline RAGAS evaluation on answers produced by the existing service."""

    def __init__(self, service: RagService) -> None:
        self.service = service

    def default_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.service.settings.resolved_project_root / "experiments" / "ragas" / f"{timestamp}.csv"

    def default_raw_output_path(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser()
        return path.with_suffix(".raw.jsonl")   # raw 输出指的是先用项目 RAG 跑出来的原始答案结果

    def default_report_path(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser()
        return path.with_suffix(".md")

    def run(    # 完整 RAGAS 流程入口
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
        metric_names = _normalize_metrics(metrics or self.service.settings.ragas_metrics)   # 优先用命令行传入的 metrics ，否则用配置中的 ragas_metrics ---> 最后都要规范化

        base_runner = EvalRunner(self.service)  # 复用基础评估器
        started = time.perf_counter()
        eval_results = base_runner.run( # 读 dataset -> 调 service.query() -> 写 raw JSONL -> 返回 EvalResult 列表
            dataset_path=dataset_path,
            output_path=raw_output,
            top_k=top_k,
            retrievers=retrievers,
            candidate_k=candidate_k,
        )
        return self.score_results(  # 把基础问答结果交给 RAGAS 评分
            eval_results,
            output_path=output,
            raw_output_path=raw_output,
            report_path=report,
            metrics=metric_names,
            show_progress=show_progress,
            started=started,
        )

    def score_results(  # 只对已有结果做 RAGAS 评分
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

        ragas_rows = self.to_ragas_rows(eval_results)   # 把项目内部结果转成 RAGAS 需要的字段
        _validate_ragas_rows(ragas_rows, metric_names)  # 检查是否有样本，以及需要 reference 的指标是否都有标准答案
        evaluate, evaluation_dataset_cls, ragas_metrics = self._load_ragas(metric_names)     # 动态加载 RAGAS   evaluate RAGAS 的评分函数   EvaluationDataset RAGAS 数据集类    ragas_metrics 具体 metric 对象列表
        ragas_dataset = evaluation_dataset_cls.from_list(ragas_rows)    # 把普通 list 转成 RAGAS 数据集对象
        ragas_result = evaluate(    # RAGAS 评分
            ragas_dataset,          # 问题、答案、上下文、reference
            metrics=ragas_metrics,  # 要算的指标
            llm=self._make_llm(),   # judge LLM ,利用 _make_llm() 创建
            embeddings=_ProjectEmbeddings(self.service.embeddings), # 项目的 embedding provider 包装成 LangChain 接口
            raise_exceptions=False, # 单条评分异常时尽量不要让整个评估中断
            show_progress=show_progress,    # 是否显示进度条
        )
        score_rows = _score_rows(ragas_result)
        rows = _merge_scores(eval_results, score_rows, metric_names)    # 以上两行是后处理，RAGAS 返回的是纯指标分数，这里可以把原始问题、答案、trace、分数和在一起
        for row in rows:
            row["ragas_latency_seconds"] = round(time.perf_counter() - start_time, 4)   # 给每一行加上总耗时
        self.write_rows(rows, output)
        self.write_report(rows, report, metric_names)   # 写文件
        return RagasRunResult(
            examples=len(rows),
            metrics=list(metric_names),
            output_path=output,
            raw_output_path=raw_output,
            report_path=report,
            rows=rows,
        )

    def to_ragas_rows(self, results: list[EvalResult]) -> list[dict[str, Any]]: # 项目内部 EvalResult result 转为 RAGAS 输入格式
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

    def write_rows(self, rows: list[dict[str, Any]], output_path: str | Path) -> None:  # 写评分结果
        path = Path(output_path).expanduser()
        if path.suffix.lower() == ".jsonl": # 如果后缀是 .jsonl 就执行，否则写 CSV
            dump_jsonl(path, rows)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _fieldnames(rows)      # 不固定列，动态手机所有字段    因为不同 metrics 会产生不同列
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})  # 对每个字段都调用 _csv_value()，把 list / dict 转成 JSON str

    def write_report(self, rows: list[dict[str, Any]], report_path: str | Path, metrics: Sequence[str]) -> None:
        path = Path(report_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_report(rows, metrics), encoding="utf-8")

    def build_report(self, rows: list[dict[str, Any]], metrics: Sequence[str]) -> str:  # 生成 MD 报告
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
        for retriever, items in sorted(by_retriever.items()):   # by_retriever.items() 会得到类似 [("dense",[dense 的所有 row]),(["bm25",[bm25 的所有 row])]    sorted 会按照 key 排序
            values = [retriever, str(len(items))]               # 上面这行代码，实际含义：按照 retriever 类型逐组遍历，每次拿到一个 retriever 名称以及这个 retriever 对应的所有评估结果（可能有多条，每条用一个字典表示，dict 中包含 question retrievere_type 评分分数 等 key)
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

    def _make_llm(self) -> Any: # 创建 RAGAS judge LLM
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

    def _load_ragas(self, metric_names: Sequence[str]) -> tuple[Any, Any, list[Any]]:   # 动态加载 RAGAS
        try:
            _ensure_ragas_vertexai_compat() # 兼容补丁
            from ragas import EvaluationDataset, evaluate   # 导入 RAGAS
        except Exception as exc:  # pragma: no cover - exercised in real envs
            raise RuntimeError(
                "RAGAS could not be imported. Install compatible eval dependencies in the "
                "`llm_doc_rag` environment, then retry `llm-doc-rag ragas-eval`."
            ) from exc
        return evaluate, EvaluationDataset, [_load_metric(name) for name in metric_names]   # 返回 evaluate 、 EvaluationDataset 、 metric 对象列表


class _ProjectEmbeddings(Embeddings):   # 继承 LangChain 的 Embedding 抽象接口，让 RAGAS 可以复用项目已有 embedding 模型
    def __init__(self, provider: Any) -> None:
        self.provider = provider    # 保存项目 embedding provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.provider.embed_query(text)


def _load_metric(name: str) -> Any:
    normalized = _normalize_metric_name(name)   # 规范化 metric 名字
    if normalized not in _METRIC_IMPORTS:
        choices = ", ".join(sorted(_METRIC_IMPORTS))
        raise ValueError(f"Unsupported RAGAS metric '{name}'. Supported metrics: {choices}.")   # 如果不支持，报错，并给出支持的指标
    module_name, attr = _METRIC_IMPORTS[normalized] # python 的序列解包 执行后类似 module_name = "ragas.metrics._faithfulness"  attr = "faithfulness"
    module = importlib.import_module(module_name)   # 动态 import ，执行后 module 就是这个模块对象
    return _with_openai_compatible_defaults(getattr(module, attr))    # getattr(obj,"name")   从对象上取出某个属性    最终返回的就是 RAGAS 的 metric 对象


def _with_openai_compatible_defaults(metric: Any) -> Any:
    metric = copy.deepcopy(metric)
    if hasattr(metric, "strictness") and int(getattr(metric, "strictness") or 1) > 1:
        setattr(metric, "strictness", 1)
    return metric


def _normalize_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_normalize_metric_name(metric) for metric in metrics if str(metric).strip()) # 遍历 Sequence 中的每一个 metric，如果 metric 不为空的话，就将 metric 名字规范化，然后放到 tuple 中
    return normalized or DEFAULT_RAGAS_METRICS


def _normalize_metric_name(metric: str) -> str:
    return str(metric).strip().lower().replace("-", "_")


def _validate_ragas_rows(rows: list[dict[str, Any]], metrics: Sequence[str]) -> None:
    if not rows:    # 没有样本无法评估
        raise ValueError("RAGAS evaluation needs at least one example.")
    needs_reference = any(metric in REFERENCE_REQUIRED_METRICS for metric in metrics)   # 只要有一个 metric 需要 reference，就进入 reference 校验
    if needs_reference and any(not row.get("reference") for row in rows):   # 如果任何一行没有 reference，就报错
        needed = ", ".join(sorted(REFERENCE_REQUIRED_METRICS & set(metrics)))
        raise ValueError(f"RAGAS metrics [{needed}] require a ground_truth/reference column for every example.")


def _score_rows(ragas_result: Any) -> list[dict[str, Any]]: # 版本兼容，不同 RAGAS 版本返回对象可能不同
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
    for result, score in zip(results, score_rows, strict=True): # 每条 EvalResult 必须对应一条 RAGAS score
        rows.append(    # 把基础评估结果和 RAGAS socre 合成一行
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth or "",
                "retriever_type": result.trace.get("retriever_type", ""),
                "context_count": len(result.contexts),
                "contexts": result.contexts,
                "trace": result.trace,
                **{metric: score.get(metric) for metric in metrics},    # Python dict 解包
            }
        )
    return rows


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["question", "answer", "ground_truth", "retriever_type", "context_count"]   # 固定核心列顺序
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), ensure_ascii=False)   # CSV 单元格不能自然表达 list/dict，所以转成 JSON 字符串，False 表示保留中文
    return value


def _format_average(rows: list[dict[str, Any]], metric: str) -> str:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]  # 只统计数字类型分数
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


def _ensure_ragas_vertexai_compat() -> None:    # 兼容补丁
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or importlib.util.find_spec(module_name): # 如果这个模块已经存在，就什么都不做
        return
    from langchain_community.llms import VertexAI

    module = types.ModuleType(module_name)  # 如果模块不存在，就需要动态创建一个模块对象

    class ChatVertexAI(VertexAI):   # 定义一个继承 VertexAI 的 ChatVertexAI 类
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module   # 把这个动态模块塞进 sys.modules，让后续 import 能成功
