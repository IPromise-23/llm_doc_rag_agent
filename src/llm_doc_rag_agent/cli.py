"""
该脚本是一个 Typer 应用
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from llm_doc_rag_agent.config import get_settings                       # 从 config.py 中读取环境变量和 YAML
from llm_doc_rag_agent.evaluation import EvalRunner, RagasEvalRunner, RetrievalEvalRunner    # 做评估
from llm_doc_rag_agent.service import RagService                        # 项目主服务
from llm_doc_rag_agent.utils import to_jsonable                         # 把 dataclass Path 复杂对象 转成可以打印的结构

app = typer.Typer(help="Local technical-document RAG agent.")           # 创建 CLI 应用对象     help=... 会显示在顶层 --help    后面所有的 @app.command() 都会注册成子命令


def _service(collection: Optional[str], config: Optional[Path]) -> RagService:  # 统一创建服务对象
    settings = get_settings().with_yaml(config)                                 # 先从环境变量/ .env 中读默认的配置     如果传了 YAML 就覆盖配置
    return RagService(settings=settings, collection=collection)


@app.command()  # 装饰器，把函数注册成为一个 CLI 子命令     ingest() ---> cli.py ingest
def ingest(
    path: Path = typer.Option(..., "--path", "-p", help="Explicit file or directory to ingest."),   # typer.Option(...) 表示这是一个命令行选项，不是位置参数    ... 表示必填，即 --path / -p 是必须提供的
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),                           # Optional[str] == str | None   这里表示 collection 是可选 命令行参数 ，并非一定要传入
    config: Optional[Path] = typer.Option(None, "--config"),                                        # 如果 collection 和 config 都不传入，_sercvice(None,None) 都会用默认配置   --config config.yaml 才是有传入，也可以不传入 -c
    recreate: bool = typer.Option(False, "--recreate", help="Recreate the target collection before ingesting."),    # 开关参数，传入 --recreate 就是 True ，不传就是 False
) -> None:
    service = _service(collection, config)                      # CLI 负责接收参数
    result = service.ingest_path(path=path, recreate=recreate)  # RagService.ingest_path() 负责把本地文件或目录入库
    print(to_jsonable(result))                                  # to_jsonable() 把结果转成适合打印的形式


@app.command()
def query(      # 检索 + 生成
    question: str = typer.Argument(...),                                    # typer.Argument(...) 表示 question 是位置参数，不是 --question 这种 flag   cli.py query "what does this function do?"
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
def retrieve(   # 只检索，不生成，返回 RetrievedChunk 列表
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
def sources(    # source lookup 的命令行入口，对应 service.list_sources()
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    service = _service(collection, config)
    print(service.list_sources(limit=limit))


@app.command("chunks")  # 命令名是 chunks   函数名是 chunks_ command
def chunks_command(
    source: str = typer.Option(..., "--source"),    # --source 是必填参数，表示要查看哪个 source 的 chunk
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    service = _service(collection, config)
    print(to_jsonable(service.chunks_for_source(source_path=source, limit=limit)))   # 返回指定 source 对应的 Chunk 列表


@app.command("delete-source")   # 删除某个 source 所有的 chunks
def delete_source_command(
    source: str = typer.Option(..., "--source"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    service = _service(collection, config)
    deleted = service.delete_source(source)
    print({"collection": service.collection, "source": source, "deleted_chunks": deleted})


@app.command("reindex-source")  # 先删除旧数据，再重新 ingest   参数从 CLI 进来，调用 RagService，最后打印结果
def reindex_source_command(
    path: Path = typer.Option(..., "--path", "-p", help="Explicit source file to reindex."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    service = _service(collection, config)
    print(to_jsonable(service.reindex_source(path)))


@app.command("inspect-collection")  # 查看 collection 状态的诊断命令，对应底层 Qdrant 存储的概览信息
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
    retrievers: Optional[list[str]] = typer.Option(None, "--retriever", help="Repeat for full RAG comparison, e.g. --retriever dense --retriever bm25 --retriever hybrid_rrf --retriever hybrid_rerank."),    # 参数可以是一个字符串列表
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Bypass LangGraph for all retrievers during this RAG eval."),
    report: Optional[Path] = typer.Option(None, "--report", help="Write a Markdown summary report. Defaults to output path with .md suffix."),
) -> None:
    service = _service(collection, config)
    runner = EvalRunner(service)    # 调用 批评估器 ，读 dataset -> 跑多组 retriever -> 调用 service.query(...) -> 写结果和报告     CLI 组织批量任务    EvalRunner 负责评估逻辑     RagService 负责单条回答
    output_path = output or runner.default_output_path()
    results = runner.run(
        dataset_path=dataset,
        output_path=output_path,
        top_k=top_k,
        retrievers=retrievers,
        candidate_k=candidate_k,
        use_graph=not no_graph,
    )
    report_path = report or runner.default_report_path(output_path)
    runner.write_report(results, report_path)
    payload = {"examples": len(results), "output": str(output_path), "report": str(report_path)}
    if service.settings.run_ragas:  # 条件升级路径，如果配置中打开了 RAGAS ，就额外调用，生成 RAGAS 评分结果、CSV 和 报告
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


@app.command("eval-retrieval")
def eval_retrieval_command(
    dataset: Path = typer.Option(..., "--dataset"),
    output: Optional[Path] = typer.Option(None, "--output"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retrievers: Optional[list[str]] = typer.Option(None, "--retriever", help="Repeat for retrieval-only comparison, e.g. --retriever dense --retriever bm25 --retriever hybrid_rrf --retriever hybrid_rerank."),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write a Markdown retrieval summary report. Defaults to output path with .md suffix."),
) -> None:
    service = _service(collection, config)
    runner = RetrievalEvalRunner(service)
    output_path = output or runner.default_output_path()
    rows = runner.run(
        dataset_path=dataset,
        output_path=output_path,
        top_k=top_k,
        retrievers=retrievers,
        candidate_k=candidate_k,
    )
    report_path = report or runner.default_report_path(output_path)
    runner.write_report(rows, report_path)
    print({"examples": len(rows), "output": str(output_path), "report": str(report_path)})


@app.command("ragas-eval")  # 直接跑 RAGAS 的入口   和 eval 的区别是，eval 先跑项目内部评估再按需追加 RAGAS ，而 ragas-eval 则是直接跑 RAGAS 流程
def ragas_eval_command(
    dataset: Path = typer.Option(..., "--dataset"),
    output: Optional[Path] = typer.Option(None, "--output"),
    raw_output: Optional[Path] = typer.Option(None, "--raw-output", help="Write raw RAG answers before RAGAS scoring."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c"),
    config: Optional[Path] = typer.Option(None, "--config"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    retrievers: Optional[list[str]] = typer.Option(None, "--retriever", help="Repeat for comparison, e.g. --retriever dense --retriever bm25 --retriever hybrid_rrf --retriever hybrid_rerank."),
    candidate_k: Optional[int] = typer.Option(None, "--candidate-k"),
    metrics: Optional[list[str]] = typer.Option(None, "--metric", help="Repeat to select RAGAS metrics."),  # --metric 可以重复传，多指标评估
    report: Optional[Path] = typer.Option(None, "--report", help="Write a Markdown RAGAS summary report."),
    show_progress: bool = typer.Option(False, "--show-progress", help="Show RAGAS progress bars."),         # --show-progress 控制是否显示进度条
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


if __name__ == "__main__":  # 直接运行这个文件时，启动 Typer CLI    被别的模块 import 时不自动执行
    app()   # 进入 Typer 的命令解析和分发逻辑

"""
为什么会出现不用传入参数的情况？

ingest 是把文档导入到向量库
query 是从已经导入的 collection 中检索并回答问题

第一次先导入了 A 内容的文档，第二次想知道 B 内容的相关回答，就需要先把 B 文档 ingest 到某个 colllection ，再 query 这个 collection
collection 控制：使用 Qdrant 中的哪一个文档集合     collection_a -> 存 A 文档   collection_b -> 存 b 文档
config 控制：本次运行时会用哪套 settings ，比如： embedding model / chunk_size / top_k / LLM model / retriever_type
如果不对 B 创建一个新的 collection_b ,而是在 collection_a 中 ingest ，那么这个 collection 中就同时有 A & B ，除非 --recreate

collection 是知识库隔离边界     config 是运行参数集合   ingest 负责建立知识库   query 负责查询已经建立好的知识库
"""
