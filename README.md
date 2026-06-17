# llm_doc_rag_agent

`llm_doc_rag_agent` 是一个面向本地技术文档的 RAG Agent 项目。它把 [`rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch)、[`qdrant-rag-eval`](https://github.com/qdrant/qdrant-rag-eval)、[`langgraph`](https://github.com/langchain-ai/langgraph) 中的主链路整理成可复用的工程代码：

```text
本地文档 -> 加载 -> 切块 -> Embedding -> Qdrant -> LangGraph 路由 -> 检索/来源查询 -> CRAG 自检 -> 生成 -> Self-RAG Trace -> 评估
```

当前工程能力已推进到 `v0.4-practice`：`v0.2` 的工程化索引、`v0.3` 的 hybrid retrieval/eval、`v0.4` 的 LangGraph 路由与规则型 CRAG/Self-RAG 自检已落地。包版本号仍保持 `0.1.0`，后续发布时再单独调整。

## 当前能力

- 本地文档加载：支持 `.md`、`.txt`、`.py`、`.ipynb`、`.rst`，PDF 通过可选依赖 `pypdf` 支持。
- 文档切块：使用确定性文本切块，保留 `source_path`、`chunk_index`、`content_hash` 等基础元数据。
- 安全忽略：默认读取 `.ragignore`，跳过 `.env`、key、索引目录、实验输出和隐藏目录。
- Embedding：默认使用 `SentenceTransformer` 加载 BGE 模型。
- 向量库：使用本地持久化 Qdrant，索引默认写入 `data/indexes/qdrant`。
- 索引管理：支持未变化 source 跳过、source 删除、source 重建和 collection inspect。
- 检索：支持 dense、轻量 BM25、hybrid RRF 检索。
- 生成：使用 OpenAI-compatible Chat Completions 接口，默认面向 DeepSeek 配置。
- 编排：LangGraph 先路由到 `source_lookup`、`direct_answer` 或 `retrieve_rag`；普通问题走 `retrieve -> grade_documents -> generate -> grade_generation`。
- Agent 自检：规则型 CRAG gate 会判断检索上下文是否足够相关，必要时改写 query 并重试；规则型 Self-RAG trace 会记录答案 groundedness/relevance。
- Trace：query 返回 `route`、`graph_path`、`retriever_type`、`candidate_k`、`document_grade_decision`、`answer_grounded` 等调试字段。
- 入口：提供 CLI、可选 FastAPI、三层 CSV/JSONL eval runner。

## 目录结构

本项目使用 Python 常见的 `src layout`。外层目录是项目根目录，`src/llm_doc_rag_agent` 才是真正可导入的 Python 包：

```text
llm_doc_rag_agent/
  pyproject.toml
  README.md
  configs/
    default.yaml
  data/
    raw/
    eval/
    indexes/
  docs/
    roadmap.md
  experiments/
  src/
    llm_doc_rag_agent/
      agents/
      api/
      chunking/
      embeddings/
      evaluation/
      generation/
      loaders/
      retrieval/
      vectorstores/
      cli.py
      config.py
      schemas.py
      service.py
  tests/
```

这里的 `src/llm_doc_rag_agent` 为了避免测试或脚本误导入当前目录中的源码文件。`pyproject.toml` 中的配置也是按这个结构声明的：

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

## 快速开始

项目按本机 conda 环境 `llm_doc_rag` 设计：

```bash
conda activate llm_doc_rag
cd /Users/ipromise/Desktop/llm_doc_rag_agent
```

如果要调用 LLM 生成答案，需要在本机 `.env` 中配置 `DEEPSEEK_API_KEY`。本项目不应把 `.env` 提交或写入文档输出。

安装为本地命令后可运行：

```bash
llm-doc-rag ingest --path ./README.md --collection project_eval
llm-doc-rag ingest --path ./docs --collection project_eval
llm-doc-rag ingest --path ./data/raw --collection project_eval
llm-doc-rag sources --collection project_eval
llm-doc-rag eval-retrieval --dataset ./data/eval/questions.csv --collection project_eval
llm-doc-rag query "这个项目如何做检索？" --collection my_docs
llm-doc-rag eval --dataset ./data/eval/questions.csv --collection project_eval
llm-doc-rag ragas-eval --dataset ./data/eval/questions.csv --collection my_docs
```

开发时也可以不安装包，直接从 `src` 运行：

```bash
PYTHONPATH=src python -m llm_doc_rag_agent.cli query "问题" --collection my_docs
```

## 常用命令

```bash
llm-doc-rag ingest --path ./docs --collection my_docs
llm-doc-rag query "问题" --collection my_docs --top-k 5
llm-doc-rag retrieve "问题" --collection my_docs --top-k 5 --retriever dense
llm-doc-rag retrieve "函数名或配置项" --collection my_docs --retriever bm25
llm-doc-rag retrieve "混合检索问题" --collection my_docs --retriever hybrid_rrf --candidate-k 20
llm-doc-rag retrieve "需要精排的问题" --collection my_docs --retriever hybrid_rerank --candidate-k 20
llm-doc-rag query "请列出当前有哪些文档" --collection my_docs
llm-doc-rag query "查看 source=path/to/file.md 的 chunks" --collection my_docs
llm-doc-rag eval-retrieval --dataset data/eval/questions.csv --collection project_eval --retriever dense --retriever bm25 --retriever hybrid_rrf --report experiments/reports/retrieval.md
llm-doc-rag eval --dataset data/eval/questions.csv --collection project_eval --retriever dense --retriever bm25 --retriever hybrid_rrf --report experiments/reports/rag.md
llm-doc-rag ragas-eval --dataset data/eval/questions.csv --collection my_docs --retriever dense --retriever hybrid_rrf --metric faithfulness --metric answer_relevancy
llm-doc-rag sources --collection my_docs
llm-doc-rag chunks --source path/to/file.md --collection my_docs
llm-doc-rag inspect-collection --collection my_docs
llm-doc-rag delete-source --source path/to/file.md --collection my_docs
llm-doc-rag reindex-source --path path/to/file.md --collection my_docs
llm-doc-rag eval --dataset data/eval/questions.csv --collection my_docs
```

## 如何跑通 ingest/query/eval

当前项目还没有外部业务知识库时，可以先把项目自己的 README、docs 和 smoke 文档作为评估语料入库：

```bash
conda activate llm_doc_rag
cd /Users/ipromise/Desktop/llm_doc_rag_agent
export PYTHONPATH=src

python -m llm_doc_rag_agent.cli ingest --path README.md --collection project_eval --config configs/default.yaml
python -m llm_doc_rag_agent.cli ingest --path docs --collection project_eval --config configs/default.yaml
python -m llm_doc_rag_agent.cli ingest --path data/raw --collection project_eval --config configs/default.yaml
python -m llm_doc_rag_agent.cli sources --collection project_eval --config configs/default.yaml
```

`ingest` 会写入本地 Qdrant 索引。`sources` 只读取已入库 collection。普通 `query` 和完整 `eval` 会调用配置的 LLM，因此需要 `.env` 中有 `DEEPSEEK_API_KEY`：

```bash
python -m llm_doc_rag_agent.cli query \
  "What is the difference between eval, eval-retrieval, and ragas-eval?" \
  --collection project_eval \
  --config configs/default.yaml \
  --retriever hybrid_rrf \
  --candidate-k 20
```

三层评估的边界如下：

1. `eval-retrieval`：只调用 retriever，不调用 LLM，用 `expected_sources` 检查 source 命中、rank、MRR。
2. `eval`：调用完整 RAG/Graph 链路，比较不同 retriever 下的答案、上下文、trace 和确定性质量诊断。
3. `ragas-eval`：在完整 RAG 结果上调用 RAGAS + judge LLM，计算 faithfulness、answer relevancy、context precision/recall 等指标。

```bash
python -m llm_doc_rag_agent.cli eval-retrieval \
  --dataset data/eval/questions.csv \
  --collection project_eval \
  --config configs/default.yaml \
  --retriever dense \
  --retriever bm25 \
  --retriever hybrid_rrf \
  --report experiments/reports/retrieval.md

python -m llm_doc_rag_agent.cli eval \
  --dataset data/eval/questions.csv \
  --collection project_eval \
  --config configs/default.yaml \
  --retriever dense \
  --retriever hybrid_rrf \
  --report experiments/reports/rag.md
```

`data/eval/questions.csv` 当前以项目自带文档为目标语料，字段包括 `question`、`ground_truth`、`expected_sources`、`expected_route`、`answerable`、`category` 和 `notes`。如果后续换成真实业务文档，先把目标文档入库，再按同样字段补充评估问题。

## 当前边界

这些能力目前还没有完全工程化，后续版本会优先补齐：

- 当前 BM25/hybrid 基于已有 chunk payload 临时计算，还没有 Qdrant named sparse vectors 和持久化 sparse 索引。
- Reranker 已有可插拔策略入口；默认未配置 `reranker_model` 时不会加载 CrossEncoder，只做候选截断。
- CRAG/Self-RAG 运行时 gate 默认仍是规则型轻量实现；离线评估已提供 `ragas-eval`，会复用 DeepSeek/OpenAI-compatible judge，并默认关闭 DeepSeek thinking。
- Eval 已拆成三层：`eval-retrieval` 负责不调用 LLM 的检索命中评估，`eval` 负责完整 RAG/Graph 答案评估，`ragas-eval` 负责离线自动质量指标。RAGAS 运行前需要 dataset 提供 `ground_truth`。
- API 已有项目根目录路径限制和统一错误响应雏形，但 request id、streaming 和 collection 管理还未补齐。
- 项目有核心单元测试和 GitHub Actions 测试工作流，但还没有结构化日志、成本统计和完整前端。

## 路线图

详细版本规划放在 [docs/roadmap.md](/Users/ipromise/Desktop/llm_doc_rag_agent/docs/roadmap.md)。简要节奏如下：

| 版本 | 目标 | 关键能力 |
| --- | --- | --- |
| `v0.1` | MVP 主链路 | loader、splitter、embedding、Qdrant、dense retrieval、QA、CLI、eval、最小 LangGraph |
| `v0.2` | 工程化索引与基础评估 | 增量索引、source 删除/重建、Qdrant 分页、实验输出、CLI 增强、安全忽略 |
| `v0.3` | Hybrid Retrieval 与 Reranker | 已有轻量 BM25、hybrid RRF、检索配置、多策略 eval、可选 reranker 入口；后续补 sparse vectors 和真实 reranker 验证 |
| `v0.4` | Self-RAG / Adaptive RAG Agent | 已有 `source_lookup/direct_answer/retrieve_rag` 路由、规则型 `grade_documents`、query rewrite、`grade_generation` 和 trace |
| `v0.5` | API 服务化与前端展示 | 统一 API 错误、request id、streaming、collection 管理、轻量 UI |
| `v0.6` | 合规、观测与发布质量 | `.ragignore`、日志、成本统计、测试、CI、故障排查文档 |

## 求职复盘材料

当前版本已经收束为适合 Agent/RAG 方向实习展示的 `v0.4-practice`。后续建议优先复盘和表达，而不是继续无限加功能：

- [项目复盘与架构讲解](/Users/ipromise/Desktop/llm_doc_rag_agent/docs/project_recap.md)：架构图、1 分钟讲解、3-5 分钟讲解、核心亮点和技术取舍。
- [Demo 指南](/Users/ipromise/Desktop/llm_doc_rag_agent/docs/demo_guide.md)：ingest、RAG query、source lookup/insufficient context 三类演示场景和验证命令。
- [简历条目与面试 Q&A](/Users/ipromise/Desktop/llm_doc_rag_agent/docs/resume_interview.md)：可直接改写进简历的项目描述，以及常见追问回答。

## 可选 API

API 代码位于 `src/llm_doc_rag_agent/api/app.py`。如果环境里安装了 `fastapi` 和 `uvicorn`，可以运行：

```bash
PYTHONPATH=src uvicorn llm_doc_rag_agent.api.app:create_app --factory --reload
```

当前阶段不默认安装可选依赖；如果缺 `fastapi`、`uvicorn`、`pypdf` 或 `pytest`，应先确认依赖策略，再决定是否安装。

## 设计原则

- 默认只读取用户显式传入的目录或文件，不扫描全盘。
- Qdrant 索引、实验结果和日志默认保存在本地。
- `.env` 只放本机配置和 API key，不进入文档输出和索引内容。
- 基础版先实现可测试、可评估、可追溯的 RAG，再逐步加入高级检索和 Agent 策略。
- notebook 中的代码只作为设计来源，项目代码按模块化边界重写。
