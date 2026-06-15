# llm_doc_rag_agent Demo 指南

这份文档用于准备面试或简历沟通时的演示。目标是展示项目主链路和 trace，不追求生产级部署。

## 环境前提

```bash
conda activate llm_doc_rag
cd /Users/ipromise/Desktop/llm_doc_rag_agent
export PYTHONPATH=src
```

如果要运行完整 `query/eval`，需要本地 `.env` 中配置 `DEEPSEEK_API_KEY`，并且 embedding 模型已可用。不要在演示材料、截图或输出里展示 `.env` 内容。

## 场景 1：索引本地文档

目的：展示 loader、splitter、embedding、Qdrant、本地持久化和增量索引。

命令：

```bash
python -m llm_doc_rag_agent.cli ingest \
  --path data/raw \
  --collection demo_docs \
  --config configs/default.yaml
```

可以讲：

- `LocalDocumentLoader` 只读取显式传入的 `data/raw`。
- `.ragignore` 会跳过 `.env`、key、索引目录和实验输出。
- 第二次 ingest 未变化文档会被跳过，避免重复 embedding。
- Qdrant 使用本地 path 持久化，适合个人项目和本地知识库。

预期观察点：

- 输出中应包含 `documents`、`changed_documents`、`skipped_documents`、`chunks`、`upserted`、`qdrant_path`。
- 再执行一次同样命令，`skipped_documents` 应增加，说明增量索引生效。

## 场景 2：普通 RAG Query

目的：展示 dense/hybrid 检索、LangGraph RAG 分支、CRAG/Self-RAG trace。

命令：

```bash
python -m llm_doc_rag_agent.cli query \
  "What does the MVP expose?" \
  --collection demo_docs \
  --config configs/default.yaml \
  --top-k 3 \
  --retriever hybrid_rrf \
  --candidate-k 20
```

可以讲：

- `retriever_type=hybrid_rrf` 同时利用 dense 和 BM25 候选。
- LangGraph path 应包含 `route_question -> retrieve -> grade_documents -> generate -> grade_generation`。
- `grade_documents` 会判断检索上下文是否足够相关。
- `grade_generation` 会把 `answer_grounded`、`answer_relevant` 写入 trace，用于调试。

预期观察点：

- `trace.route` 是 `retrieve_rag`。
- `trace.graph_path` 包含 `grade_documents` 和 `grade_generation`。
- `citations` 里能看到 source path、chunk index、score 和 snippet。

## 场景 3：Source Lookup 或 Insufficient Context

目的：展示 Agent 路由不只是 RAG 问答，还能跳过不必要的 embedding/LLM。

Source lookup 命令：

```bash
python -m llm_doc_rag_agent.cli query \
  "请列出当前有哪些文档" \
  --collection demo_docs \
  --config configs/default.yaml
```

可以讲：

- 这类问题不需要语义检索，也不需要 LLM。
- Graph 会路由到 `source_lookup`。
- trace 中 `retrieval_skipped=true`，说明系统没有走昂贵的 RAG 生成链路。

Insufficient context 的测试级演示：

```bash
PYTHONPATH=src python -m pytest tests/test_agents.py -q
```

可以讲：

- 单元测试覆盖了检索失败后 rewrite 一次再检索。
- 也覆盖了超过 rewrite 预算后返回 insufficient context，而不是让 LLM 强答。
- 这个设计用于控制幻觉风险，符合 RAG Agent 的基本工程边界。

## 已验证的轻量命令

以下命令不需要真实 LLM 调用，适合演示前做 sanity check：

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m llm_doc_rag_agent.cli query --help
PYTHONPATH=src python -m llm_doc_rag_agent.cli eval --help
```

最近一次验证结果：

```text
27 passed
query --help 正常渲染
eval --help 正常渲染
```

## 演示时的注意事项

- 不展示 `.env`、API key、完整 prompt 或本机敏感路径。
- 如果现场没有 API key，就展示 source lookup、CLI help、pytest 和已有文档说明。
- 如果 embedding 模型首次运行需要下载，提前在本地缓存好模型，避免现场等待。
- 不把规则型 CRAG/Self-RAG 说成企业级 judge；准确表达为“低成本、可测试、可解释的实践版自检策略”。
