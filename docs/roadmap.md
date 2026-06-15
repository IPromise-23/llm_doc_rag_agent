# llm_doc_rag_agent Roadmap

本文记录 `llm_doc_rag_agent` 从 `v0.1 MVP` 走向完整本地文档 RAG Agent 项目的版本规划。README 只保留项目门面和简要路线，本文件承接更细的技术拆解和验收标准。

## v0.1：MVP 主链路

目标：把学习阶段掌握的 RAG 主链路落成可运行的项目骨架。

MVP 基线能力：

- 本地文档加载、确定性切块、基础元数据保留。
- SentenceTransformer/BGE embedding provider。
- Qdrant 本地持久化 collection。
- Dense retrieval。
- OpenAI-compatible QA generation。
- 最小 LangGraph `retrieve -> generate`。
- CLI、可选 FastAPI、CSV/JSONL eval runner。

当前全局边界：

- `ingest/query/retrieve` 会加载 embedding 模型；如果本机没有模型缓存，第一次运行可能触发模型下载。
- `query/eval` 会调用配置的 OpenAI-compatible LLM；必须先配置 API key。
- `sources/chunks` 只读取已有 Qdrant collection，不主动加载 embedding 模型。
- BM25/hybrid 目前基于已有 chunk payload 临时计算，尚无 Qdrant named sparse vectors。Reranker 已有入口，但真实 CrossEncoder 需要配置 `reranker_model` 后才会加载。
- API 还是轻量入口，尚无 request id、streaming 和 collection 管理。

## v0.2：工程化索引与基础评估

目标：让项目从“可跑通 MVP”升级为“可重复维护的本地知识库”。

技术改造：

- 完善增量索引：当前已按 `document_hash` 跳过未变化 source，后续补更完整的索引 manifest 和变更报告。
- 删除与重建：当前已有 `delete-source` 和 `reindex-source`，后续补更严格的 dry-run 和批量操作确认。
- Qdrant 分页：当前 `list_sources` 和 `chunks_for_source` 已使用 scroll offset 循环，后续补更多 collection 统计。
- 元数据 schema：payload 固定为 `text/source_path/chunk_index/content_hash/metadata`，metadata 内补 `file_type/section_title/heading_path/token_estimate`。
- 安全忽略：当前已增加 `.ragignore`，默认跳过 `.env`、key、证书、隐藏目录、索引目录、实验输出和大文件。
- Eval 输出：当前已默认写入 `experiments/runs/{timestamp}.jsonl`，后续补多策略对比字段和更完整指标。
- CLI 增强：当前已有 `inspect-collection`、`delete-source`、`reindex-source`，后续补更完整的 inspect 输出和批量管理能力。

验收标准：

- 同一目录重复 ingest，不变文件不会重复 embedding。
- 删除一个 source 后，`sources` 和 `query` 都不再返回该文件。
- `sources/chunks` 能分页读取大 collection。
- Eval 结果可以用 CSV/JSONL 对比不同 `top_k/chunk_size/chunk_overlap`。
- 敏感文件不会被默认索引。

## v0.3：Hybrid Retrieval 与 Reranker

目标：把 `qdrant-rag-eval` 中学到的 dense、BM25、hybrid、rerank 工程化。

技术改造：

- 当前已实现：新增轻量 `BM25Retriever`，从已有 Qdrant chunk payload 临时构建词法索引。
- 当前已实现：新增 `HybridRetriever`，用 RRF 融合 dense 与 BM25 结果。
- 当前已实现：配置支持 `retriever_type`、`candidate_k`、`eval_retrievers`。
- 当前已实现：`eval` 支持重复 `--retriever` 进行 dense、BM25、hybrid RRF 横向对比，并可生成 Markdown 汇总报告。
- 当前已实现：`dense_rerank`、`hybrid_rerank` 策略入口和可选 CrossEncoder reranker；默认未配置模型时不加载外部模型。
- 后续 Sparse 编码器：新增 `retrieval/sparse.py`，实现持久化 `SimpleBM25Encoder`，字段包括 `vocab/idf/doc_len/avg_doc_len`。
- Qdrant named vectors：collection 同时保存 dense vector 和 sparse vector，payload 仍保留 chunk 文本和来源。
- Retriever router：当前支持 `dense`、`bm25`、`hybrid_rrf`、`dense_rerank`、`hybrid_rerank`。
- RRF 融合：当前 dense 和 BM25 各召回 `candidate_k`，用 reciprocal rank fusion 合并后截断到 `top_k`。
- Reranker：已新增 `retrieval/reranker.py`，默认 NoOp，配置 `reranker_model` 后使用 CrossEncoder；只对 `candidate_k` 候选精排，不替代向量库召回。
- 参数配置：后续可拆出 `configs/retrieval.yaml`，补 `hybrid_alpha/rerank_model` 等高级参数。
- 对比评估：当前已输出同一问题下不同策略的上下文和答案，并生成报告；后续补自动质量指标和更完整实验模板。

验收标准：

- 对包含函数名、配置项、错误码的问题，BM25/hybrid 比 dense 更容易召回精确 chunk。
- Reranker 不改变候选集合，只改变排序；trace 中能看到 rerank 前后顺序。
- 同一 eval dataset 能横向比较 dense、hybrid、hybrid_rerank。

## v0.4：Self-RAG / Adaptive RAG Agent

目标：把当前最小 LangGraph 流程升级为可控的文档问答 agent。

当前已实现：

- GraphState 已扩展 `route/decision/graph_path/trace/sources/chunks` 等字段，为后续 CRAG/Self-RAG 节点预留状态边界。
- Adaptive 路由雏形：入口 `route_question` 可把问题分到 `source_lookup/direct_answer/retrieve_rag`。
- Source lookup：当用户问“有哪些文档/列出 sources/查看 source 的 chunks”时，不走 semantic retrieval，不调用 LLM，直接读 collection metadata。
- RAG 分支：普通问题走 `retrieve -> grade_documents -> generate -> grade_generation`，并在 answer trace 中返回 `route`、`graph_path`、`retriever_type`、`candidate_k`、`top_k`。
- CRAG 节点：已增加规则型 `grade_documents`，根据 query terms、retriever score 和最少相关 chunk 数判断是否接受上下文。
- Query rewrite：当没有足够相关上下文且未超过 `max_rewrites` 时，进入 `rewrite_query` 并重试检索；超过预算后返回 insufficient context，不调用 LLM 强答。
- Self-RAG trace：已增加规则型 `grade_generation`，记录 `answer_grounded`、`answer_relevant`、grounded overlap 和 answer-question overlap。
- 循环控制：已在配置中加入 `max_rewrites`、`min_relevance_score`、`min_relevant_chunks`、`min_grounded_overlap`。
- Service 集成：`RagService.query(..., use_graph=True)` 统一进入 graph；非 source lookup 的检索由配置好的 retriever adapter 执行。
- 测试：已覆盖 source lookup 不触发 embedding/LLM、普通问题触发 retrieve/grade/generate、source chunks inspect、路由规则、source hint 提取、rewrite retry 和 insufficient context 分支。

当前边界：

- `grade_documents` 和 `grade_generation` 运行时默认仍是规则型实现，适合项目复盘和本地调试；离线评估已增加 `ragas-eval`，可用 DeepSeek/OpenAI-compatible judge 计算 RAGAS 指标。
- Source summarize：当用户问“某个文件讲了什么”时，当前只返回 chunk 摘要列表；后续可增加一个显式 summarization 节点，并把是否调用 LLM 写入 trace。
- Trace 输出能解释 graph path 和关键判定，但还不是完整的生产级观测系统。

验收标准：

- Source lookup 类问题不触发 embedding 查询和 LLM 生成。
- 普通问题的 trace 能解释这次走了哪条 graph path。
- 检索失败时能自动改写问题并重试一次。
- 超过改写预算仍没有上下文时，返回“不足以回答”，而不是强答。

## v0.5：API 服务化与前端展示

目标：把 CLI 能力稳定暴露给本地服务或轻量前端。

技术改造：

- 安装可选依赖 `fastapi/uvicorn` 后启用 API。
- API 增加 request id 和错误码；错误响应统一为 `{error_code, message, details}`。
- 增加 streaming query：先返回检索结果，再流式返回答案。
- 增加 collection 管理接口：创建、列出、删除、查看统计。
- 增加简单 Web UI 或接入现有前端：支持上传或选择本地目录、查看 chunk、发起 query、查看 citations 和 trace。
- 安全边界：API 默认只允许访问配置中的 workspace roots，不接受任意绝对路径。

验收标准：

- API 与 CLI 使用同一个 `RagService`，不复制业务逻辑。
- 前端显示答案时同时显示 source path、chunk index、score、snippet。
- 错误时不泄露 API key、完整 prompt 或敏感本地路径。

## v0.6：合规、观测与发布质量

目标：让项目具备更完整的工程展示水平。

技术改造：

- 日志：结构化记录 ingest/query/eval，不记录完整密钥，不默认记录完整原文。
- 成本统计：记录 prompt token、completion token、embedding batch 数、LLM latency。
- 数据合规：完善 `.ragignore`，默认跳过 `.env`、key、证书、浏览器数据、隐藏目录、大文件。
- 测试：补单元测试、Qdrant 临时目录集成测试、CLI smoke test、API test。
- CI：跑语法、测试、lint；不跑真实 LLM 调用。
- 文档：补架构图、模块边界、配置参考、故障排查、实验报告模板。

验收标准：

- 新用户能按 README 在本地复现 sample ingest/query。
- CI 不依赖外部 API key。
- 敏感文件不会被默认索引。
- 运行失败时能从日志和文档中定位是配置、依赖、索引、检索还是 LLM 问题。
