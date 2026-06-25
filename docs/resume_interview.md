# 简历条目与面试 Q&A

这份文档用于把 `llm_doc_rag_agent` 转成简历项目描述和面试回答素材。表达策略：强调工程化实践、Agent/RAG 主链路、技术取舍和可验证结果，不夸大成企业级平台。

## 简历项目条目

### 版本 A：偏工程实现

**本地技术文档 RAG Agent | Python, Qdrant, LangGraph, SentenceTransformer, DeepSeek-compatible LLM**

- 构建面向本地技术文档的 RAG Agent，支持文档加载、确定性切块、embedding、Qdrant 本地持久化、引用生成、CLI/API/eval 入口。
- 实现增量索引与 source 管理，通过 `document_hash` 跳过未变化文件，支持 source 删除、重建、collection inspect，并用 `.ragignore` 规避敏感文件索引。
- 实现 dense、BM25、hybrid RRF 与可选 reranker 入口，支持多检索策略横向 eval 和 Markdown 报告输出。
- 基于 LangGraph 实现 `source_lookup/direct_answer/retrieve_rag` 自适应路由，并加入 CRAG/Self-RAG 质量门，支持检索相关性判断、query rewrite、答案 groundedness 判断、重答和重新检索。
- 编写单元测试覆盖 loader、chunking、retrieval、service、graph routing、rewrite retry、生成重试、重新检索与 insufficient context 分支，当前核心测试 `40 passed`。

### 版本 B：偏 Agent/RAG 方向

**本地技术文档 RAG Agent | LangGraph + Qdrant**

- 从 RAG、Qdrant 检索评估和 LangGraph notebook 学习代码出发，重构为模块化 Python 工程，沉淀 loader、retriever、vector store、agent graph、service、CLI/eval 等边界。
- 设计 hybrid retrieval pipeline，结合 dense semantic search 与 BM25 lexical search，并用 RRF 融合候选结果，改善函数名、配置项、命令类问题的召回。
- 使用 LangGraph 构建自适应问答流程：source 查询直接走 metadata lookup，普通问题走 RAG 分支，并在 trace 中暴露 graph path、检索策略和自检结果。
- 实现轻量 CRAG/Self-RAG 控制策略：检索上下文不足时自动改写 query 并重试，生成后由 hybrid/LLM judge 判定是否结束、重答或重新检索，超过预算后返回 insufficient context。

### 版本 C：更短的一栏版

**llm_doc_rag_agent：本地技术文档 RAG Agent**  
基于 Python、Qdrant、SentenceTransformer、LangGraph 实现本地文档 RAG Agent，支持增量索引、dense/BM25/hybrid 检索、多策略 eval、source lookup、CRAG/Self-RAG 质量门和 CLI/API 入口；通过单元测试覆盖核心检索、路由、query rewrite、生成重试、重新检索与 insufficient context 分支。

## 30 秒自我介绍版

我做了一个本地技术文档 RAG Agent，主要是把之前学习的 RAG、Qdrant 检索评估和 LangGraph 编排，从 notebook 改造成一个模块化工程。它支持本地文档索引、Qdrant 持久化、dense/BM25/hybrid 检索、LangGraph 自适应路由，以及 CRAG/Self-RAG 质量门。这个项目的重点不是做一个聊天 demo，而是把 RAG Agent 的索引、检索、路由、trace、eval 和测试都串成一个能复盘的工程实践。

## 面试追问 Q&A

### 1. 为什么用 Qdrant？

Qdrant 适合做本地持久化向量检索，Python client 也比较直接。这个项目里我使用 `QdrantClient(path=...)`，让索引写在本地目录，适合个人实践项目和本地知识库，不需要先部署远程向量数据库。它还支持 payload 存储，所以每个 chunk 可以保留 `source_path`、`chunk_index`、`content_hash`，后续做 citations、source 删除和 collection inspect 都比较方便。

### 2. 为什么不只做 dense retrieval，还要 BM25/hybrid？

Dense retrieval 对语义相似问题更好，但技术文档里经常有函数名、配置项、命令、错误码，这类精确 token 不一定靠 embedding 召回稳定。BM25 对这类词法匹配更敏感。Hybrid RRF 的作用是把 dense 和 BM25 的排名融合起来，在不强行比较两种分数尺度的情况下，提高召回鲁棒性。

### 3. LangGraph 在这个项目里解决了什么？

如果只是 `retrieve -> generate`，普通函数调用就够了。LangGraph 的价值在于显式表达流程分支和循环：项目里入口先判断问题类型，source lookup 类问题直接读 metadata，不走 LLM；普通问题进入 RAG 分支，再经过 `grade_documents`，必要时 `rewrite_query` 后重试。这样 graph path 本身就是可解释 trace，能说明每次回答是怎么来的。

### 4. CRAG/Self-RAG 为什么用 hybrid judge？

主要是成本和可测试性。项目现在采用 hybrid gate：真实运行时有 API key 就用 LLM judge 判断文档相关性、答案 groundedness/relevance 和下一步动作；无 key 或调用失败时回到规则判断。这样既能展示 LLM-as-judge 的 Agent 控制流，也保证单元测试不依赖外部服务。

### 5. 如何评估 RAG 效果？

项目里先做了可落地的基础评估：eval runner 读取 CSV dataset，对同一问题运行不同 retriever，比如 dense、BM25、hybrid RRF，然后输出 JSONL/CSV 和 Markdown 报告，便于对比上下文、答案和 trace。现在还接入了离线 RAGAS 评估入口，可以计算 faithfulness、answer relevancy、context precision/recall，并用 DeepSeek/OpenAI-compatible judge 做自动指标评估。

### 6. 如何降低幻觉风险？

首先 prompt 要求只基于上下文回答。其次 graph 里有 `grade_documents`，如果检索结果为空或低相关，会先 rewrite query 重试。生成后 `grade_generation` 还会用 hybrid/LLM judge 判断答案是否 grounded 和 relevant：合格才结束，不合格会带反馈重答或重新检索。超过预算仍不通过时返回 insufficient context，而不是继续让 LLM 强答。

### 7. 项目最大的工程难点是什么？

最大难点是把 notebook 中线性的学习代码改造成模块化工程：需要明确 loader、splitter、embedding、vector store、retriever、agent graph、service、CLI/eval 的边界。另一个难点是控制 Agent 不要无限循环，所以 query rewrite 有 `max_rewrites`，生成重答有 `max_generation_retries`，source lookup 也单独分支，避免不必要的 LLM 调用。

### 8. 如果继续优化，你会做什么？

优先做三件事：第一，把 BM25 从临时 payload 计算升级成 Qdrant sparse/named vector 的持久化混合检索；第二，继续增强 judge prompt、失败诊断和评估集，让 LLM-as-judge 的决策更可解释；第三，补一个轻量 UI 或 API streaming，让用户能看到答案、citations、retrieved chunks 和 graph path。

### 9. 这个项目和普通 RAG demo 的区别是什么？

普通 RAG demo 往往只有读取文档、向量检索、调用 LLM。这个项目多了工程边界：增量索引、source 删除重建、安全忽略、多检索策略、eval 报告、LangGraph 路由、query rewrite、insufficient context、trace 和单元测试。它不追求企业级平台，但已经能展示完整 RAG Agent 的工程思路。

### 10. 你在项目里学到了什么？

我学到 RAG 项目真正难的不是只接一个向量库，而是如何处理索引更新、检索策略选择、上下文质量判断、回答可追溯和失败兜底。LangGraph 这类框架的价值也不是“更像 Agent”，而是让路由、循环和状态更新显式化，方便调试和复盘。

## 面试时不要这样说

- 不要说“实现了企业级 RAG 平台”；更准确说“实践级本地文档 RAG Agent”。
- 不要说“CRAG/Self-RAG 已经完全实现论文方案”；更准确说“实现了 hybrid LLM judge + 规则兜底的轻量质量门和控制流”。
- 不要说“评估指标已经企业级完整”；更准确说“支持多检索策略对比、报告输出和离线 RAGAS 指标评估，运行时 gate 采用 LLM judge 和规则兜底结合”。
- 不要说“项目已经生产可用”；更准确说“工程边界完整，适合作为实习项目展示和继续迭代的基础”。
