"""
这个脚本是整个 RAG agent 的流程控制器，用 Langgraph 把下面这些能力串起来：

用户问题：
    -> 判断走哪条路线
    -> 直接回答 or 查 source or 正常 RAG 检索
    -> 检索
    -> 判断文档质量
    -> 必要时改写 query 再检索
    -> 上下文足够则生成答案
    -> 判断答案是否 grounded or relevant
    -> 返回 Answer + trace

最终生成的是一个可运行的 LangGraph 工作流   流程编排脚本

direct_answer     直接回答帮助类问题
source_lookup     查询 sources / chunks
retrieve_rag      正常 RAG 检索问答
"""
from __future__ import annotations 

import re
from collections.abc import Callable    # collections.abc 是集合抽象基类库  Callable 是类型注解标识，用于标注函数参数、变量、返回值为可调用对象，配合静态类型检查使用
from pathlib import Path
from typing import Any, TypedDict       # typing.Any 类型注解工具，表示任意数据类型     typing.TypeDict 专门用于给字典做结构化类型注解

from langgraph.graph import END, START, StateGraph

from llm_doc_rag_agent.agents.quality import (
    grade_answer,
    grade_retrieved_documents,
    rewrite_query as rewrite_query_text,
)
from llm_doc_rag_agent.generation import QAService
from llm_doc_rag_agent.schemas import Answer, Chunk, Citation, RetrievedChunk
from llm_doc_rag_agent.utils import safe_snippet


class RagGraphState(TypedDict, total=False):    # 图状态字典：LangGraph 在节点之间传递的状态对象    TypedDict 表示它本质上还是 dict 但是会有类型提示    total = False 表示这些字段并非必须同时出现
    question: str
    search_query: str
    top_k: int
    retriever_type: str
    candidate_k: int | None
    rewrite_count: int
    retrieved: list[RetrievedChunk]
    filtered_documents: list[RetrievedChunk]    # 质量判断后保留下来的 chunks
    generation: str                             # 生成出来的答案文本
    answer: Answer                              # 完整的 Answer 对象
    route: str                                  # 第一步路由结果
    decision: str
    document_decision: str                      # 文档质量判断结果
    graph_path: list[str]                       # 记录走过哪些节点
    trace: dict[str, Any]                       # 调试黑匣子，记录路由、检索、评分、生成过程
    sources: list[str]
    source_path: str
    chunks: list[Chunk]
    errors: list[str]
    error: str

# 给函数类型起一个别名
SourceLister = Callable[[int | None], list[str]]                # 回调函数类型，接收 int | None ，返回 list[str]    作用是列出当前 collection 中有哪些 source
SourceChunkGetter = Callable[[str, int | None], list[Chunk]]    # 给定某个 source path ，取出它的 chunks

_SOURCE_EXTENSIONS = ("md", "txt", "py", "ipynb", "rst", "pdf") # 哪些文件后缀被认为是可识别的 source
_SOURCE_HINT_RE = re.compile(                                   # 这个正则用来从用户问题中抓取文件名或路径，把匹配到的内容命名为 hint ，方便后面用 match.group("hint")
    rf"(?P<hint>[A-Za-z0-9_./~:@-]+\.({'|'.join(_SOURCE_EXTENSIONS)}))"
)


def route_question(question: str) -> str:       # 决定问题走哪一条分支
    """Choose the first graph branch with lightweight, deterministic rules."""

    normalized = question.strip().lower()       # strip() 只会删除开头和结尾的空白字符
    compact = re.sub(r"\s+", "", normalized)    # re.sub(pattern, replacement, text) 正则替换，这里是把所有空白字符删掉
    if not normalized:
        return "direct_answer"  # 用户问题为空的分支

    source_list_terms = (
        "有哪些文档",
        "列出文档",
        "文档列表",
        "文件列表",
        "资料来源",
        "来源列表",
        "列出来源",
        "有哪些来源",
        "有哪些文件",
        "已索引",
        "索引了哪些",
    )
    source_detail_terms = (
        "这个文件",
        "某个文件",
        "这份文档",
        "文件讲了什么",
        "文档讲了什么",
        "查看chunk",
        "列出chunk",
        "chunks",
    )
    english_source_lookup = (
        re.search(r"\b(list|show|inspect)\s+(sources|documents|files|chunks)\b", normalized)
        or re.search(r"\b(sources|documents|files|chunks)\s+(list|indexed|available)\b", normalized)
        or "source=" in normalized
        or "source:" in normalized  # or 的短路逻辑，只要其中一个条件成立整个表达式就成立   匹配到了或者类似 source 的字样出现在 normalized 中就返回 True
    )
    if (
        any(term in compact for term in source_list_terms)
        or any(term in compact for term in source_detail_terms)
        or english_source_lookup
    ):
        return "source_lookup"  # 问 source / 文件 / chunks 分支

    direct_terms = ("help", "usage", "/help", "你能做什么", "怎么使用", "如何使用")
    if normalized in direct_terms or any(term in compact for term in direct_terms[3:]): # 前一个逻辑是用户的整个问题刚好等于这些词之一才成立，负责精确匹配简单命令    后面的逻辑是子串匹配，负责宽松匹配中文自然语言
        return "direct_answer"  # 问 help / usage / 怎么使用之类的问题的分支

    return "retrieve_rag"       # 其他普通问题分支


def extract_source_hint(question: str) -> str | None:   # 从问题中提取 source 线索（提取可能的文件名或路径）
    """Extract a likely source path or basename from a source lookup question."""

    explicit_patterns = (
        r"(?:source|path|file)\s*[:=]\s*(?P<hint>[^，,\s]+)",
        r"(?:来源|文件|文档|路径)\s*[：:=]\s*(?P<hint>[^，,\s]+)",  # 两个正则模式，分别用来匹配英文和中文
    )
    for pattern in explicit_patterns:   # 循环两个正则
        match = re.search(pattern, question, flags=re.IGNORECASE)   # 匹配到了就取出 hint
        if match:
            return _clean_hint(match.group("hint"))                 # 清理后返回

    for quoted in re.findall(r"[`\"'“”‘’]([^`\"'“”‘’]+)[`\"'“”‘’]", question):  # re.findall() 会返回所有匹配结果列表，这里是在找引号、反引号中的内容
        if _looks_like_source_hint(quoted):
            return _clean_hint(quoted)

    match = _SOURCE_HINT_RE.search(question)    # 从整句话中找到文件路径
    if match:
        return _clean_hint(match.group("hint"))
    return None


def build_rag_graph(
    retriever: Any,
    qa: QAService,
    list_sources: SourceLister | None = None,           # collection 中有哪些 source
    chunks_for_source: SourceChunkGetter | None = None, # 对应 source_path 中有哪些 chunks 
    quality_grader: Any | None = None,
    max_rewrites: int = 1,
    min_relevance_score: float = 0.05,
    min_relevant_chunks: int = 1,
    min_grounded_overlap: float = 0.2,
):
    """Build an adaptive LangGraph flow for document RAG."""

    quality_grader_name = str(getattr(quality_grader, "name", "rule"))  # 如果 quality_grader 中有 name 属性就取它，如果没有就用 rule ，在用 str() 转写成字符串

    def route(state: RagGraphState) -> dict[str, Any]:      # state ---> dict ，观察用户问题应该走哪条路线并初始化后续流程需要的状态字段
        selected_route = route_question(state["question"])  # 从 state 状态字典中取出用户问题后选择路径
        path = _append_path(state, "route_question")        # 记录 图路径 ：把当前节点追加到 graph_path 中
        return {
            "route": selected_route,                                # 保存路由结果，控制下一步图该怎么走 （图内部路由）
            "decision": selected_route,                             # 对外调试/记录，告诉调试者这一步做出的决策是什么
            "search_query": state["question"],
            "rewrite_count": int(state.get("rewrite_count") or 0),
            "graph_path": path,
            "trace": _merge_trace(                                  # 把需要更新的值放入到旧的 trace 中
                state,                                              # state 时运行时控制流程用的，trace 是最终给调试、日志、前段展示的
                {
                    "route": selected_route,
                    "decision": selected_route,
                    "graph_path": path,
                },
            ),
        }

    def direct_answer(state: RagGraphState) -> dict[str, Any]:
        path = _append_path(state, "direct_answer")
        answer = Answer(
            question=state["question"],
            answer=(
                "我可以基于已索引的本地技术文档回答问题，也可以列出当前 collection "
                "中的 sources，或查看指定 source 的 chunks。"
            ),
            citations=[],   # 没有引用
            contexts=[],    # 没有上下文
            trace=_merge_trace(
                state,
                {
                    "route": "direct_answer",
                    "graph_path": path,
                    "context_count": 0,
                    "retrieval_skipped": True,
                },
            ),
        )
        return {"answer": answer, "graph_path": path, "trace": answer.trace}

    def source_lookup(state: RagGraphState) -> dict[str, Any]:  # 用户问 有哪些文档、查看某个文件、列出某个 source 的 chunks 时走的逻辑
        path = _append_path(state, "source_lookup")
        answer = _source_lookup_answer(
            question=state["question"],
            graph_path=path,
            state=state,
            list_sources=list_sources,
            chunks_for_source=chunks_for_source,
        )
        return {"answer": answer, "graph_path": path, "trace": answer.trace}

    def retrieve(state: RagGraphState) -> dict[str, Any]:
        question = state.get("search_query") or state["question"]
        top_k = int(state.get("top_k") or 5)
        path = _append_path(state, "retrieve")
        retrieved = retriever.retrieve(question, top_k=top_k)
        retriever_types = sorted({item.retriever_type for item in retrieved}) or [  # 遍历所有检索回来的 RetrievedChunk ，取出每个 chunk 的 retriever_type 并自动去重
            str(state.get("retriever_type") or "unknown")
        ]
        return {
            "retrieved": retrieved,
            "graph_path": path,
            "trace": _merge_trace(
                state,
                {
                    "route": "retrieve_rag",
                    "graph_path": path,
                    "retriever_type": ",".join(retriever_types),
                    "top_k": top_k,
                    "candidate_k": state.get("candidate_k"),
                    "context_count": len(retrieved),
                    "search_query": question,
                },
            ),
        }

    def grade_documents(state: RagGraphState) -> dict[str, Any]:
        path = _append_path(state, "grade_documents")
        active_query = state.get("search_query") or state["question"]
        retrieved = state.get("retrieved", [])
        if quality_grader is None:  # 如果没有配置 quality_grader 就用规则评分
            grade = grade_retrieved_documents(
                query=active_query,
                retrieved=retrieved,
                min_relevance_score=min_relevance_score,
                min_relevant_chunks=min_relevant_chunks,
            )
        else:
            grade = quality_grader.grade_documents(
                query=active_query,
                retrieved=retrieved,
                min_relevance_score=min_relevance_score,
                min_relevant_chunks=min_relevant_chunks,
            )
        return {
            "filtered_documents": grade.filtered_documents,
            "document_decision": grade.decision,
            "graph_path": path,
            "trace": _merge_trace(
                state,
                {
                    "graph_path": path,
                    "document_grade_decision": grade.decision,
                    "document_grade_reason": grade.reason,
                    "retrieved_count": grade.retrieved_count,
                    "relevant_count": grade.relevant_count,
                    "max_retrieval_score": grade.max_score,
                    "query_terms": grade.query_terms,
                    "min_relevance_score": min_relevance_score,
                    "min_relevant_chunks": min_relevant_chunks,
                    "rewrite_count": int(state.get("rewrite_count") or 0),
                    "quality_grader": quality_grader_name,
                },
            ),
        }

    def rewrite_query(state: RagGraphState) -> dict[str, Any]:
        path = _append_path(state, "rewrite_query")
        rewrite_count = int(state.get("rewrite_count") or 0) + 1
        previous_query = state.get("search_query") or state["question"]
        if quality_grader is None:
            rewritten = rewrite_query_text(state["question"], previous_query=previous_query)
        else:
            rewritten = quality_grader.rewrite_query(state["question"], previous_query=previous_query)
        return {
            "search_query": rewritten,
            "rewrite_count": rewrite_count,
            "graph_path": path,
            "trace": _merge_trace(
                state,
                {
                    "graph_path": path,
                    "rewrite_count": rewrite_count,
                    "previous_query": previous_query,
                    "rewritten_query": rewritten,
                    "quality_grader": quality_grader_name,
                },
            ),
        }

    def insufficient_context(state: RagGraphState) -> dict[str, Any]:   # 检索出来的上下文不够，不能可靠回答
        path = _append_path(state, "insufficient_context")
        retrieved = state.get("filtered_documents") or state.get("retrieved", [])
        answer = Answer(
            question=state["question"],
            answer="没有检索到足够相关的上下文，当前无法基于已索引文档可靠回答这个问题。",
            citations=_citations_from_retrieved(retrieved),
            contexts=[item.chunk.text for item in retrieved],
            trace=_merge_trace(
                state,
                {
                    "route": "retrieve_rag",
                    "graph_path": path,
                    "final_decision": "insufficient_context",
                    "context_count": len(retrieved),
                    "max_rewrites": max_rewrites,
                    "retrieval_skipped": False,
                },
            ),
        )
        return {"answer": answer, "graph_path": path, "trace": answer.trace}

    def generate(state: RagGraphState) -> dict[str, Any]:
        path = _append_path(state, "generate")
        retrieved = state.get("filtered_documents") or state.get("retrieved", [])
        answer = qa.answer(state["question"], retrieved)
        trace = dict(answer.trace)
        trace.update(
            _merge_trace(
                state,
                {
                    "route": "retrieve_rag",
                    "graph_path": path,
                    "top_k": int(state.get("top_k") or 5),
                    "candidate_k": state.get("candidate_k"),
                    "context_count": len(retrieved),
                    "search_query": state.get("search_query") or state["question"],
                },
            )
        )
        return {
            "answer": Answer(
                question=answer.question,
                answer=answer.answer,
                citations=answer.citations,
                contexts=answer.contexts,
                trace=trace,
            ),
            "generation": answer.answer,
            "graph_path": path,
            "trace": trace,
        }

    def grade_generation(state: RagGraphState) -> dict[str, Any]:
        path = _append_path(state, "grade_generation")
        answer = state["answer"]
        if quality_grader is None:
            grade = grade_answer(
                question=state["question"],
                answer=answer.answer,
                contexts=answer.contexts,
                min_grounded_overlap=min_grounded_overlap,
            )
        else:
            grade = quality_grader.grade_answer(
                question=state["question"],
                answer=answer.answer,
                contexts=answer.contexts,
                min_grounded_overlap=min_grounded_overlap,
            )
        trace = dict(answer.trace)
        trace.update(
            _merge_trace(
                state,
                {
                    "graph_path": path,
                    "answer_grounded": grade.grounded,
                    "answer_relevant": grade.relevant,
                    "grounded_overlap_ratio": grade.grounded_overlap_ratio,
                    "answer_question_overlap_ratio": grade.answer_question_overlap_ratio,
                    "min_grounded_overlap": min_grounded_overlap,
                    "final_decision": "generated",
                    "quality_grader": quality_grader_name,
                },
            )
        )
        return {
            "answer": Answer(
                question=answer.question,
                answer=answer.answer,
                citations=answer.citations,
                contexts=answer.contexts,
                trace=trace,
            ),
            "graph_path": path,
            "trace": trace,
        }

    def choose_route(state: RagGraphState) -> str:                      # 选择路由的函数，决定从 route_question 之后走哪一条边
        return state.get("route") or "retrieve_rag"

    def choose_after_document_grade(state: RagGraphState) -> str:       # 选择评估文档质量之后该执行什么节点的函数
        if state.get("document_decision") == "accept":
            return "generate"
        if int(state.get("rewrite_count") or 0) < max_rewrites:
            return "rewrite_query"
        return "insufficient_context"

    workflow = StateGraph(RagGraphState)                                # 创建一张状态图
    workflow.add_node("route_question", route)                          # 添加节点，左边字符串是节点名，右边是实际执行的函数
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("source_lookup", source_lookup)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("insufficient_context", insufficient_context)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_edge(START, "route_question")                          # .add_edge("a","b") 普通边，固定从 a 到 b
    workflow.add_conditional_edges(                                     # .add_conditional_edges("a",choose_function,{"branch_name":"node_name"},)
        "route_question",
        choose_route,
        {
            "direct_answer": "direct_answer",
            "source_lookup": "source_lookup",
            "retrieve_rag": "retrieve",
        },
    )
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("source_lookup", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        choose_after_document_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "insufficient_context": "insufficient_context",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("insufficient_context", END)
    workflow.add_edge("generate", "grade_generation")
    workflow.add_edge("grade_generation", END)
    return workflow.compile()   # 将图编译成可执行的对象


def _source_lookup_answer(                          # 列出 sources / 查看某个 source 的问题     入口函数：负责判断用户到底是想列出 source 列表，还是查某个 source
    question: str,
    graph_path: list[str],                          # 当前 LangGraph 走过的节点路径
    state: RagGraphState,                           # 整张图当前的状态字典
    list_sources: SourceLister | None,              # 回调函数，用来列出当前 collection 中有哪些 source
    chunks_for_source: SourceChunkGetter | None,    # 用来读取某个 source 的 chunks 
) -> Answer:
    base_trace = _merge_trace(  # 构造基础 trace ，不回答问题，准备调试信息
        state,
        {
            "route": "source_lookup",
            "graph_path": graph_path,
            "retrieval_skipped": True,
        },
    )
    if list_sources is None:    # 如果没有提供列 source 的回调函数，就直接返回错误说明
        return Answer(
            question=question,
            answer="当前 graph 没有配置 source lookup 回调，无法列出 sources。",
            citations=[],
            contexts=[],
            trace={**base_trace, "source_lookup_available": False},     # **base_trace 表示字典展开，这里整体上是在原 trace 基础上加一个字段
        )

    try:
        sources = list_sources(None)    # 调用 lsit_sources 获取所有 sources
    except Exception as exc:  # pragma: no cover - defensive boundary for CLI/API use.
        return Answer(
            question=question,
            answer=f"读取 sources 失败：{exc}",
            citations=[],
            contexts=[],
            trace={**base_trace, "error": str(exc), "source_lookup_available": True},
        )

    source_hint = extract_source_hint(question)                                         # 从用户问题中提取 source 线索（文件名 or 路径）    source_hint 是用户问题中可能出现的文件名或者路径
    matched_source = _match_source_hint(source_hint, sources) if source_hint else None  # 利用文件线索 source_hint 来匹配真实已索引的 source 路径
    if matched_source:  # 匹配到了 source 就进入 chunk 查询     意义：用户提到了某个文件，且该文件在已索引的 sources 中找到了，所以继续读取这个 source 的 chunks
        return _source_chunks_answer(
            question=question,
            graph_path=graph_path,
            state=state,
            sources=sources,
            source_hint=source_hint or matched_source,
            source_path=matched_source,
            chunks_for_source=chunks_for_source,
        )
    if source_hint:     # 用户提到了文件，但没有匹配到，即 matched_source == None ,该分支说明 source_hint 有值，但是 matched_source 为空
        preview = "\n".join(f"- {source}" for source in sources[:10])                       # 取前十个 source 路径展示
        suffix = "" if len(sources) <= 10 else f"\n... 还有 {len(sources) - 10} 个 source"   # sources 如果超过了十个，就提示还是多少个没展示
        hint_text = f"没有找到与 `{source_hint}` 匹配的已索引 source。"                         # 用户提到了文件，但 collection 中没有匹配项
        if sources:
            hint_text += f"\n可用 sources：\n{preview}{suffix}"                              # 如果 sources 列表非空，就把可用 sources 拼到回答里 
        return Answer(
            question=question,
            answer=hint_text,
            citations=[],
            contexts=sources,
            trace={
                **base_trace,
                "source_lookup_available": True,
                "source_hint": source_hint,
                "source_count": len(sources),
                "context_count": len(sources),
                "matched_source": False,
            },
        )
    # 用户没有指定文件，那就只是列 sources
    preview = sources[:50]
    if not sources:
        text = "当前 collection 中还没有可列出的 source。"
    else:
        rows = "\n".join(f"- {source}" for source in preview)
        suffix = "" if len(sources) <= len(preview) else f"\n... 还有 {len(sources) - len(preview)} 个 source"
        text = f"当前 collection 中有 {len(sources)} 个 source：\n{rows}{suffix}"
    return Answer(
        question=question,
        answer=text,
        citations=[],
        contexts=sources,
        trace={
            **base_trace,
            "source_lookup_available": True,
            "source_count": len(sources),
            "context_count": len(sources),
        },
    )


def _source_chunks_answer(      # 细致函数，负责某个 source 已经确定后，读取并展示它的 chunks
    question: str,
    graph_path: list[str],      # 当前 Langgraph 走过的节点路径，用于 trace 调试 
    state: RagGraphState,       # 整张图当前的状态字典
    sources: list[str],
    source_hint: str,
    source_path: str,
    chunks_for_source: SourceChunkGetter | None,    # 读取 chunks 的回调函数
) -> Answer:
    base_trace = _merge_trace(
        state,
        {
            "route": "source_lookup",
            "graph_path": graph_path,
            "retrieval_skipped": True,
            "source_hint": source_hint,
            "source_path": source_path,
            "source_count": len(sources),
        },
    )
    if chunks_for_source is None:
        return Answer(
            question=question,
            answer=f"找到了 source：{source_path}，但当前 graph 没有配置 chunk 读取回调。",
            citations=[],
            contexts=[],
            trace={**base_trace, "chunk_lookup_available": False},
        )

    try:
        chunks = chunks_for_source(source_path, 20) # 读取这个 source 的 chunks ，最多取 20个
    except Exception as exc:  # pragma: no cover - defensive boundary for CLI/API use.
        return Answer(
            question=question,
            answer=f"读取 source chunks 失败：{exc}",
            citations=[],
            contexts=[],
            trace={**base_trace, "error": str(exc), "chunk_lookup_available": True},
        )

    if not chunks:
        return Answer(
            question=question,
            answer=f"没有找到与 `{source_hint}` 匹配的已索引 chunks。",
            citations=[],
            contexts=[],
            trace={**base_trace, "chunk_lookup_available": True, "chunk_count": 0},
        )

    rows = "\n".join(f"- chunk {chunk.chunk_index}: {safe_snippet(chunk.text, limit=160)}" for chunk in chunks[:8])
    suffix = "" if len(chunks) <= 8 else f"\n... 还有 {len(chunks) - 8} 个 chunk"
    citations = [
        Citation(
            source_path=chunk.source_path,      # 引用来自哪个 source
            chunk_id=chunk.id,                  # 引用 chunk 的唯一 ID
            chunk_index=chunk.chunk_index,      # 引用 chunk 在原文档中的编号/索引
            score=1.0,
            snippet=safe_snippet(chunk.text),
        )
        for chunk in chunks
    ]
    return Answer(
        question=question,
        answer=f"找到 source：{source_path}\n共有 {len(chunks)} 个 chunks：\n{rows}{suffix}",
        citations=citations,
        contexts=[chunk.text for chunk in chunks],
        trace={
            **base_trace,
            "chunk_lookup_available": True,
            "chunk_count": len(chunks),
            "context_count": len(chunks),
        },
    )


def _append_path(state: RagGraphState, node_name: str) -> list[str]:    # 把当前节点名追加到 graph_path 中，记录当前流程已经经过了 xxx 节点     RAG agent 的路线可能是不同的，所以需要记录流程经过了哪些节点，方便调试
    return [*state.get("graph_path", []), node_name]    # e.g. 假设 state 中原来没有 graph_path ,那么有 [] ---> return [*[],"route_question"]


def _citations_from_retrieved(retrieved: list[RetrievedChunk]) -> list[Citation]:   # 把检索结果 RetrievedChunk 转成展示用的 Citation
    return [
        Citation(
            source_path=item.chunk.source_path,
            chunk_id=item.chunk.id,
            chunk_index=item.chunk.chunk_index,
            score=item.score,
            snippet=safe_snippet(item.chunk.text),
        )
        for item in retrieved
    ]


def _merge_trace(state: RagGraphState, update: dict[str, Any]) -> dict[str, Any]:   # 把 旧 trace 与 新 trace 合并
    return {**dict(state.get("trace") or {}), **update}     # ** 在 dict 中表示展开 dict


def _looks_like_source_hint(value: str) -> bool:    # 判断一个字符串是否像文件路径
    normalized = value.strip().lower()
    return "/" in normalized or any(normalized.endswith(f".{extension}") for extension in _SOURCE_EXTENSIONS)


def _clean_hint(value: str) -> str:                 # 1. 去掉前后空白   2. 去掉前后的引号、反引号、中文标点、英文标点
    return value.strip().strip("`\"'“”‘’。，,;；")


def _match_source_hint(source_hint: str | None, sources: list[str]) -> str | None:  # 利用用户给的文件线索 source_hint 来匹配真实已索引的 source 路径   比如：source_hint = "graph.py" ---> 经过本函数就匹配成真实的 sources = "/Users/.../graph.py"
    if not source_hint:                                                             # source_hint 用户给出的文件线索    sources:list[str] 系统里真是存在的 source 路径列表
        return None
    cleaned = _clean_hint(source_hint)  # 清理用户输入，去掉前后空白、引号、标点
    if cleaned in sources:              # 判断用户输入清理后，是否刚好就是一个完整的 source 路径
        return cleaned
    hint_name = Path(cleaned).name      # 取路径最后的文件名，用于支持用户自输入文件名，但是 sources 中为完整路径的情况
    matches = [
        source
        for source in sources           # 遍历每一个真实 source ，只要满足下列三种条件之一，就加入 matches
        if source.endswith(cleaned) or Path(source).name == hint_name or cleaned in source  # 匹配三种情况：source 以用户输入结尾   source 的文件名等于用户输入文件名   用户输入是 source 的一部分
    ]
    if not matches:
        return None
    return sorted(matches, key=len)[0]  # 把匹配项按长度排序，返回最短的那个，通常最短路径更精确

"""
source 是被引进 RAG 系统的原始来源文件，通常是一个文件路径      such as: /Users/xxx/Desktop/projects/src/graph.py
在 schemas.py 中，source_path 表示 Document 来自哪个文件 / chunk 来自哪个原始文件   --->    source_path 表示原始文件路径
sources 是已经索引的 source_path 的去重列表，因为一个 source 文件可以切分成很多个 chunks
在本脚本中，source_lookup 不是文件本身，而是一条 workflow 路线，它处理：列出当前有哪些文档等操作
source_hint 表示为用户输入里的 source 线索，这是从用户问题中提取出来的线索，并非准确路径
source_hint = 用户给的模糊文件线索      source_path = 系统真实记录的文件路径
"""