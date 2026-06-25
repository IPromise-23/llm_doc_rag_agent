"""
规则性质量判断模块  低成本质量控制层
利用 token 重叠、检索分数、阈值来做轻量判断
grade_documents     判断 检索到的 chunks 是否与问题相关
grade_generation    判断 生成答案是否被上下文支持，是否回应了问题
"""
from __future__ import annotations 

from dataclasses import dataclass

from llm_doc_rag_agent.retrieval.bm25 import tokenize
from llm_doc_rag_agent.schemas import RetrievedChunk


STOPWORDS = {   # 停用词表，放入十分常见、区分度的词    质量判断如果把这类词都算进去就会产生很多假相关
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "这个",
    "那个",
    "哪些",
    "什么",
    "如何",
    "怎么",
    "是否",
    "可以",
    "当前",
    "项目",
}


@dataclass(frozen=True)
class DocumentGrade:                            # 检索结果质量
    filtered_documents: list[RetrievedChunk]    # 通过质量判断、被认为相关的 chunks
    relevant_count: int                         # 相关 chunk 数量
    retrieved_count: int                        # 原始检索结果数量
    max_score: float                            # 检索结果中的最高分
    query_terms: list[str]                      # 从 query 中提取出的有效词 token
    decision: str                               # 决定后续 graph 该怎么走
    reason: str                                 # 为什么 graph 要这么走


@dataclass(frozen=True)
class AnswerGrade:                              # 生成答案质量
    grounded: bool                              # 答案是否被上下文支持
    relevant: bool                              # 答案是否回应了问题
    grounded_overlap_ratio: float               # 答案和上下文重叠了多少
    answer_question_overlap_ratio: float        # 答案和问题重叠了多少
    answer_terms: list[str]
    context_terms: list[str]
    question_terms: list[str]                   # 这三个参数作为调试使用，描述规则判断时抽取了哪些有效 token
    decision: str = "accept"                    # accept / regenerate / rewrite_query
    reason: str = ""


def meaningful_terms(text: str) -> list[str]:   # 文本 ---> 有意义的 token 列表
    """Tokenize text and remove tiny/common terms for lightweight graders."""

    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if token in STOPWORDS or len(token) <= 1 and not _is_cjk(token):    # 过滤规则：停用词不要 or 长度小于等于 1 的非中文 token 不要
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)
    return terms


def grade_retrieved_documents(  # 规则型 CRAG gate ，判断检索结果够不够相关
    query: str,
    retrieved: list[RetrievedChunk],
    min_relevance_score: float = 0.05,  # 最低检索分数阈值
    min_relevant_chunks: int = 1,       # 至少要有多少个相关 chunk 才能接受
) -> DocumentGrade:
    """Rule-based CRAG-style document relevance gate.

    A chunk is relevant when it has enough lexical overlap with the active query
    or when its retriever score passes the configured floor. The score floor is
    intentionally conservative and configurable because dense, BM25, and hybrid
    scores live on different scales.
    """

    query_terms = meaningful_terms(query)
    query_term_set = set(query_terms)   # 提取 query 的有效词并转为 set
    filtered: list[RetrievedChunk] = []
    max_score = max((item.score for item in retrieved), default=0.0)    # 检索结果最高分

    for item in retrieved:
        chunk_terms = set(meaningful_terms(item.chunk.text))
        overlap = len(query_term_set & chunk_terms)             # 计算 query 和 chunk 有多少共同有效词
        lexical_match = bool(query_terms) and overlap > 0       # 只要 query 有有效词，而且 chunk 至少命中了一个 query 词，那么就有词法匹配     lexical_match 是 bool 类型 
        score_match = item.score >= min_relevance_score         # 如果检索分数超过阈值，也认为它可用
        if lexical_match or score_match:                        # 词法命中或者分数过线，满足其中之一就可以保留
            filtered.append(item)

    relevant_count = len(filtered)                              # 统计相关结果数量
    if relevant_count >= min_relevant_chunks:
        decision = "accept"
        reason = "relevant_context_found"
    elif retrieved:                                             # 如果检索到了内容，但是质量不够，改写 query 重试
        decision = "rewrite"
        reason = "retrieved_context_below_relevance_threshold"
    else:
        decision = "rewrite"
        reason = "no_context_retrieved"
    return DocumentGrade(
        filtered_documents=filtered,
        relevant_count=relevant_count,
        retrieved_count=len(retrieved),                         # retrieved_count 一般大于等于 relevant_count
        max_score=float(max_score),
        query_terms=query_terms,
        decision=decision,
        reason=reason,
    )


def rewrite_query(question: str, previous_query: str | None = None) -> str:
    """Produce a deterministic fallback query for one CRAG retry."""

    source = previous_query or question
    terms = meaningful_terms(source)
    rewritten = " ".join(terms[:12]).strip() or question.strip()
    if rewritten == source.strip():
        rewritten = f"{rewritten} implementation configuration source code".strip() # 如果改写结果和原 query 一样，就追加几个工程检索常用词，目的是让检索 query 更偏向代码/配置/实现细节
    return rewritten


def grade_answer(
    question: str,
    answer: str,
    contexts: list[str],
    min_grounded_overlap: float = 0.2,
) -> AnswerGrade:
    """Rule-based Self-RAG-style answer support check for trace/debugging."""

    answer_terms = meaningful_terms(answer)
    context_terms = meaningful_terms(" ".join(contexts))
    question_terms = meaningful_terms(question)
    context_term_set = set(context_terms)
    question_term_set = set(question_terms)
    answer_term_set = set(answer_terms)

    grounded_overlap = _overlap_ratio(answer_term_set, context_term_set)    # 答案中的有效词比例 ---> 答案是否可以被上下文支持
    question_overlap = _overlap_ratio(question_term_set, answer_term_set)   # 问题里的有效词比例 ---> 答案是否回应了问题
    insufficient_answer = any(phrase in answer for phrase in ("不足以回答", "无法回答", "insufficient"))    # 如果答案明确说上下文不足也视为一种合规的回答
    grounded = insufficient_answer or grounded_overlap >= min_grounded_overlap      # 如果答案承认不足 or 答案上下文重叠比例超过阈值，就认为 grounded
    relevant = insufficient_answer or not question_terms or question_overlap > 0    # 如果答案承认不足，或者问题没有有效词，或者答案至少覆盖了问题中的一些有效词，就认为 relevant
    if grounded and relevant:
        decision = "accept"
        reason = "answer_grounded_and_relevant"
    elif relevant:
        decision = "regenerate"
        reason = "answer_relevant_but_not_grounded"
    else:
        decision = "rewrite_query"
        reason = "answer_not_relevant_to_question"
    return AnswerGrade(
        grounded=grounded,
        relevant=relevant,
        grounded_overlap_ratio=grounded_overlap,
        answer_question_overlap_ratio=question_overlap,
        answer_terms=answer_terms,
        context_terms=context_terms,
        question_terms=question_terms,
        decision=decision,
        reason=reason,
    )


def _overlap_ratio(source_terms: set[str], target_terms: set[str]) -> float:     # 重叠比例
    if not source_terms:
        return 0.0
    return len(source_terms & target_terms) / len(source_terms)


def _is_cjk(token: str) -> bool:    # 判断一个 token 是否为单个中文汉字
    return len(token) == 1 and "\u4e00" <= token <= "\u9fff"

"""
目前的 quality.py 并不是完整的评估系统，而是 Langgraph 运行时的轻量质量控制层，它只做两件事：
- 生成前：判断检索到的文档是否与问题相关
- 生成后：粗略判断答案是否被上下文支持、是否回应了问题

目前是规则形式的，后续可以升级为 LLM 来判断
"""
