"""
BM25 不依靠 embedding 向量，而是靠词项匹配、词频、文档频率来算分数，因此需要先把文本切成 token
BM25 会考虑文档长度，很长的文档中天然更容易包含 query 的词项
idf 反文档频率，越少见的词越能区分文档，自然权重越高
一个 chunk 的 BM25 分数 = query 中每个词在这个 chunk 中的匹配贡献之和

BM25 的本质就是：精确匹配 query 词，但同时奖励稀有词、控制重复词、校正文档长度

retrieve 函数中：
top_k           最终想要返回给下游的结果数量
candidate_k     中间候选数量，通常给 hybrid/rerank 用
这里要同时拥有两个参数的目的是 统一接口，方便被 HybridRetriever 调用
在纯 BM25 场景下可以只传 top_k      在 hybrid/rerank 场景下可以传 candidate_k
"""
from __future__ import annotations

import math
import re
from collections import Counter

from llm_doc_rag_agent.schemas import Chunk, RetrievedChunk
from llm_doc_rag_agent.vectorstores import QdrantVectorStore


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE) # 匹配连续的英文、数字、下划线  匹配单个中文汉字    按照 Unicode 规则处理文本


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]  # 找出文本中所有匹配的 token 并返回列表


class BM25Retriever:
    """Lightweight lexical retriever built from existing Qdrant chunk payloads."""

    def __init__(self, store: QdrantVectorStore, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[RetrievedChunk]:
        chunks = self.store.list_chunks(limit=None) # 把库中所有 chunks 取出来 --> 交给 rank() 打分并排序
        return self.rank(query=query, chunks=chunks, top_k=candidate_k or top_k)

    def rank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        if not query_terms or not chunks:
            return []

        tokenized_docs = [tokenize(chunk.text) for chunk in chunks] # 把每个 chunk 的正文都分词，嵌套列表
        doc_lengths = [len(tokens) for tokens in tokenized_docs]    # 统计每个 chunk 分词后有多少个 token 
        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0   # 计算平均文档长度
        document_frequency = Counter(term for tokens in tokenized_docs for term in set(tokens)) # 1. 拿出每个 chunk 文档中所有的词项，最多出现一次  2. 统计每个 term 出现了多少次   ---> document_frequency 统计这个词出现在多少个 chunk 中
        query_counts = Counter(query_terms) # 统计 query 中每个词出现的次数，query 中重复出现的词会有更高的权重
        scored: list[RetrievedChunk] = []
        for chunk, tokens, doc_len in zip(chunks, tokenized_docs, doc_lengths, strict=True):
            term_frequency = Counter(tokens)    # 统计（当前）单个 chunk 的每个词出现的次数
            score = 0.0
            for term, query_weight in query_counts.items(): # query_counts 是一个 dict{"token":int}
                tf = term_frequency.get(term, 0)    # 如果 query 中的 token 不在当前 chunk 中出现 tf 就为 0
                if tf == 0:
                    continue
                df = document_frequency.get(term, 0)    # 当前 query token 出现在多少个 chunk 中
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))   # 反文档频率
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(avg_doc_len, 1e-9))   # 长度归一化
                score += query_weight * idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score, retriever_type="bm25"))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]    # 按照分数从高到低排序，并截断

"""
tf ———— query 的当前 token 在当前 chunk 中出现的次数
doc_len ———— 当前 chunk 共有多少个 token 数量
avg_doc_len ———— 所有 chunk 的平均 token 数
k1 ———— 控制词频饱和
b ————控制长度归一化程度

query_weight ———— query 中这个 token 出现次数，出现越多权重越高
idf ———— 这个 token 越是稀有，权重越高
(tf * (self.k1 + 1)) / denominator  当前 chunk 中这个 token 出现越多分数越高
"""