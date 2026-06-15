from __future__ import annotations  # 把所有类型注解当成 字符串注释 ，延迟求值

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True) # 装数据，对象创建后字段无法被重新赋值
class Document:
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)  # 使得每个类的实例都有独立的 dict


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_path: str    # 源文件路径，表示 chunk 来自哪个文件
    chunk_index: int
    content_hash: str   # 内容哈希  用于判断内容是否变了、避免重复、支持重新索引识别同一段文本
    metadata: dict[str, Any] = field(default_factory=dict)  # 保证每个 Chunk 都有自己独立的字典


@dataclass(frozen=True)
class RetrievedChunk:       # 检索的产物，包括完整的 chunk && score
    chunk: Chunk
    score: float
    retriever_type: str = "dense"   # 来自哪个检索器


@dataclass(frozen=True)
class Citation:            # 展示阶段的产物，精简的引用信息和截断的 snippet 
    source_path: str
    chunk_id: str           # 这个 chunk 的唯一身份标识         定位这是哪一个 chunk                      适合去重、引用、定位、存储
    chunk_index: int        # 这个 chunk 在原文档中的顺序编号    确定这是 source 文件中的第几个 chunk       适合展示和排序              chunk_index 只在同一个 source_path 中才有意义
    score: float
    snippet: str            # chunk 的前 n 个字符


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    contexts: list[str]
    trace: dict[str, Any] = field(default_factory=dict) # trace 是项目的 黑匣子，记录了问答的完整路径


@dataclass(frozen=True)
class EvalExample:
    question: str
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    question: str
    answer: str
    ground_truth: str | None
    contexts: list[str]
    citations: list[Citation]
    trace: dict[str, Any] = field(default_factory=dict)
