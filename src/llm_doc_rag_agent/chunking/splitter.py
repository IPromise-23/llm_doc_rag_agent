"""
把 loader 读出来的 Documnet 切成多个 Chunk
embedding 和 向量检索一般都是处理比较短的 Chunk
"""
from __future__ import annotations 

from llm_doc_rag_agent.schemas import Chunk, Document
from llm_doc_rag_agent.utils import short_hash, stable_hash


class SimpleTextSplitter:
    """Deterministic character splitter for technical notes and markdown."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: Document) -> list[Chunk]:    # 将一个 Document 切分成多个 Chunk 
        text = document.text.strip()
        if not text:
            return []
        source_hash = short_hash(document.source_path)  # 根据文件路径生成短 hash ，用来参与 chunk id
        content_hash = stable_hash(text)    # 根据整篇文档内容生成完整的 hash ，用来记录这篇文档的当前内容状态
        pieces = self._split_text(text)     # pieces 还不是 Chunk 对象，现在是 list[str]
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            piece_hash = short_hash(f"{document.source_path}:{index}:{piece}")  # 给当前小片段生成短 hash ，组合了来源路径、块索引、块内容
            metadata = dict(document.metadata)
            metadata.update({"source_hash": source_hash, "document_hash": content_hash})
            chunks.append(
                Chunk(
                    id=f"{source_hash}-{index:05d}-{piece_hash}",
                    text=piece,
                    source_path=document.source_path,
                    chunk_index=index,
                    content_hash=stable_hash(piece),
                    metadata=metadata,
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        # 过滤空段落，并返回区去除段落前后空白的结果    先按照段落切，如果段落太长再从中间切分
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks: list[str] = []
        current = ""    # 当前正在累积的 chunk 文本     作为临时缓冲区
        for paragraph in paragraphs:    # 如果 current 空，候选文本就是当前段落；如果 current 有内容，就把旧内容和新段落拼接起来
            candidate = paragraph if not current else current + "\n\n" + paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue    # 循环结束，进入下一轮循环拿出 paragraph
            if current:     # 候选文本超长了，就先把已有的 current 放进结果中
                chunks.extend(self._split_long_text(current))
            current = paragraph # current 变为当前段落进入下一个循环
        if current: # 处理循环结束后的最后一段 current 未被放入结果中的情况
            chunks.extend(self._split_long_text(current))
        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        step = self.chunk_size - self.chunk_overlap # 滑动窗口步长
        pieces = []
        start = 0
        while start < len(text):    # 起点没有超过文本长度就要继续切分
            piece = text[start : start + self.chunk_size].strip()   # strip() 去掉切出来片段两边的空白
            if piece:
                pieces.append(piece)
            start += step
        return pieces
