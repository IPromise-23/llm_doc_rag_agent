from __future__ import annotations

from pathlib import Path 
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from llm_doc_rag_agent.schemas import Chunk, RetrievedChunk


class QdrantVectorStore:
    """Thin Qdrant wrapper for local persistent document indexes."""

    def __init__(self, path: str | Path, collection: str, vector_size: int | None = None) -> None:
        self.path = Path(path).expanduser()             # path 本地 Qdrant 数据目录
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection = collection                    # collection 集合名
        self.vector_size = vector_size
        self.client = QdrantClient(path=str(self.path)) # 本地模式 Qdrant client ，把数据存在 path 指向的目录中

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:    # 确保 collection 已经存在
        exists = self.collection_exists()
        if exists and not recreate:     # 如果已经存在就不强制重建，直接返回
            return
        if exists and recreate:         # 如果存在且要求重建，就先删掉 collection 再重建————用于重新索引
            self.client.delete_collection(self.collection)
        if self.vector_size is None:
            raise RuntimeError(
                f"Collection '{self.collection}' does not exist and vector_size is unknown. "
                "Run ingest first or initialize the vector store with an embedding dimension."
            )
        self.client.create_collection(  # 建立 Qdrant 数据库 
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 64) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection()    # 写入数据表前需要先确保 collection 存在
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.id)), # uuid5() 同样的 chunk.id 永远生成同样的 UUID
                    vector=vector,
                    payload=self._chunk_to_payload(chunk),
                )   # chunk --> Qdrant point
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)  # strict 表示两个列表长度不一致就报错
            ]
            self.client.upsert(collection_name=self.collection, points=points)  # upsert = update or insert
            total += len(points)
        return total

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.ensure_collection()
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,     # 用问题向量去搜索 
            limit=top_k,
            with_payload=True,      # 返回结果时带上 原始文本 和 metadata
        )
        return [self._point_to_retrieved(point) for point in response.points]   # point --> RetrievedChunk

    def list_sources(self, limit: int | None = 200) -> list[str]:   # 列出有哪些源文件入库了，返回源文件路径列表    默认最多读 200 个
        self.ensure_collection()                                
        points = self._scroll_all(limit=limit)  # 列 source 通常只是作概览，不用扫完这个库
        sources = sorted({str(point.payload.get("source_path", "")) for point in points if point.payload})  # 只处理有 payload 的 point
        return [source for source in sources if source] # sources 是去重后的路径排序，里面存在 "" & source_path ，一般小于 points 数量，因为一篇文档可以被切分成多个 chunk

    def chunks_for_source(self, source_path: str, limit: int | None = 100) -> list[Chunk]:  # 按某个源文件路径取回它的 chunks
        self.ensure_collection()
        points = self._scroll_all(limit=limit, scroll_filter=self._source_filter(source_path))
        chunks = [self._payload_to_chunk(point.payload or {}) for point in points]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)  # 按照原文顺序排序

    def list_chunks(self, limit: int | None = None) -> list[Chunk]: # 列出当前 collection 中的所有 Chunk
        self.ensure_collection()
        points = self._scroll_all(limit=limit)
        chunks = [self._payload_to_chunk(point.payload or {}) for point in points]
        return sorted(chunks, key=lambda chunk: (chunk.source_path, chunk.chunk_index)) # 按照两个字段排序，先按照源文件路径排序；同一个源文件内用 chunk 在原文里的顺序排序

    def source_content_hashes(self, source_path: str) -> set[str]:  # 查某个 source 当前在库中的 document_hash
        if not self.collection_exists():
            return set()
        self.ensure_collection()
        points = self._scroll_all(limit=None, scroll_filter=self._source_filter(source_path))
        hashes = {
            str((point.payload or {}).get("metadata", {}).get("document_hash", ""))
            for point in points
            if point.payload    # 只处理有 payload 的 point
        }
        return {value for value in hashes if value}

    def delete_source(self, source_path: str) -> int:
        if not self.collection_exists():
            return 0
        self.ensure_collection()
        deleted = self.count_source(source_path)
        if deleted == 0:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source_path)),
        )
        return deleted

    def count_source(self, source_path: str) -> int:
        if not self.collection_exists():
            return 0
        self.ensure_collection()
        result = self.client.count(
            collection_name=self.collection,
            count_filter=self._source_filter(source_path),
            exact=True,
        )
        return int(result.count)    # .count 是 Qdrant 返回对象里的数量字段

    def inspect_collection(self) -> dict[str, Any]: # 查看 collection 的概览信息，诊断/查看状态，返回 dict ，展示当前 collection 的基本情况
        self.ensure_collection()
        info = self.client.get_collection(collection_name=self.collection)  # 获取 collection 的元信息
        total_points = int(self.client.count(collection_name=self.collection, exact=True).count)    # exact 表示精确计数
        sources = self.list_sources(limit=None) # 列出所有 source
        return {
            "collection": self.collection,                                          # 当前 collection 名
            "qdrant_path": str(self.path),                                          # 本地 Qdrant 存储目录
            "points": total_points,                                                 # 当前库中有多少个 points
            "sources": len(sources),                                                # 入库了多少个不同的源文件
            "source_paths": sources,                                                # 具体有哪些源文件路径
            "status": str(getattr(info, "status", "")),                             # collection 状态，info 有 status 就返回
            "vectors_count": getattr(info, "vectors_count", None),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        }

    def close(self) -> None:
        self.client.close()

    # 把符合条件的数据分页拿出来
    def _scroll_all(    # scroll 分页，从 Qdrant 分页读取 points
        self,
        limit: int | None,                              # 总共读多少条
        scroll_filter: models.Filter | None = None,     # Qdrant 的过滤条件
        page_size: int = 256,                           # 每一页最多读多少条，单次请求 Qdrant 的批大小
    ) -> list[Any]:
        points: list[Any] = []  # 存储结果
        offset: Any = None      # 下一页游标    从头开始读，Qdrant 每返回一批数据后会同步给出一个 offset
        remaining = limit       # 最多还要读多少条
        while True: # 无限循环
            batch_limit = page_size if remaining is None else min(page_size, remaining)
            if batch_limit <= 0:
                break
            batch, offset = self.client.scroll( # 从 Qdtrant 读取数据后返回 当前这一页读到的 points && 下一页游标
                collection_name=self.collection,
                scroll_filter=scroll_filter,    # 过滤条件
                limit=batch_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)    # batch = [point_1,...,point_batch]
            if offset is None or not batch: # 没有下一页 or 这一页什么都没有读到
                break
            if remaining is not None:
                remaining -= len(batch)
        return points
    
    # 指定目标文件
    def _source_filter(self, source_path: str) -> models.Filter:    # 筛选哪些 points ，返回一个 Qdrant filter
        return models.Filter(
            must=[  # 必须满足这些条件
                models.FieldCondition(
                    key="source_path",  # 表示看 payload 中的 source_path 字段
                    match=models.MatchValue(value=source_path), # 字段值必须等于传入的 source_path
                )                                               # 即 payload["source_path"] == source_path
            ]
        )

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "text": chunk.text,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "metadata": chunk.metadata,
        }

    def _payload_to_chunk(self, payload: dict[str, Any]) -> Chunk:
        return Chunk(
            id=str(payload["id"]),
            text=str(payload["text"]),
            source_path=str(payload["source_path"]),
            chunk_index=int(payload["chunk_index"]),
            content_hash=str(payload["content_hash"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _point_to_retrieved(self, point: Any) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            chunk=self._payload_to_chunk(payload),
            score=float(point.score or 0.0),
            retriever_type="dense",
        )
