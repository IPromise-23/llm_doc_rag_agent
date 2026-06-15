"""
负责把文本变成向量， Chunk.text -> list[float] ,后续再交给 Qdrant 存储和检索
"""
from __future__ import annotations 

from abc import ABC, abstractmethod # ABC 是 Abstract Base Class ，抽象基类     这个类不直接创建对象，而是规定子类必须实现哪些方法

# 定义所有 embedding 模型必须长什么样
class EmbeddingProvider(ABC):
    @property        # 调用时就像访问属性一样     provider.vector_size ，不用加 () 即可调用
    @abstractmethod  # 表示子类必须实现它，否则子类不能被正常实例化
    def vector_size(self) -> int:
        raise NotImplementedError   # 占位，表示 基类不负责具体实现，子类必须写

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:   # 输入：多个字符串  输出：多个向量
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

# 用 sentence-transformers 模型生成向量
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None                      # 一开始不马上加载模型
        self._vector_size: int | None = None    # 一开始不知道向量维度，等第一次需要时再计算

    @property
    def model(self):    # 懒加载 + 缓存
        if self._model is None: # 如果模型还没加载，就加载一次
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self.embed_query("vector size probe"))  # 动态探测向量维度
        return self._vector_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # 调用模型生成向量，归一化，不显示进度条
        return [list(map(float, vector)) for vector in vectors] # 把模型返回的向量统一转成 Python 的 list[float]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return list(map(float, vector))
