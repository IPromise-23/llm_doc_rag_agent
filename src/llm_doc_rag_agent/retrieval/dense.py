from __future__ import annotations

from llm_doc_rag_agent.embeddings import EmbeddingProvider
from llm_doc_rag_agent.schemas import RetrievedChunk
from llm_doc_rag_agent.vectorstores import QdrantVectorStore


class DenseRetriever:
    def __init__(self, embeddings: EmbeddingProvider, store: QdrantVectorStore) -> None:
        self.embeddings = embeddings
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(query)
        return self.store.search(query_vector=query_vector, top_k=top_k)
