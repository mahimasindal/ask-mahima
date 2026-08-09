"""Vector store abstraction and retrieval.

`VectorStore` is an interface any vector database can implement. `Retriever`
is the only thing the rest of the app (specifically app/services/rag.py) is
allowed to depend on — it converts raw vector-store hits into typed
`RetrievedChunk` models, so no `chromadb`-specific type ever leaks upward.

To swap ChromaDB for FAISS/Pinecone/etc. later, write a new VectorStore
subclass and change one line where Retriever is constructed. Retriever and
everything above it (rag.py, the API layer) stays untouched.
"""

from abc import ABC, abstractmethod
from typing import Any

import chromadb

from app.models.retrieval import RetrievedChunk
from app.services.config import Settings
from app.services.embeddings import get_embedding_function


class RetrievalError(Exception):
    """Raised when the vector store can't be queried (e.g. not yet ingested)."""


class VectorStore(ABC):
    """Minimal interface a vector database backend must implement."""

    @abstractmethod
    def add(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None: ...

    @abstractmethod
    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Returns raw hits, each with at least: text, metadata, distance."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored data (used by ingestion for a full rebuild)."""
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed implementation, persisted locally to disk."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._embedding_function = get_embedding_function(settings)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_function,
        )

    def add(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        try:
            results = self._collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as exc:  # noqa: BLE001 — surface as our own error type
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        hits: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            hits.append({"text": text, "metadata": metadata, "distance": distance})
        return hits

    def reset(self) -> None:
        self._client.delete_collection(name=self._settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.collection_name,
            embedding_function=self._embedding_function,
        )


class Retriever:
    """Retrieves relevant chunks for a query, as typed RetrievedChunk models.

    This is the sole interface the RAG layer talks to — see the module
    docstring for why that boundary matters.
    """

    def __init__(self, store: VectorStore, top_k: int):
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        hits = self._store.query(query, top_k=self._top_k)

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            metadata = hit["metadata"] or {}
            chunks.append(
                RetrievedChunk(
                    text=hit["text"],
                    source=metadata.get("source", "unknown"),
                    document_type=metadata.get("document_type", "unknown"),
                    chunk_id=metadata.get("chunk_id", ""),
                    distance=hit["distance"],
                )
            )
        return chunks
