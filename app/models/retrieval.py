"""Internal models for retrieval results (not part of the public API contract)."""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A single chunk returned from the vector store, with its provenance."""

    text: str
    source: str  # original filename, e.g. "career.md"
    document_type: str  # filename stem, e.g. "career"
    chunk_id: str
    distance: float  # lower = more similar (Chroma's default distance metric)
