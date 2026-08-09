"""Pydantic models for the /chat API contract."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]  # deduped source filenames, e.g. ["career.md", "projects.md"]
