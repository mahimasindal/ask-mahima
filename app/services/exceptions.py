"""FastAPI exception handlers — map internal service errors to HTTP responses.

Registered globally in main.py rather than try/except in each route body,
so route functions stay simple and error-to-status mapping lives in one place.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.services.retrieval import RetrievalError
from app.services.llm import LLMServiceError


async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Knowledge base error: {exc}. Have you run scripts/ingest.py?"},
    )


async def llm_error_handler(request: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "LLM service unavailable, please try again."},
    )
