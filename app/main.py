"""FastAPI application entrypoint — wires routes, error handlers, and the
static frontend together.

Run: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import chat, health
from app.services.exceptions import llm_error_handler, retrieval_error_handler
from app.services.llm import LLMServiceError
from app.services.retrieval import RetrievalError

app = FastAPI(title="Ask Mahima")

# Routers must be included BEFORE the static-files catch-all mount below,
# so /chat and /health resolve to their handlers instead of falling through
# to the static file server.
app.include_router(chat.router)
app.include_router(health.router)

app.add_exception_handler(RetrievalError, retrieval_error_handler)
app.add_exception_handler(LLMServiceError, llm_error_handler)

# Serves static/index.html at "/" and any other files under static/.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
