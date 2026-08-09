"""GET /health — simple liveness check, useful to confirm the server and
embedding model loaded before wiring up the frontend."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
