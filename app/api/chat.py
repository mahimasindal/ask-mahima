"""POST /chat — the app's single functional endpoint."""

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag import RagService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, rag_service: RagService = Depends(get_rag_service)) -> ChatResponse:
    return rag_service.answer(request.message)
