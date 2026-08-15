"""RAG orchestration: retrieve context, build a prompt, ask the LLM.

This is the only module that imports both `Retriever` and `LLMClient` — it
owns the prompt template and the decision of how retrieval results become
LLM input and how the LLM's answer becomes an API response. Retrieval and
LLM logic stay independent of each other; this is where they're composed.
"""

from app.models.chat import ChatResponse
from app.models.retrieval import RetrievedChunk
from app.services.llm import LLMClient
from app.services.reranker import Reranker
from app.services.retrieval import Retriever

SYSTEM_PROMPT = (
    "You are Mahima's personal AI. Answer questions using only the "
    "information in the supplied context. You may combine, summarize, and "
    "reason across multiple pieces of context to form a complete answer — "
    "for example, describing her strengths or making a case for hiring her "
    "based on her stated skills and experience. That is synthesis, not "
    "fabrication, as long as every claim you make is grounded in the "
    "context. Do not invent specific facts, numbers, dates, employers, or "
    "claims that are not stated or clearly implied by the context.\n\n"
    "If a question is broad or a little ambiguous (e.g. 'what's your "
    "favourite?' without saying favourite what), don't refuse just because "
    "it isn't phrased precisely — answer with whatever the context most "
    "directly speaks to, and briefly note what you're basing it on. Only "
    "say you don't know if the context genuinely has nothing relevant to "
    "offer.\n\n"
    "Voice: talk the way Mahima actually talks — warm, witty, a little "
    "irreverent, and genuinely into the things she's into. She describes "
    "herself as funny, opinionated but respectful, someone who hates small "
    "talk and lights up over deep conversations, and she writes with real "
    "personality (self-deprecating asides, the occasional 'lol', dry humor) "
    "rather than corporate polish. Let that come through: when a topic is "
    "clearly something she cares about (her work, her writing, her "
    "opinions), answer with actual enthusiasm, not a flat summary — like a "
    "friend of hers explaining why it's cool, not a résumé being read aloud. "
    "Skip stiff, listy phrasing ('She has experience in X, Y, and Z') in "
    "favor of how a person would actually say it out loud. Still keep it "
    "tight — passionate doesn't mean rambling, and don't force jokes where "
    "they don't fit (e.g. someone asking for her address plainly doesn't "
    "know it, no bit needed)."
)

NO_CONTEXT_ANSWER = (
    "I don't have any information about that in my knowledge base yet."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a labeled context block the LLM can cite."""
    blocks = [f"[Source: {chunk.source}]\n{chunk.text}" for chunk in chunks]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Deduped source filenames, preserving retrieval order (most relevant first)."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.source not in seen:
            seen.add(chunk.source)
            sources.append(chunk.source)
    return sources


class RagService:
    def __init__(self, retriever: Retriever, llm_client: LLMClient, reranker: Reranker, top_k: int):
        self._retriever = retriever
        self._llm_client = llm_client
        self._reranker = reranker
        self._top_k = top_k

    def answer(self, question: str) -> ChatResponse:
        candidates = self._retriever.retrieve(question)

        # No chunks at all (e.g. empty/unpopulated knowledge base) — short-circuit
        # without calling the LLM. This is deterministic and free, and distinct
        # from the prompt-level "say I don't know" instruction, which handles
        # the case where chunks WERE retrieved but don't answer the question.
        if not candidates:
            return ChatResponse(answer=NO_CONTEXT_ANSWER, sources=[])

        # Reranker narrows the (wider) candidate pool down to top_k using
        # actual relevance judgment rather than raw embedding distance —
        # see app/services/reranker.py for why. NoOpReranker just truncates,
        # so this line behaves the same as before when reranking is off.
        chunks = self._reranker.rerank(question, candidates, top_k=self._top_k)

        context = format_context(chunks)
        user_prompt = build_user_prompt(question, context)
        answer_text = self._llm_client.generate(SYSTEM_PROMPT, user_prompt)

        return ChatResponse(answer=answer_text, sources=_dedupe_sources(chunks))
