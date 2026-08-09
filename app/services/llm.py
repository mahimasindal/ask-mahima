"""LLM client — talks to a model via OpenRouter's OpenAI-compatible API.

OpenRouter is not the Anthropic API: it exposes an OpenAI-style
chat-completions endpoint that can route to many providers/models (including
Claude models, by name, if you ever want to switch `llm_model`). We use the
official `openai` SDK pointed at OpenRouter's base_url rather than the
Anthropic SDK, because that's the interface OpenRouter actually implements.

This client knows nothing about retrieval, chunks, or prompt templates — it
takes a system prompt and a user prompt and returns text. That's what keeps
retrieval and LLM logic independent: app/services/rag.py is the only place
that combines them.
"""

from openai import APIError, OpenAI

from app.services.config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMServiceError(Exception):
    """Raised when the LLM call fails (auth, network, rate limit, etc.)."""


class LLMClient:
    def __init__(self, settings: Settings):
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key,
        )
        self._model = settings.llm_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIError as exc:
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMServiceError("LLM returned an empty response")
        return content
