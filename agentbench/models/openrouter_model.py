"""OpenRouter model adapter — routes requests through OpenRouter's unified API."""

import openai

from .base import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(BaseModel):
    """Adapter that accesses any model via OpenRouter's OpenAI-compatible API."""

    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id)
        self._client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    @property
    def name(self) -> str:
        return f"openrouter/{self.model_id}"
