"""Anthropic model adapter."""

import anthropic

from .base import BaseModel


class AnthropicModel(BaseModel):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        super().__init__(model_id)
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @property
    def name(self) -> str:
        return f"anthropic/{self.model_id}"
