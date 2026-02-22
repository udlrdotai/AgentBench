"""OpenAI model adapter."""

import openai

from .base import BaseModel


class OpenAIModel(BaseModel):
    """Adapter for OpenAI chat completion models."""

    def __init__(self, model_id: str = "gpt-5.2", api_key: str | None = None):
        super().__init__(model_id)
        self._client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    @property
    def name(self) -> str:
        return f"openai/{self.model_id}"
