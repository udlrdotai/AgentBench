"""Configuration management — environment variables and model registry."""

import os

from dotenv import load_dotenv

from agentbench.models.base import BaseModel
from agentbench.models.openrouter_model import OpenRouterModel

load_dotenv()

# Default model IDs used when routing through OpenRouter
OPENROUTER_MODEL_IDS: dict[str, str] = {
    "openai": "openai/gpt-5.2",
    "anthropic": "anthropic/claude-sonnet-4.6",
    "google": "google/gemini-3.1-pro-preview",
    "deepseek": "deepseek/deepseek-v3.2",
}

# Model registry: short name -> factory class
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "openai": OpenRouterModel,
    "anthropic": OpenRouterModel,
    "google": OpenRouterModel,
    "deepseek": OpenRouterModel,
}


def get_model(name: str) -> BaseModel:
    """Create a model instance by short name.

    All models are routed through OpenRouter using a single OPENROUTER_API_KEY.

    Args:
        name: One of 'openai', 'anthropic', 'google', 'deepseek'.

    Returns:
        A configured model instance.
    """
    name = name.strip().lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY)}")

    api_key = os.getenv("OPENROUTER_API_KEY")
    model_id = OPENROUTER_MODEL_IDS[name]
    return OpenRouterModel(model_id=model_id, api_key=api_key)


def get_judge_api_key() -> str | None:
    """Get the API key for the judge model (OpenRouter)."""
    return os.getenv("OPENROUTER_API_KEY")
