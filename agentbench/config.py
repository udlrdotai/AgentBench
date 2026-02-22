"""Configuration management — environment variables and model registry."""

import os

from dotenv import load_dotenv

from agentbench.models.anthropic_model import AnthropicModel
from agentbench.models.base import BaseModel
from agentbench.models.openai_model import OpenAIModel

load_dotenv()

# Model registry: short name -> factory function
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
}


def get_model(name: str) -> BaseModel:
    """Create a model instance by short name.

    Args:
        name: One of 'openai', 'anthropic'.

    Returns:
        A configured model instance.
    """
    name = name.strip().lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[name]

    if name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return model_cls(api_key=api_key)
    elif name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return model_cls(api_key=api_key)

    return model_cls()


def get_judge_api_key() -> str | None:
    """Get the API key for the judge model (defaults to OpenAI)."""
    return os.getenv("OPENAI_API_KEY")
