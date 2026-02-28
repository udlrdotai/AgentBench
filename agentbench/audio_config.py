"""Configuration for ASMR audio benchmark — model registry and environment."""

import os

from dotenv import load_dotenv

from agentbench.models.audio.base import AudioModel
from agentbench.models.audio.gemini_tts_model import GeminiTTSModel
from agentbench.models.audio.minimax_music_model import MiniMaxMusicModel
from agentbench.models.audio.openai_tts_model import OpenAITTSModel

load_dotenv()

# Audio model registry: short name -> factory class
AUDIO_MODEL_REGISTRY: dict[str, type[AudioModel]] = {
    "openai-tts": OpenAITTSModel,
    "gemini-tts": GeminiTTSModel,
    "minimax-music": MiniMaxMusicModel,
}


def get_audio_model(name: str) -> AudioModel:
    """Create an audio model instance by short name.

    Each model uses its own API key from environment variables:
    - openai-tts: OPENAI_API_KEY
    - gemini-tts: GOOGLE_API_KEY
    - minimax-music: MINIMAX_API_KEY (+ MINIMAX_GROUP_ID)

    Args:
        name: One of 'openai-tts', 'gemini-tts', 'minimax-music'.

    Returns:
        A configured audio model instance.
    """
    name = name.strip().lower()
    if name not in AUDIO_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown audio model: {name!r}. Available: {list(AUDIO_MODEL_REGISTRY)}"
        )

    if name == "openai-tts":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAITTSModel(api_key=api_key)

    elif name == "gemini-tts":
        api_key = os.getenv("GOOGLE_API_KEY")
        return GeminiTTSModel(api_key=api_key)

    elif name == "minimax-music":
        api_key = os.getenv("MINIMAX_API_KEY")
        group_id = os.getenv("MINIMAX_GROUP_ID")
        return MiniMaxMusicModel(api_key=api_key, group_id=group_id)

    raise ValueError(f"Unknown audio model: {name!r}")
