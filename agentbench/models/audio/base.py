"""Abstract base class for audio model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioResult:
    """Result of audio generation."""

    file_path: Path
    duration_seconds: float
    sample_rate: int
    format: str  # "wav", "mp3", etc.
    model_name: str = ""
    task_id: str = ""
    generation_time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class AudioModel(ABC):
    """Base class that all audio model adapters must implement."""

    # Subclasses should override to declare which ASMR types they can handle.
    supported_asmr_types: set[str] = {"whisper", "trigger", "ambient", "roleplay", "music"}

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate_audio(self, prompt: str, output_dir: Path, task_id: str = "", asmr_type: str = "") -> AudioResult:
        """Generate audio from a text prompt.

        Args:
            prompt: The text description for audio generation.
            output_dir: Directory to save the generated audio file.
            task_id: Optional task identifier for file naming.
            asmr_type: ASMR category (whisper, trigger, ambient, roleplay, music).

        Returns:
            AudioResult with file path and metadata.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this audio model."""
