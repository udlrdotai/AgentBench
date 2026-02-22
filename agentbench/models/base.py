"""Abstract base class for model adapters."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class that all model adapters must implement."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt to send to the model.

        Returns:
            The model's text response.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this model adapter."""
