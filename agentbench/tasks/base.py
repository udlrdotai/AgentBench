"""Task data structures."""

from dataclasses import dataclass


@dataclass
class Task:
    """A single evaluation task."""

    id: str
    domain: str
    capability: str
    prompt: str
    criteria: str
