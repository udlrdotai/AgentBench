"""Load ASMR audio tasks from JSON files."""

import json
from dataclasses import dataclass
from pathlib import Path

from .base import Task


@dataclass
class AudioTask(Task):
    """An evaluation task for ASMR audio generation.

    Extends the base Task with audio-specific fields.
    """

    asmr_type: str = ""  # whisper, trigger, ambient, roleplay, music
    expected_duration: float = 15.0  # seconds


def load_audio_tasks(path: str | Path) -> list[AudioTask]:
    """Load audio tasks from a JSON file.

    Args:
        path: Path to the JSON file containing audio task definitions.

    Returns:
        List of AudioTask objects.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        AudioTask(
            id=item["id"],
            domain=item["domain"],
            capability=item["capability"],
            prompt=item["prompt"],
            criteria=item["criteria"],
            asmr_type=item.get("asmr_type", ""),
            expected_duration=item.get("expected_duration", 15.0),
        )
        for item in data["tasks"]
    ]
