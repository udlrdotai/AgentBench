"""Load tasks from JSON files."""

import json
from pathlib import Path

from .base import Task


def load_tasks(path: str | Path) -> list[Task]:
    """Load tasks from a JSON file.

    Args:
        path: Path to the JSON file containing task definitions.

    Returns:
        List of Task objects.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        Task(
            id=item["id"],
            domain=item["domain"],
            capability=item["capability"],
            prompt=item["prompt"],
            criteria=item["criteria"],
        )
        for item in data["tasks"]
    ]
