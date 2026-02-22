"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentbench.tasks.base import Task


@dataclass
class EvalResult:
    """Result of an evaluation."""

    task_id: str
    score: float
    comment: str


class BaseEvaluator(ABC):
    """Base class that all evaluators must implement."""

    @abstractmethod
    def evaluate(self, task: Task, model_output: str) -> EvalResult:
        """Evaluate a model's output for a given task.

        Args:
            task: The task that was evaluated.
            model_output: The model's generated output.

        Returns:
            An EvalResult with score and comment.
        """
