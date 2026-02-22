"""LLM-as-Judge evaluator — routes through OpenRouter."""

import json
import re

import openai

from agentbench.models.openrouter_model import OPENROUTER_BASE_URL
from agentbench.tasks.base import Task

from .base import BaseEvaluator, EvalResult

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. You will be given a task prompt, evaluation criteria, \
and a model's output. Score the output from 1 to 10 based on how well it meets the criteria.

You MUST respond in the following JSON format and nothing else:
{"score": <integer 1-10>, "comment": "<brief evaluation in the same language as the task>"}\
"""

JUDGE_USER_TEMPLATE = """\
## Task Prompt
{prompt}

## Evaluation Criteria
{criteria}

## Model Output
{output}

Please evaluate the output above. Respond with JSON only.\
"""


class LLMJudge(BaseEvaluator):
    """Uses an LLM to judge model outputs."""

    def __init__(self, model_id: str = "openai/gpt-5.2", api_key: str | None = None):
        self.model_id = model_id
        self._client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    def evaluate(self, task: Task, model_output: str) -> EvalResult:
        user_message = JUDGE_USER_TEMPLATE.format(
            prompt=task.prompt,
            criteria=task.criteria,
            output=model_output,
        )

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content or ""
        return self._parse_response(task.id, raw)

    @staticmethod
    def _parse_response(task_id: str, raw: str) -> EvalResult:
        """Parse the JSON response from the judge model."""
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return EvalResult(task_id=task_id, score=0, comment=f"Failed to parse judge response: {raw[:200]}")

        try:
            data = json.loads(json_match.group())
            score = max(1, min(10, int(data["score"])))
            comment = str(data.get("comment", ""))
            return EvalResult(task_id=task_id, score=score, comment=comment)
        except (json.JSONDecodeError, KeyError, ValueError):
            return EvalResult(task_id=task_id, score=0, comment=f"Failed to parse judge response: {raw[:200]}")
