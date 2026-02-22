# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentBench is an AI Agent evaluation platform for "super-individuals" (独立开发者、自媒体创作者等). It benchmarks AI models across four domains: code development, social media, options trading, and personal health. The core question it answers: which model + agent framework combination performs best in real scenarios.

## Common Commands

```bash
# Install (editable/dev mode)
pip install -e .

# Run evaluation (default: OpenAI model via OpenRouter)
python run.py

# Run with multiple models
python run.py --models openai,anthropic,google,deepseek

# Full options (judge uses OpenRouter model ID format)
python run.py --models openai,anthropic,google,deepseek --tasks benchmarks/text_generation.json --output results/output.json --judge openai/gpt-5.2
```

There is no test suite or linter configured yet.

## Architecture

The pipeline flows: **load tasks → initialize models → generate outputs → judge with LLM → report results**.

- `run.py` — CLI entry point, parses args and calls `runner.run_benchmark()`
- `agentbench/runner.py` — Orchestrates the full pipeline: load → generate → evaluate → report
- `agentbench/config.py` — `MODEL_REGISTRY` dict maps short names ("openai", "anthropic", "google", "deepseek") to `OpenRouterModel`; `get_model()` factory instantiates them with `OPENROUTER_API_KEY` from `.env`
- `agentbench/models/base.py` — `BaseModel` ABC with `generate(prompt) -> str` and `name` property
- `agentbench/models/openrouter_model.py` — OpenRouter adapter; routes all models through OpenRouter's OpenAI-compatible API
- `agentbench/models/openai_model.py` — Legacy OpenAI direct adapter (kept for reference)
- `agentbench/models/anthropic_model.py` — Legacy Anthropic direct adapter (kept for reference)
- `agentbench/tasks/base.py` — `Task` dataclass (id, domain, capability, prompt, criteria)
- `agentbench/tasks/loader.py` — Loads tasks from JSON files
- `agentbench/evaluators/llm_judge.py` — LLM-as-Judge via OpenRouter; scores 1-10, temperature=0.0
- `agentbench/results/reporter.py` — Terminal table output + JSON export with timestamp

## Adding a New Model

1. Create `agentbench/models/yourmodel.py` extending `BaseModel`
2. Register it in `MODEL_REGISTRY` in `agentbench/config.py`
3. Handle API key retrieval in `get_model()` if needed

## Task Format

Benchmark tasks are in `benchmarks/*.json`:
```json
{
  "tasks": [
    { "id": "domain-task-N", "domain": "...", "capability": "...", "prompt": "...", "criteria": "..." }
  ]
}
```

## Key Conventions

- Python 3.10+ with type hints (`|` union syntax, dataclasses)
- API keys managed via `.env` file (loaded by `python-dotenv`); uses `OPENROUTER_API_KEY`; never commit `.env`
- Git commit messages use conventional prefixes: `feat:`, `fix:`, `docs:`
- Automated workflow: GitHub Issues with `run-on:<runner>` label trigger Claude agent via `.github/workflows/agent-issue-runner.yml`, which creates a branch `agent/issue-N`, commits, pushes, and opens a PR
