"""CLI entry point for ASMR audio benchmark evaluation."""

import argparse

from agentbench.audio_runner import run_audio_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="AgentBench - ASMR Audio Evaluation (Phase 1)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai-tts,gemini-tts,minimax-music",
        help="Comma-separated audio model names (e.g. openai-tts,gemini-tts,minimax-music)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="benchmarks/asmr_audio.json",
        help="Path to the ASMR audio benchmark JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/audio",
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/audio_results.json",
        help="Path for the output JSON results file",
    )

    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    run_audio_benchmark(
        model_names=model_names,
        tasks_path=args.tasks,
        output_dir=args.output_dir,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
