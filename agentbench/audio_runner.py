"""Audio evaluation runner — orchestrates the ASMR audio benchmark pipeline.

Pipeline flow:
  load audio tasks → initialize audio models → generate audio →
  evaluate with objective metrics → print comparison table → save results
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentbench.audio_config import get_audio_model
from agentbench.evaluators.audio_evaluator import AudioEvalResult, AudioEvaluator
from agentbench.models.audio.base import AudioModel
from agentbench.tasks.audio_loader import AudioTask, load_audio_tasks


def run_audio_benchmark(
    model_names: list[str],
    tasks_path: str = "benchmarks/asmr_audio.json",
    output_dir: str = "results/audio",
    output_json: str = "results/audio_results.json",
) -> dict[str, list[AudioEvalResult]]:
    """Run the ASMR audio evaluation pipeline.

    Args:
        model_names: List of audio model short names (e.g. ['openai-tts', 'gemini-tts']).
        tasks_path: Path to the ASMR audio benchmark JSON file.
        output_dir: Directory to save generated audio files.
        output_json: Path for the output JSON results file.

    Returns:
        Mapping of model name -> list of AudioEvalResult.
    """
    # Load tasks
    tasks = load_audio_tasks(tasks_path)
    print(f"Loaded {len(tasks)} ASMR audio tasks from {tasks_path}")

    # Initialize evaluator
    evaluator = AudioEvaluator()

    # Initialize models
    models: list[AudioModel] = []
    for name in model_names:
        try:
            models.append(get_audio_model(name))
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    if not models:
        print("Error: No valid audio models specified.", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    all_results: dict[str, list[AudioEvalResult]] = {}
    output_base = Path(output_dir)

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model.name}")
        print(f"{'='*60}")
        model_results: list[AudioEvalResult] = []

        model_output_dir = output_base / model.name.replace("/", "_")

        for task in tasks:
            print(f"\n  Task {task.id} ({task.asmr_type}): generating audio...", end=" ", flush=True)

            try:
                audio_result = model.generate_audio(
                    prompt=task.prompt,
                    output_dir=model_output_dir,
                    task_id=task.id,
                )
                print(
                    f"OK ({audio_result.duration_seconds:.1f}s, "
                    f"{audio_result.generation_time_seconds:.1f}s gen time)"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                # Create a failed result
                model_results.append(
                    AudioEvalResult(
                        task_id=task.id,
                        model_name=model.name,
                        metrics=None,  # type: ignore[arg-type]
                        technical_score=0.0,
                        comment=f"Audio generation failed: {e}",
                    )
                )
                continue

            # Evaluate the generated audio
            print(f"    Evaluating...", end=" ", flush=True)
            try:
                if audio_result.format != "wav":
                    print(f"SKIP (format={audio_result.format}, need WAV for analysis)")
                    model_results.append(
                        AudioEvalResult(
                            task_id=task.id,
                            model_name=model.name,
                            metrics=None,  # type: ignore[arg-type]
                            technical_score=0.0,
                            comment=f"Cannot analyze {audio_result.format} format; WAV required",
                        )
                    )
                    continue

                eval_result = evaluator.evaluate(
                    audio_path=audio_result.file_path,
                    task_id=task.id,
                    model_name=model.name,
                )
                model_results.append(eval_result)
                print(f"score={eval_result.technical_score}")
            except Exception as e:
                print(f"ERROR: {e}")
                model_results.append(
                    AudioEvalResult(
                        task_id=task.id,
                        model_name=model.name,
                        metrics=None,  # type: ignore[arg-type]
                        technical_score=0.0,
                        comment=f"Evaluation failed: {e}",
                    )
                )

        all_results[model.name] = model_results

    # Report
    _print_comparison_table(all_results, tasks)
    _save_results_json(all_results, output_json)

    return all_results


def _print_comparison_table(
    results: dict[str, list[AudioEvalResult]],
    tasks: list[AudioTask],
) -> None:
    """Print a comparison table of results to the terminal."""
    model_names = list(results.keys())
    if not model_names:
        return

    print(f"\n{'='*80}")
    print("  ASMR Audio Benchmark Results")
    print(f"{'='*80}")

    # Header
    header = f"{'Task ID':<20} {'Type':<10}"
    for name in model_names:
        short_name = name.split("/")[-1][:15]
        header += f" {short_name:>15}"
    print(header)
    print("-" * len(header))

    # Task rows
    task_map = {t.id: t for t in tasks}
    model_totals: dict[str, list[float]] = {name: [] for name in model_names}

    for task in tasks:
        row = f"{task.id:<20} {task.asmr_type:<10}"
        for name in model_names:
            model_results = results.get(name, [])
            task_result = next((r for r in model_results if r.task_id == task.id), None)
            if task_result and task_result.technical_score > 0:
                score = task_result.technical_score
                row += f" {score:>15.1f}"
                model_totals[name].append(score)
            else:
                row += f" {'N/A':>15}"
        print(row)

    # Average row
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<20} {'':10}"
    for name in model_names:
        scores = model_totals[name]
        if scores:
            avg = sum(scores) / len(scores)
            avg_row += f" {avg:>15.1f}"
        else:
            avg_row += f" {'N/A':>15}"
    print(avg_row)
    print()

    # Detailed metrics for each model
    for name in model_names:
        print(f"\n  Detailed Metrics: {name}")
        print(f"  {'-'*60}")
        for result in results.get(name, []):
            if result.metrics:
                print(f"    {result.task_id}: {result.comment}")
            else:
                print(f"    {result.task_id}: {result.comment}")


def _save_results_json(
    results: dict[str, list[AudioEvalResult]], output_path: str
) -> None:
    """Save results to a JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "asmr_audio",
        "models": {},
    }

    for model_name, eval_results in results.items():
        model_data = []
        for r in eval_results:
            entry = {
                "task_id": r.task_id,
                "technical_score": r.technical_score,
                "comment": r.comment,
            }
            if r.metrics:
                entry["metrics"] = {
                    "duration_seconds": r.metrics.duration_seconds,
                    "sample_rate": r.metrics.sample_rate,
                    "snr_db": round(r.metrics.snr_db, 2),
                    "spectral_centroid_hz": round(r.metrics.spectral_centroid_hz, 1),
                    "loudness_lufs": round(r.metrics.loudness_lufs, 2),
                    "peak_dbfs": round(r.metrics.peak_dbfs, 2),
                    "rms_dbfs": round(r.metrics.rms_dbfs, 2),
                    "spectral_rolloff_hz": round(r.metrics.spectral_rolloff_hz, 1),
                    "crest_factor_db": round(r.metrics.crest_factor_db, 2),
                    "low_freq_energy_ratio": round(r.metrics.low_freq_energy_ratio, 4),
                }
            model_data.append(entry)
        serializable["models"][model_name] = model_data

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")
