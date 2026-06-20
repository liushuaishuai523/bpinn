from __future__ import annotations

import csv
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .atr import ATRResult, run_atr
from .config import ExperimentConfig
from .extrapolation import combine_terminal_estimates, extrapolate_thresholds
from .plotting import plot_extrapolations
from .problems import FOURTH_ORDER_REFERENCE_TIME
from .training import device_from_name, train_bpinn


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_experiment(
    config: ExperimentConfig,
    device_name: str = "auto",
    progress: Callable[[str], None] = print,
    resume: bool = False,
) -> dict[str, object]:
    device = device_from_name(device_name, config.training.dtype)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_dict())
    progress(f"Using device: {device}")

    atr_path = output_dir / "atr_runs.json"
    atr_results: list[ATRResult] = []
    if resume and atr_path.exists():
        previous_payload = json.loads(atr_path.read_text(encoding="utf-8"))
        atr_results = [ATRResult.from_dict(item) for item in previous_payload]
        atr_results = [item for item in atr_results if item.converged]
        progress(f"Resuming with {len(atr_results)} converged runs")
    completed_keys = {
        (item.terminal_time, item.gamma, item.seed) for item in atr_results
    }
    for terminal_time in config.terminal_times:
        pending_seeds = {
            seed
            for gamma in config.gammas
            for seed in config.seeds
            if (terminal_time, gamma, seed) not in completed_keys
        }
        if not pending_seeds:
            continue
        prepared = {}
        for seed in sorted(pending_seeds):
            progress(f"Preparing shared BPINN for T={terminal_time:g}, seed={seed}")
            prepared[seed] = train_bpinn(
                config, terminal_time, seed + 1009, device, progress=None
            )
        for gamma in config.gammas:
            seed_predictions: list[ATRResult] = []
            for seed in config.seeds:
                if (terminal_time, gamma, seed) in completed_keys:
                    continue
                initial_model, initial_losses = prepared[seed]
                result = run_atr(
                    config,
                    terminal_time,
                    gamma,
                    seed,
                    device,
                    progress=progress,
                    initial_model=copy.deepcopy(initial_model),
                    initial_losses=initial_losses,
                )
                seed_predictions.append(result)
                atr_results.append(result)
            if any(item.predicted_time is None for item in seed_predictions):
                progress(f"Rejected T={terminal_time:g}, gamma={gamma:g}: no detection")

    _write_json(atr_path, [item.to_dict() for item in atr_results])

    grouped: dict[tuple[float, float], list[ATRResult]] = defaultdict(list)
    for result in atr_results:
        grouped[(result.terminal_time, result.gamma)].append(result)

    predictions: dict[float, list[float]] = {}
    run_failures: list[str] = []
    for terminal_time in config.terminal_times:
        predictions[terminal_time] = []
        for gamma in config.gammas:
            runs = grouped[(terminal_time, gamma)]
            if not runs or any(not run.converged or run.predicted_time is None for run in runs):
                run_failures.append(f"T={terminal_time:g},gamma={gamma:g}")
                predictions[terminal_time].append(float("nan"))
            else:
                predictions[terminal_time].append(
                    float(np.mean([run.predicted_time for run in runs if run.predicted_time is not None]))
                )

    fits = {}
    if not run_failures:
        for terminal_time, values in predictions.items():
            fits[terminal_time] = extrapolate_thresholds(
                terminal_time, config.gammas, values, config
            )
        combined = combine_terminal_estimates(list(fits.values()), config)
        plot_extrapolations(
            config.gammas, predictions, fits, output_dir / "threshold_extrapolation.png"
        )
    else:
        combined = None

    rows = []
    for terminal_time, values in predictions.items():
        for gamma, predicted in zip(config.gammas, values):
            rows.append({"terminal_time": terminal_time, "gamma": gamma, "predicted_time": predicted})
    with (output_dir / "threshold_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["terminal_time", "gamma", "predicted_time"])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "method": "fixed-threshold ATR-BPINN with 1/gamma extrapolation",
        "prediction_used_reference_time": False,
        "device": str(device),
        "run_failures": run_failures,
        "terminal_fits": {str(key): value.to_dict() for key, value in fits.items()},
        "combined": combined.to_dict() if combined else None,
        "reference_time": FOURTH_ORDER_REFERENCE_TIME,
        "absolute_error": (
            abs(combined.estimate - FOURTH_ORDER_REFERENCE_TIME) if combined else None
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    progress(json.dumps(summary, indent=2))
    return summary
