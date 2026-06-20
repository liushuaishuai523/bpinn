from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .example_references import solve_example_reference
from .fixed_bpinn import FixedTrainingConfig, predict_fixed_bpinn, train_fixed_bpinn
from .problems import FOURTH_ORDER_REFERENCE_TIME, PROBLEMS, ProblemSpec
from .training import device_from_name


@dataclass(frozen=True)
class ExampleMetrics:
    problem: str
    global_relative_error: float
    terminal_relative_error: float
    paper_global_error: float
    paper_terminal_error: float
    no_worse_global: bool
    no_worse_terminal: bool
    losses: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def relative_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))


def training_config_for(problem: ProblemSpec, profile: str) -> FixedTrainingConfig:
    if profile == "quick":
        return FixedTrainingConfig.quick()
    common = {
        "width": 64,
        "hidden_layers": 4,
        "adam_steps": 2500,
        "lbfgs_steps": 1000,
    }
    if problem.key == "cubic":
        return FixedTrainingConfig(**common, interior_points_scale=0.5)
    if problem.key == "fourth":
        return FixedTrainingConfig(**common, interior_points_scale=2.0, navier_weight=10.0)
    if problem.key == "convection":
        return FixedTrainingConfig(
            width=80,
            hidden_layers=5,
            adam_steps=4000,
            lbfgs_steps=2000,
            interior_points_scale=2.0,
        )
    return FixedTrainingConfig(**common, interior_points_scale=2.0)


def _plot_example(
    problem: ProblemSpec,
    x: np.ndarray,
    t: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    history: tuple[float, ...],
    output: Path,
) -> None:
    extent = [x[0], x[-1], t[0], t[-1]]
    error = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1.0e-8)
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.7), constrained_layout=True)
    for axis, values, title in zip(
        axes[:3], (reference, prediction, error), ("Reference", "BPINN", "Relative error")
    ):
        image = axis.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis, shrink=0.82)
    axes[3].semilogy(np.arange(len(history)), history)
    axes[3].set_title("Training loss")
    axes[3].set_xlabel("record")
    axes[3].set_ylabel("loss")
    axes[3].grid(alpha=0.25)
    figure.suptitle(problem.title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_fixed_examples(
    problem_keys: Iterable[str],
    profile: str,
    output_dir: Path,
    device_name: str,
    seed: int = 20260620,
) -> list[ExampleMetrics]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[ExampleMetrics] = []

    for key in problem_keys:
        problem = PROBLEMS[key]
        config = training_config_for(problem, profile)
        device = device_from_name(device_name, config.dtype)
        print(f"Training BPINN for {key} on {device}")
        model, training = train_fixed_bpinn(problem, config, seed, device, progress=print)
        reference = solve_example_reference(problem)
        prediction = predict_fixed_bpinn(model, reference.x, reference.t, device)
        global_error = relative_error(prediction, reference.u)
        terminal_error = relative_error(prediction[-1], reference.u[-1])
        item = ExampleMetrics(
            problem=key,
            global_relative_error=global_error,
            terminal_relative_error=terminal_error,
            paper_global_error=problem.paper_global_error,
            paper_terminal_error=problem.paper_terminal_error,
            no_worse_global=global_error <= problem.paper_global_error,
            no_worse_terminal=terminal_error <= problem.paper_terminal_error,
            losses=training.losses.to_dict(),
        )
        metrics.append(item)
        _plot_example(
            problem,
            reference.x,
            reference.t,
            reference.u,
            prediction,
            training.history,
            output_dir / f"{key}_bpinn.png",
        )
        torch.save(
            {"problem": problem.to_dict(), "config": asdict(config), "state_dict": model.state_dict()},
            output_dir / f"{key}_model.pt",
        )
        print(json.dumps(item.to_dict(), indent=2))

    summary = {
        "profile": profile,
        "seed": seed,
        "g_strategy": "strict_positive_decay_with_log_coupling",
        "fourth_order_reference_time": FOURTH_ORDER_REFERENCE_TIME,
        "metrics": [item.to_dict() for item in metrics],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "problem",
                "global_relative_error",
                "terminal_relative_error",
                "paper_global_error",
                "paper_terminal_error",
                "no_worse_global",
                "no_worse_terminal",
            ),
        )
        writer.writeheader()
        for item in metrics:
            row = item.to_dict()
            row.pop("losses")
            writer.writerow(row)
    return metrics
