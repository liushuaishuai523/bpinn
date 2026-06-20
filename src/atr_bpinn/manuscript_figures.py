from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .example_references import solve_example_reference
from .fixed_bpinn import FixedBPINN, FixedTrainingConfig, predict_fixed_bpinn
from .problems import PROBLEMS
from .training import device_from_name


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )


def _load_model(checkpoint: Path, key: str, device: torch.device) -> FixedBPINN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    allowed = {item.name for item in fields(FixedTrainingConfig)}
    config_data = {name: value for name, value in payload["config"].items() if name in allowed}
    config = FixedTrainingConfig(**config_data)
    model = FixedBPINN(PROBLEMS[key], config).to(device=device, dtype=torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def make_heatmap_figure(
    key: str,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    problem = PROBLEMS[key]
    model = _load_model(checkpoint_dir / f"{key}_model.pt", key, device)
    reference = solve_example_reference(problem, x_points=181, t_points=181)
    prediction = predict_fixed_bpinn(model, reference.x, reference.t, device)
    relative = np.abs(prediction - reference.u) / np.maximum(np.abs(reference.u), 1.0e-6)
    log_relative = np.log10(np.maximum(relative, 1.0e-6))
    extent = [reference.x[0], reference.x[-1], reference.t[0], reference.t[-1]]

    figure, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True)
    panels = (
        (reference.u, "Reference", "viridis", None, None),
        (prediction, "BPINN", "viridis", None, None),
        (log_relative, r"$\log_{10}$ relative error", "magma", -6.0, 0.0),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            rasterized=True,
        )
        axis.set_title(title, pad=4)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$t$")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.025)
        colorbar.ax.tick_params(labelsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{key}_bpinn_heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)

    single_panels = (
        (prediction, "BPINN", "viridis", None, None, "solution"),
        (log_relative, r"$\log_{10}$ relative error", "magma", -6.0, 0.0, "error"),
    )
    for values, title, cmap, lower, upper, suffix in single_panels:
        single, axis = plt.subplots(figsize=(2.65, 2.55), constrained_layout=True)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            rasterized=True,
        )
        axis.set_title(title, fontsize=15, pad=4)
        axis.set_xlabel(r"$x$", fontsize=14)
        axis.set_ylabel(r"$t$", fontsize=14)
        axis.tick_params(labelsize=12)
        colorbar = single.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
        colorbar.ax.tick_params(labelsize=11)
        single.savefig(
            output_dir / f"{key}_bpinn_{suffix}.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close(single)


def make_atr_figure(result_dir: Path, output_dir: Path) -> None:
    with (result_dir / "threshold_predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    grouped: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(float(row["terminal_time"]), []).append(
            (float(row["gamma"]), float(row["predicted_time"]))
        )

    figure, axis = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for color, (terminal_time, values) in zip(colors, sorted(grouped.items())):
        values = sorted(values)
        gamma = np.asarray([item[0] for item in values])
        detected = np.asarray([item[1] for item in values])
        inverse = 1.0 / gamma
        tail = 3
        coefficients = np.polyfit(inverse[-tail:], detected[-tail:], deg=1)
        grid = np.linspace(0.0, inverse.max() * 1.04, 200)
        axis.scatter(inverse, detected, color=color, s=34, zorder=3, label=rf"$T={terminal_time:g}$")
        axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=1.8)
        estimate = summary["terminal_fits"][str(terminal_time)]["estimate"]
        axis.scatter([0.0], [estimate], color=color, marker="x", s=55, linewidths=1.8)
    axis.set_xlabel(r"$1/\gamma$")
    axis.set_ylabel(r"$t_\gamma$")
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, loc="lower left")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "atr_threshold_extrapolation.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript-ready BPINN PDF figures.")
    parser.add_argument("--results", type=Path, default=Path("results/bpinn"))
    parser.add_argument("--atr-results", type=Path, default=Path("results/quick"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    configure_matplotlib()
    device = device_from_name(args.device, "float64")
    for key in PROBLEMS:
        make_heatmap_figure(key, args.results, args.output, device)
    make_atr_figure(args.atr_results, args.output)


if __name__ == "__main__":
    main()
