from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .extrapolation import ExtrapolationResult


def plot_extrapolations(
    gammas: Sequence[float],
    predictions: Mapping[float, Sequence[float]],
    fits: Mapping[float, ExtrapolationResult],
    output_path: Path,
) -> None:
    inverse_gamma = 1.0 / np.asarray(gammas, dtype=float)
    grid = np.linspace(0.0, inverse_gamma.max() * 1.05, 200)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for terminal_time, times in sorted(predictions.items()):
        values = np.asarray(times, dtype=float)
        axis.scatter(inverse_gamma, values, s=35, label=f"T={terminal_time:g}")
        tail = min(3, len(values))
        coefficients = np.polyfit(inverse_gamma[-tail:], values[-tail:], deg=1)
        axis.plot(grid, np.polyval(coefficients, grid), linewidth=1.4)
        axis.scatter([0.0], [fits[terminal_time].estimate], marker="x", s=55)
    axis.set_xlabel(r"$1/\gamma$")
    axis.set_ylabel(r"Detected time $t_\gamma$")
    axis.set_title("ATR-BPINN threshold extrapolation")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)

