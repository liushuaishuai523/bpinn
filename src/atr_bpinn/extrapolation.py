from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True)
class ExtrapolationResult:
    terminal_time: float
    linear_intercept: float
    quadratic_intercept: float
    estimate: float
    tail_spread: float
    fit_rmse: float
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


@dataclass(frozen=True)
class CombinedEstimate:
    estimate: float
    uncertainty: float
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def extrapolate_thresholds(
    terminal_time: float,
    gammas: Sequence[float],
    predicted_times: Sequence[float],
    config: ExperimentConfig,
) -> ExtrapolationResult:
    gamma = np.asarray(gammas, dtype=float)
    times = np.asarray(predicted_times, dtype=float)
    if gamma.ndim != 1 or times.shape != gamma.shape or len(gamma) < 3:
        raise ValueError("gammas and predicted_times must be equal one-dimensional arrays")
    order = np.argsort(gamma)
    gamma = gamma[order]
    times = times[order]
    inverse_gamma = 1.0 / gamma
    tail = config.stable_tail_size

    linear_coefficients = np.polyfit(inverse_gamma[-tail:], times[-tail:], deg=1)
    quadratic_coefficients = np.polyfit(inverse_gamma, times, deg=2)
    linear_intercept = float(linear_coefficients[-1])
    quadratic_intercept = float(quadratic_coefficients[-1])
    fitted = np.polyval(quadratic_coefficients, inverse_gamma)
    fit_rmse = float(np.sqrt(np.mean((times - fitted) ** 2)))
    tail_spread = float(np.max(times[-tail:]) - np.min(times[-tail:]))

    reasons: list[str] = []
    if np.any(np.diff(times) < -config.refinement_tolerance):
        reasons.append("predictions_not_monotone_in_gamma")
    if tail_spread > config.stable_tail_tolerance:
        reasons.append("large_threshold_tail_not_stable")
    if abs(linear_intercept - quadratic_intercept) > config.quadratic_disagreement_tolerance:
        reasons.append("linear_quadratic_extrapolations_disagree")
    if not np.isfinite(linear_intercept + quadratic_intercept):
        reasons.append("non_finite_extrapolation")

    # The asymptotic law is first order in 1/gamma. The quadratic fit is a
    # curvature diagnostic; it is deliberately not used to cherry-pick a value.
    estimate = linear_intercept
    return ExtrapolationResult(
        terminal_time=terminal_time,
        linear_intercept=linear_intercept,
        quadratic_intercept=quadratic_intercept,
        estimate=estimate,
        tail_spread=tail_spread,
        fit_rmse=fit_rmse,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def combine_terminal_estimates(
    estimates: Sequence[ExtrapolationResult], config: ExperimentConfig
) -> CombinedEstimate:
    reasons: list[str] = []
    invalid = [item.terminal_time for item in estimates if not item.valid]
    if invalid:
        reasons.append(f"invalid_terminal_estimates:{invalid}")
    values = np.asarray([item.estimate for item in estimates], dtype=float)
    estimate = float(np.mean(values))
    uncertainty = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if float(np.ptp(values)) > config.max_terminal_dispersion:
        reasons.append("terminal_time_extrapolations_inconsistent")
    return CombinedEstimate(estimate, uncertainty, not reasons, tuple(reasons))

