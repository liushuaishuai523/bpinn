from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .config import ExperimentConfig
from .model import BPINN
from .training import Detection, LossRecord, detect_threshold, train_bpinn


@dataclass(frozen=True)
class RefinementRecord:
    iteration: int
    input_horizon: float
    effective_threshold: float
    detected_time: float | None
    max_gradient: float
    valid_fraction: float
    losses: LossRecord

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        return data


@dataclass(frozen=True)
class ATRResult:
    terminal_time: float
    gamma: float
    seed: int
    predicted_time: float | None
    converged: bool
    reason: str
    refinements: tuple[RefinementRecord, ...]

    @property
    def final_losses(self) -> LossRecord | None:
        return self.refinements[-1].losses if self.refinements else None

    def to_dict(self) -> dict[str, object]:
        return {
            "terminal_time": self.terminal_time,
            "gamma": self.gamma,
            "seed": self.seed,
            "predicted_time": self.predicted_time,
            "converged": self.converged,
            "reason": self.reason,
            "refinements": [item.to_dict() for item in self.refinements],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ATRResult":
        refinements = []
        for raw in data["refinements"]:
            item = dict(raw)
            item["losses"] = LossRecord(**item["losses"])
            refinements.append(RefinementRecord(**item))
        return cls(
            terminal_time=float(data["terminal_time"]),
            gamma=float(data["gamma"]),
            seed=int(data["seed"]),
            predicted_time=(
                None if data["predicted_time"] is None else float(data["predicted_time"])
            ),
            converged=bool(data["converged"]),
            reason=str(data["reason"]),
            refinements=tuple(refinements),
        )


def run_atr(
    config: ExperimentConfig,
    terminal_time: float,
    gamma: float,
    seed: int,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
    initial_model: BPINN | None = None,
    initial_losses: LossRecord | None = None,
) -> ATRResult:
    horizon = float(terminal_time)
    previous_estimate = horizon
    previous_detection: float | None = None
    lower_horizon = 0.0
    upper_horizon = horizon
    records: list[RefinementRecord] = []
    model = initial_model

    for iteration in range(1, config.max_refinements + 1):
        if progress:
            progress(
                f"T={terminal_time:g}, gamma={gamma:g}, refinement={iteration}, "
                f"horizon={horizon:.6f}"
            )
        if iteration == 1 and model is not None and initial_losses is not None:
            losses = initial_losses
        else:
            model, losses = train_bpinn(
                config,
                horizon,
                seed + 1009 * iteration,
                device,
                model=model,
                progress=None,
            )
        effective_threshold = terminal_time / previous_estimate * gamma
        detection: Detection = detect_threshold(
            model, config, horizon, effective_threshold, device
        )
        records.append(
            RefinementRecord(
                iteration=iteration,
                input_horizon=horizon,
                effective_threshold=effective_threshold,
                detected_time=detection.time,
                max_gradient=detection.max_gradient,
                valid_fraction=detection.valid_fraction,
                losses=losses,
            )
        )
        if detection.time is None:
            if iteration > 1:
                lower_horizon = max(lower_horizon, horizon)
                bracket_estimate = 0.5 * (lower_horizon + upper_horizon)
                losses_converged = losses.max_component() <= config.max_component_loss
                if upper_horizon - lower_horizon < config.refinement_tolerance:
                    return ATRResult(
                        terminal_time,
                        gamma,
                        seed,
                        bracket_estimate,
                        losses_converged,
                        "threshold_bracket_converged"
                        if losses_converged
                        else "loss_not_converged",
                        tuple(records),
                    )
                previous_estimate = bracket_estimate
                horizon = bracket_estimate
                continue
            return ATRResult(
                terminal_time,
                gamma,
                seed,
                None,
                False,
                "threshold_not_reached",
                tuple(records),
            )

        upper_horizon = min(upper_horizon, horizon)
        if (
            previous_detection is not None
            and abs(detection.time - previous_detection) < config.refinement_tolerance
        ):
            losses_converged = losses.max_component() <= config.max_component_loss
            return ATRResult(
                terminal_time,
                gamma,
                seed,
                detection.time,
                losses_converged,
                "refinement_converged" if losses_converged else "loss_not_converged",
                tuple(records),
            )
        previous_detection = detection.time
        previous_estimate = detection.time
        horizon = detection.time

    predicted = records[-1].detected_time if records else None
    if lower_horizon > 0.0 and upper_horizon > lower_horizon:
        predicted = 0.5 * (lower_horizon + upper_horizon)
    converged = False
    return ATRResult(
        terminal_time,
        gamma,
        seed,
        predicted,
        converged,
        "refinement_converged" if converged else "max_refinements_reached",
        tuple(records),
    )
