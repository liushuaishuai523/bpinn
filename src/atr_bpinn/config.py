from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    width: int = 50
    hidden_layers: int = 3
    n_interior: int = 256
    n_boundary: int = 64
    adam_steps: int = 1200
    adam_lr: float = 2.0e-3
    lbfgs_steps: int = 300
    lbfgs_lr: float = 0.8
    weight_pde: float = 1.0
    weight_geo: float = 10.0
    weight_navier: float = 1.0
    end_time_sampling_power: float = 2.0
    grad_clip: float = 100.0
    dtype: str = "float64"


@dataclass(frozen=True)
class ExperimentConfig:
    domain_length: float = 5.0
    terminal_times: tuple[float, ...] = (1.2, 1.5, 1.8)
    gammas: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0)
    seeds: tuple[int, ...] = (20260620,)
    max_refinements: int = 18
    refinement_tolerance: float = 5.0e-4
    detection_time_points: int = 1200
    detection_space_points: int = 161
    minimum_positive_g: float = 1.0e-10
    stable_tail_size: int = 3
    stable_tail_tolerance: float = 8.0e-3
    max_terminal_dispersion: float = 1.0e-2
    max_component_loss: float = 2.0e-3
    quadratic_disagreement_tolerance: float = 8.0e-3
    output_dir: Path = field(default_factory=lambda: Path("results"))
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data

    @classmethod
    def quick(cls, output_dir: Path = Path("results/quick")) -> "ExperimentConfig":
        return cls(
            max_refinements=18,
            refinement_tolerance=2.5e-3,
            detection_time_points=500,
            detection_space_points=81,
            stable_tail_tolerance=1.5e-2,
            max_terminal_dispersion=2.0e-2,
            max_component_loss=1.0e-2,
            output_dir=output_dir,
            training=TrainingConfig(
                width=32,
                hidden_layers=3,
                n_interior=160,
                n_boundary=40,
                adam_steps=350,
                lbfgs_steps=100,
            ),
        )
