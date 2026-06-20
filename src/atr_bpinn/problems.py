from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .model import derivative


FOURTH_ORDER_REFERENCE_TIME = 0.96508


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    title: str
    x_left: float
    x_right: float
    final_time: float
    source_kind: str
    source_strength: float
    spatial_kind: str
    reference_blowup_time: float
    paper_global_error: float
    paper_terminal_error: float
    paper_interior_points: int

    @property
    def domain_length(self) -> float:
        return self.x_right - self.x_left

    @property
    def is_fourth_order(self) -> bool:
        return self.spatial_kind == "negative_biharmonic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fourth_order:
            return torch.zeros_like(x)
        return torch.cos(torch.pi * x / 2.0)

    def source(self, u: torch.Tensor) -> torch.Tensor:
        if self.source_kind == "exponential":
            return torch.exp(torch.clamp(u, min=-20.0, max=50.0))
        if self.source_kind == "shifted_cubic":
            return (u + 1.0) ** 3
        raise ValueError(f"Unknown source: {self.source_kind}")

    def transform(self, u: torch.Tensor) -> torch.Tensor:
        if self.source_kind == "exponential":
            return torch.exp(torch.clamp(-u, min=-50.0, max=20.0))
        if self.source_kind == "shifted_cubic":
            return 0.5 / torch.clamp((u + 1.0) ** 2, min=1.0e-14)
        raise ValueError(f"Unknown source: {self.source_kind}")

    def inverse_transform(self, g: torch.Tensor, floor: float = 1.0e-12) -> torch.Tensor:
        positive_g = torch.clamp(g, min=floor)
        if self.source_kind == "exponential":
            return -torch.log(positive_g)
        if self.source_kind == "shifted_cubic":
            return torch.sqrt(0.5 / positive_g) - 1.0
        raise ValueError(f"Unknown source: {self.source_kind}")

    def spatial_operator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_kind == "laplacian":
            return derivative(u, x, order=2)
        if self.spatial_kind == "convection_diffusion":
            return derivative(u, x, order=2) - derivative(u, x, order=1)
        if self.spatial_kind == "negative_biharmonic":
            return -derivative(u, x, order=4)
        raise ValueError(f"Unknown spatial operator: {self.spatial_kind}")


PROBLEMS: dict[str, ProblemSpec] = {
    "fk": ProblemSpec(
        key="fk",
        title="FK combustion model",
        x_left=-1.0,
        x_right=1.0,
        final_time=0.1663,
        source_kind="exponential",
        source_strength=3.0,
        spatial_kind="laplacian",
        reference_blowup_time=0.1664,
        paper_global_error=0.0448,
        paper_terminal_error=0.1179,
        paper_interior_points=200,
    ),
    "cubic": ProblemSpec(
        key="cubic",
        title="Semilinear shifted-cubic model",
        x_left=-1.0,
        x_right=1.0,
        final_time=0.0264,
        source_kind="shifted_cubic",
        source_strength=5.0,
        spatial_kind="laplacian",
        reference_blowup_time=0.0265,
        paper_global_error=0.0855,
        paper_terminal_error=0.1861,
        paper_interior_points=2500,
    ),
    "fourth": ProblemSpec(
        key="fourth",
        title="Fourth-order exponential model",
        x_left=0.0,
        x_right=5.0,
        # Fixed-domain BPINN is evaluated immediately before the reference time.
        final_time=0.9600,
        source_kind="exponential",
        source_strength=1.0,
        spatial_kind="negative_biharmonic",
        reference_blowup_time=FOURTH_ORDER_REFERENCE_TIME,
        paper_global_error=0.0704,
        paper_terminal_error=0.1377,
        paper_interior_points=200,
    ),
    "convection": ProblemSpec(
        key="convection",
        title="Convection-diffusion exponential model",
        x_left=-1.0,
        x_right=1.0,
        final_time=0.16664,
        source_kind="exponential",
        source_strength=3.0,
        spatial_kind="convection_diffusion",
        reference_blowup_time=0.16665,
        paper_global_error=0.02825,
        paper_terminal_error=0.09587,
        paper_interior_points=400,
    ),
}


def get_problem(key: str) -> ProblemSpec:
    try:
        return PROBLEMS[key]
    except KeyError as error:
        raise ValueError(f"Unknown problem {key!r}; choose from {sorted(PROBLEMS)}") from error
