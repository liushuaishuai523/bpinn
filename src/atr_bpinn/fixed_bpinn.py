from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import derivative
from .problems import ProblemSpec
from .training import torch_dtype


@dataclass(frozen=True)
class FixedTrainingConfig:
    width: int = 50
    hidden_layers: int = 3
    adam_steps: int = 1200
    adam_lr: float = 2.0e-3
    lbfgs_steps: int = 300
    lbfgs_lr: float = 0.8
    interior_points_scale: float = 1.0
    geometric_weight: float = 10.0
    navier_weight: float = 1.0
    end_time_sampling_power: float = 2.0
    dtype: str = "float64"

    @classmethod
    def quick(cls) -> "FixedTrainingConfig":
        return cls(
            width=36,
            hidden_layers=3,
            adam_steps=500,
            lbfgs_steps=140,
            interior_points_scale=1.0,
        )


@dataclass(frozen=True)
class FixedLosses:
    total: float
    pde: float
    geometric: float
    initial: float
    boundary: float
    navier: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FixedTrainingResult:
    losses: FixedLosses
    history: tuple[float, ...]


class FixedBPINN(nn.Module):
    """BPINN on a fixed pre-blow-up space-time domain."""

    def __init__(self, problem: ProblemSpec, config: FixedTrainingConfig):
        super().__init__()
        self.problem = problem
        input_width = 3 if problem.spatial_kind == "convection_diffusion" else 2
        layers: list[nn.Module] = [nn.Linear(input_width, config.width), nn.Tanh()]
        for _ in range(config.hidden_layers - 1):
            layers.extend([nn.Linear(config.width, config.width), nn.Tanh()])
        layers.append(nn.Linear(config.width, 2))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.bias[0] = 0.3
            output.bias[1] = 1.0

    def _space_shape(self, x: torch.Tensor) -> torch.Tensor:
        problem = self.problem
        phase = math.pi * (x - problem.x_left) / problem.domain_length
        if problem.is_fourth_order:
            return torch.sin(phase)
        xi = 2.0 * (x - problem.x_left) / problem.domain_length - 1.0
        return 1.0 - xi.square()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        problem = self.problem
        x_scaled = 2.0 * (x - problem.x_left) / problem.domain_length - 1.0
        tau = t / problem.final_time
        features = [x_scaled, 2.0 * tau - 1.0]
        if problem.spatial_kind == "convection_diffusion":
            features.append(x_scaled - 2.0 * t / problem.domain_length)
        raw = self.network(torch.cat(features, dim=1))
        shape = tau * self._space_shape(x)
        initial_u = problem.initial_condition(x)
        initial_g = problem.transform(initial_u)
        u = initial_u + shape * F.softplus(raw[:, :1])
        # On a fixed pre-blow-up domain g is positive and decreases from its
        # initial value. This keeps the inverse map valid.
        g = initial_g * torch.exp(-shape * F.softplus(raw[:, 1:2]))
        return u, g


def _sample_points(
    problem: ProblemSpec,
    config: FixedTrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch_dtype(config.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    count = max(64, int(round(problem.paper_interior_points * config.interior_points_scale)))
    x1 = problem.x_left + problem.domain_length * torch.rand(
        (count, 1), generator=generator, device=device, dtype=dtype
    )
    r1 = torch.rand((count, 1), generator=generator, device=device, dtype=dtype)
    t1 = problem.final_time * (1.0 - (1.0 - r1) ** config.end_time_sampling_power)
    x2 = problem.x_left + problem.domain_length * torch.rand(
        (count, 1), generator=generator, device=device, dtype=dtype
    )
    t2 = problem.final_time * torch.rand(
        (count, 1), generator=generator, device=device, dtype=dtype
    )
    return torch.cat((x1, x2)), torch.cat((t1, t2))


def fixed_component_losses(
    model: FixedBPINN,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    problem = model.problem
    x = x.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)
    u, g = model(x, t)
    g_t = derivative(g, t)
    recovered_u = problem.inverse_transform(g)
    spatial = problem.spatial_operator(recovered_u, x)
    pde_residual = (
        problem.source_strength + g_t + spatial / problem.source(recovered_u)
    )
    transformed_u = problem.transform(u)
    # A raw absolute residual becomes blind as both terms approach zero near
    # blow-up. The log residual controls relative mismatch.
    geometric_residual = torch.log(torch.clamp(g, min=1.0e-14)) - torch.log(
        torch.clamp(transformed_u, min=1.0e-14)
    )

    if problem.is_fourth_order:
        boundary_t = torch.linspace(
            0.0, problem.final_time, 32, device=x.device, dtype=x.dtype
        ).reshape(-1, 1)
        navier = []
        for location in (problem.x_left, problem.x_right):
            xb = torch.full_like(boundary_t, location).requires_grad_(True)
            _, gb = model(xb, boundary_t)
            recovered_boundary = problem.inverse_transform(gb)
            navier.append(derivative(recovered_boundary, xb, order=2))
        navier_loss = torch.mean(torch.cat(navier).square())
    else:
        navier_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    return (
        torch.mean(pde_residual.square()),
        torch.mean(geometric_residual.square()),
        navier_loss,
    )


def train_fixed_bpinn(
    problem: ProblemSpec,
    config: FixedTrainingConfig,
    seed: int,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> tuple[FixedBPINN, FixedTrainingResult]:
    torch.manual_seed(seed)
    dtype = torch_dtype(config.dtype)
    model = FixedBPINN(problem, config).to(device=device, dtype=dtype)
    x, t = _sample_points(problem, config, seed + 31, device)
    history: list[float] = []

    def objective() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pde, geometric, navier = fixed_component_losses(model, x, t)
        total = pde + config.geometric_weight * geometric + config.navier_weight * navier
        return total, (pde, geometric, navier)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    log_every = max(config.adam_steps // 100, 1)
    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, _ = objective()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        if step % log_every == 0 or step + 1 == config.adam_steps:
            history.append(float(total.detach().cpu()))
        if progress and (step + 1) % max(config.adam_steps // 4, 1) == 0:
            progress(f"{problem.key}: Adam {step + 1}/{config.adam_steps}, loss={total.item():.3e}")

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_steps,
        max_eval=int(config.lbfgs_steps * 1.25),
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        total, _ = objective()
        total.backward()
        return total

    lbfgs.step(closure)
    total, pieces = objective()
    pde, geometric, navier = pieces
    history.append(float(total.detach().cpu()))
    losses = FixedLosses(
        total=float(total.detach().cpu()),
        pde=float(pde.detach().cpu()),
        geometric=float(geometric.detach().cpu()),
        initial=0.0,
        boundary=0.0,
        navier=float(navier.detach().cpu()),
    )
    return model, FixedTrainingResult(losses, tuple(history))


def predict_fixed_bpinn(
    model: FixedBPINN,
    x_values: np.ndarray,
    t_values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    xx, tt = np.meshgrid(x_values, t_values)
    x = torch.as_tensor(xx.reshape(-1, 1), device=device, dtype=dtype)
    t = torch.as_tensor(tt.reshape(-1, 1), device=device, dtype=dtype)
    with torch.no_grad():
        _, g = model(x, t)
        prediction = model.problem.inverse_transform(g)
    return prediction.detach().cpu().numpy().reshape(len(t_values), len(x_values))
