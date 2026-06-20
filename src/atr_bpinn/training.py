from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import ExperimentConfig
from .model import BPINN, derivative


@dataclass(frozen=True)
class LossRecord:
    total: float
    pde: float
    geometric: float
    navier: float

    def max_component(self) -> float:
        return max(self.pde, self.geometric, self.navier)


@dataclass(frozen=True)
class Detection:
    time: float | None
    max_gradient: float
    valid_fraction: float


def torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_training_points(
    config: ExperimentConfig,
    horizon: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train = config.training
    dtype = torch_dtype(train.dtype)
    generator = make_generator(seed, device)
    x = config.domain_length * torch.rand(
        (train.n_interior, 1), generator=generator, device=device, dtype=dtype
    )
    uniform = torch.rand(
        (train.n_interior, 1), generator=generator, device=device, dtype=dtype
    )
    power = train.end_time_sampling_power
    t = horizon * (1.0 - (1.0 - uniform) ** power)
    t_uniform = torch.rand(t.shape, generator=generator, device=device, dtype=dtype)
    t = torch.cat((t, horizon * t_uniform), dim=0)
    x = torch.cat(
        (
            x,
            config.domain_length
            * torch.rand((train.n_interior, 1), generator=generator, device=device, dtype=dtype),
        ),
        dim=0,
    )
    boundary_t = horizon * torch.rand(
        (train.n_boundary, 1), generator=generator, device=device, dtype=dtype
    )
    boundary_x0 = torch.zeros_like(boundary_t)
    boundary_x1 = torch.full_like(boundary_t, config.domain_length)
    return x, t, boundary_x0, boundary_x1


def component_losses(
    model: BPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    boundary_x0: torch.Tensor,
    boundary_x1: torch.Tensor,
    boundary_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)
    u, g = model(x, t)
    g_t = derivative(g, t)
    u_xxxx = derivative(u, x, order=4)
    exp_minus_u = torch.exp(torch.clamp(-u, min=-50.0, max=20.0))
    pde_residual = 1.0 + g_t - exp_minus_u * u_xxxx
    geometric_residual = g - exp_minus_u
    # The inverse map only exists for g>0. Past the learned zero crossing there
    # is no physical post-blow-up solution, so those points must not force the
    # geometric relation back into its domain. The PDE residual still trains
    # the continuation and keeps the crossing identifiable.
    physical_mask = (g.detach() > 0.0).to(g.dtype)

    if boundary_t is None:
        boundary_t = torch.linspace(
            0.0,
            model.horizon,
            boundary_x0.shape[0],
            device=boundary_x0.device,
            dtype=boundary_x0.dtype,
        ).reshape(-1, 1)
    navier_terms: list[torch.Tensor] = []
    for boundary_x in (boundary_x0, boundary_x1):
        xb = boundary_x.detach().requires_grad_(True)
        tb = boundary_t.detach().clone().requires_grad_(True)
        ub, _ = model(xb, tb)
        u_xx = derivative(ub, xb, order=2)
        navier_terms.append(u_xx)

    return (
        torch.mean(pde_residual.square()),
        torch.sum(physical_mask * geometric_residual.square())
        / torch.clamp(torch.sum(physical_mask), min=1.0),
        torch.mean(torch.cat(navier_terms).square()),
    )


def train_bpinn(
    config: ExperimentConfig,
    horizon: float,
    seed: int,
    device: torch.device,
    model: BPINN | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[BPINN, LossRecord]:
    train = config.training
    dtype = torch_dtype(train.dtype)
    torch.manual_seed(seed)
    if model is None:
        model = BPINN(train.width, train.hidden_layers, config.domain_length, horizon)
        model = model.to(device=device, dtype=dtype)
    # A warm-started model keeps its original input scaling. Changing the
    # normalization at every refinement would change the represented function.

    x, t, bx0, bx1 = sample_training_points(config, horizon, seed + 17, device)
    boundary_t = horizon * torch.linspace(
        0.0, 1.0, train.n_boundary, device=device, dtype=dtype
    ).reshape(-1, 1)

    def objective() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pieces = component_losses(model, x, t, bx0, bx1, boundary_t)
        pde, geometric, navier = pieces
        total = (
            train.weight_pde * pde
            + train.weight_geo * geometric
            + train.weight_navier * navier
        )
        return total, pieces

    optimizer = torch.optim.Adam(model.parameters(), lr=train.adam_lr)
    for step in range(train.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, _ = objective()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
        optimizer.step()
        if progress and (step + 1) % max(train.adam_steps // 4, 1) == 0:
            progress(f"Adam {step + 1}/{train.adam_steps}: loss={total.item():.3e}")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=train.lbfgs_lr,
        max_iter=train.lbfgs_steps,
        max_eval=int(train.lbfgs_steps * 1.25),
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer_lbfgs.zero_grad(set_to_none=True)
        total, _ = objective()
        total.backward()
        return total

    optimizer_lbfgs.step(closure)
    total, pieces = objective()
    pde, geometric, navier = pieces
    record = LossRecord(
        total=float(total.detach().cpu()),
        pde=float(pde.detach().cpu()),
        geometric=float(geometric.detach().cpu()),
        navier=float(navier.detach().cpu()),
    )
    return model, record


def detect_threshold(
    model: BPINN,
    config: ExperimentConfig,
    horizon: float,
    threshold: float,
    device: torch.device,
) -> Detection:
    dtype = next(model.parameters()).dtype
    times = torch.linspace(
        0.0, horizon, config.detection_time_points, device=device, dtype=dtype
    )
    space = torch.linspace(
        0.0, config.domain_length, config.detection_space_points, device=device, dtype=dtype
    )
    earliest: float | None = None
    global_max = 0.0
    valid_count = 0
    total_count = 0
    previous_time = 0.0
    previous_gradient = 0.0

    for time_value in times:
        x = space.reshape(-1, 1).detach().requires_grad_(True)
        t = torch.full_like(x, time_value).requires_grad_(True)
        _, g = model(x, t)
        g_t = derivative(g, t)
        valid = g > config.minimum_positive_g
        temporal_gradient = torch.full_like(g, float("nan"))
        temporal_gradient[valid] = torch.abs(g_t[valid] / g[valid])
        total_count += valid.numel()
        valid_count += int(valid.sum().detach().cpu())
        if bool(valid.any()):
            max_gradient = float(torch.nan_to_num(temporal_gradient, nan=0.0).max().detach().cpu())
            global_max = max(global_max, max_gradient)
            if max_gradient >= threshold:
                current_time = float(time_value.detach().cpu())
                if max_gradient > previous_gradient:
                    fraction = (threshold - previous_gradient) / (max_gradient - previous_gradient)
                    fraction = min(max(float(fraction), 0.0), 1.0)
                    earliest = previous_time + fraction * (current_time - previous_time)
                else:
                    earliest = current_time
                break
            previous_gradient = max_gradient
            previous_time = float(time_value.detach().cpu())

    return Detection(
        time=earliest,
        max_gradient=global_max,
        valid_fraction=valid_count / max(total_count, 1),
    )


def device_from_name(name: str, dtype_name: str = "float64") -> torch.device:
    if name == "auto":
        # Apple MPS does not support float64, which is important for stable
        # fourth derivatives. Prefer CPU unless the profile explicitly uses
        # float32.
        if dtype_name == "float32" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and dtype_name == "float64":
        raise ValueError("MPS does not support the float64 publication profile; use --device cpu")
    return torch.device(name)


def numpy_state_dict(model: BPINN) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
