from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dst, idst
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .problems import ProblemSpec


@dataclass(frozen=True)
class ReferenceGrid:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


def _source_numpy(problem: ProblemSpec, values: np.ndarray) -> np.ndarray:
    if problem.source_kind == "exponential":
        return np.exp(np.clip(values, -50.0, 50.0))
    if problem.source_kind == "shifted_cubic":
        return (values + 1.0) ** 3
    raise ValueError(problem.source_kind)


def _second_order_reference(
    problem: ProblemSpec, x_eval: np.ndarray, t_eval: np.ndarray, modes: int
) -> ReferenceGrid:
    dx = problem.domain_length / (modes + 1)
    interior_x = problem.x_left + dx * np.arange(1, modes + 1)
    initial = np.cos(np.pi * interior_x / 2.0)

    def rhs(_time: float, values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, (1, 1))
        u_xx = (padded[2:] - 2.0 * values + padded[:-2]) / dx**2
        spatial = u_xx
        if problem.spatial_kind == "convection_diffusion":
            u_x = (padded[2:] - padded[:-2]) / (2.0 * dx)
            spatial = spatial - u_x
        return spatial + problem.source_strength * _source_numpy(problem, values)

    sparsity = diags(
        [np.ones(modes - 1), np.ones(modes), np.ones(modes - 1)], [-1, 0, 1],
        shape=(modes, modes),
        format="csc",
    )
    solution = solve_ivp(
        rhs,
        (0.0, problem.final_time),
        initial,
        method="BDF",
        t_eval=t_eval,
        rtol=2.0e-8,
        atol=1.0e-10,
        jac_sparsity=sparsity,
        max_step=problem.final_time / 300.0,
    )
    if not solution.success:
        raise RuntimeError(f"Reference solve failed for {problem.key}: {solution.message}")
    full_x = np.concatenate(([problem.x_left], interior_x, [problem.x_right]))
    full_u = np.pad(solution.y.T, ((0, 0), (1, 1)))
    interpolated = np.vstack([np.interp(x_eval, full_x, row) for row in full_u])
    return ReferenceGrid(x_eval, t_eval, interpolated)


def _fourth_order_reference(
    problem: ProblemSpec, x_eval: np.ndarray, t_eval: np.ndarray, modes: int
) -> ReferenceGrid:
    wave_numbers = np.arange(1, modes + 1, dtype=float)
    eigenvalues = (np.pi * wave_numbers / problem.domain_length) ** 4

    def rhs(_time: float, values: np.ndarray) -> np.ndarray:
        coefficients = dst(values, type=1, norm="ortho")
        biharmonic = idst(-eigenvalues * coefficients, type=1, norm="ortho")
        return biharmonic + np.exp(np.clip(values, -50.0, 50.0))

    solution = solve_ivp(
        rhs,
        (0.0, problem.final_time),
        np.zeros(modes),
        method="BDF",
        t_eval=t_eval,
        rtol=2.0e-8,
        atol=1.0e-10,
        max_step=problem.final_time / 300.0,
    )
    if not solution.success:
        raise RuntimeError(f"Reference solve failed for {problem.key}: {solution.message}")
    interior_x = problem.x_left + problem.domain_length * np.arange(1, modes + 1) / (modes + 1)
    full_x = np.concatenate(([problem.x_left], interior_x, [problem.x_right]))
    full_u = np.pad(solution.y.T, ((0, 0), (1, 1)))
    interpolated = np.vstack([np.interp(x_eval, full_x, row) for row in full_u])
    return ReferenceGrid(x_eval, t_eval, interpolated)


def solve_example_reference(
    problem: ProblemSpec,
    x_points: int = 161,
    t_points: int = 161,
    modes: int = 256,
) -> ReferenceGrid:
    x_eval = np.linspace(problem.x_left, problem.x_right, x_points)
    t_eval = np.linspace(0.0, problem.final_time, t_points)
    if problem.is_fourth_order:
        return _fourth_order_reference(problem, x_eval, t_eval, modes)
    return _second_order_reference(problem, x_eval, t_eval, modes)

