from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    result = y
    for _ in range(order):
        result = torch.autograd.grad(
            result,
            x,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
            retain_graph=True,
        )[0]
    return result


class BPINN(nn.Module):
    """Two-output BPINN for u_t = -u_xxxx + exp(u).

    Initial and Dirichlet conditions are embedded exactly through the factor
    tau*sin(pi*x/L). The second Navier condition is enforced by a loss term.
    """

    def __init__(self, width: int, hidden_layers: int, domain_length: float, horizon: float):
        super().__init__()
        self.domain_length = float(domain_length)
        self.horizon = float(horizon)
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.bias[0] = 0.2
            final.bias[1] = -1.0

    def set_horizon(self, horizon: float) -> None:
        self.horizon = float(horizon)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_scaled = 2.0 * x / self.domain_length - 1.0
        t_scaled = 2.0 * t / self.horizon - 1.0
        raw = self.network(torch.cat((x_scaled, t_scaled), dim=1))
        # Keep the physical time factor independent of the current ATR window.
        # This makes warm starts invariant when the terminal time is refined.
        shape = t * torch.sin(math.pi * x / self.domain_length)
        u = shape * F.softplus(raw[:, :1])
        g = 1.0 + shape * raw[:, 1:2]
        return u, g
