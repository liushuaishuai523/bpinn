import torch

from atr_bpinn.model import BPINN


def test_embedded_initial_and_dirichlet_boundaries() -> None:
    model = BPINN(width=8, hidden_layers=2, domain_length=5.0, horizon=1.8).double()
    interior_x = torch.linspace(0.0, 5.0, 9, dtype=torch.float64).reshape(-1, 1)
    initial_t = torch.zeros_like(interior_x)
    u_initial, g_initial = model(interior_x, initial_t)
    assert torch.allclose(u_initial, torch.zeros_like(u_initial))
    assert torch.allclose(g_initial, torch.ones_like(g_initial))

    for location in (0.0, 5.0):
        boundary_x = torch.full((6, 1), location, dtype=torch.float64, requires_grad=True)
        boundary_t = torch.linspace(0.0, 1.2, 6, dtype=torch.float64).reshape(-1, 1)
        u_boundary, g_boundary = model(boundary_x, boundary_t)
        assert torch.allclose(u_boundary, torch.zeros_like(u_boundary), atol=1.0e-12)
        assert torch.allclose(g_boundary, torch.ones_like(g_boundary), atol=1.0e-12)
