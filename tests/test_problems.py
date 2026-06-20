import torch

from atr_bpinn.problems import PROBLEMS


def test_transform_inverse_round_trip() -> None:
    values = torch.tensor([[0.0], [0.5], [2.0]], dtype=torch.float64)
    for problem in PROBLEMS.values():
        transformed = problem.transform(values)
        recovered = problem.inverse_transform(transformed)
        assert torch.allclose(recovered, values, atol=1.0e-12)


def test_fourth_reference_time() -> None:
    assert PROBLEMS["fourth"].reference_blowup_time == 0.96508
    assert PROBLEMS["fourth"].final_time == 0.96
