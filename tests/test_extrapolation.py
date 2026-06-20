import numpy as np

from atr_bpinn.config import ExperimentConfig
from atr_bpinn.extrapolation import combine_terminal_estimates, extrapolate_thresholds


def test_recovers_asymptotic_intercept() -> None:
    config = ExperimentConfig()
    gamma = np.asarray(config.gammas)
    expected = 0.96
    detected = expected - 1.25 / gamma + 0.4 / gamma**2
    result = extrapolate_thresholds(1.2, gamma, detected, config)
    assert result.valid
    assert abs(result.estimate - expected) < 2.0e-4


def test_rejects_large_threshold_jump() -> None:
    config = ExperimentConfig()
    detected = [0.92, 0.94, 0.951, 0.965, 0.94]
    result = extrapolate_thresholds(1.2, config.gammas, detected, config)
    assert not result.valid
    assert "predictions_not_monotone_in_gamma" in result.reasons


def test_combines_terminal_estimates() -> None:
    config = ExperimentConfig()
    gamma = np.asarray(config.gammas)
    results = [
        extrapolate_thresholds(T, gamma, 0.96 - scale / gamma, config)
        for T, scale in ((1.2, 1.0), (1.5, 1.1), (1.8, 0.9))
    ]
    combined = combine_terminal_estimates(results, config)
    assert combined.valid
    assert abs(combined.estimate - 0.96) < 1.0e-10

