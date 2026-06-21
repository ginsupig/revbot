import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from research.selection import (
    deflated_sharpe_ratio, passes_dsr_gate, _sharpe_and_moments, deflated_sharpe,
)


def test_moments_recover_normal():
    xs = np.random.default_rng(0).normal(0, 1, 5000)
    sr, skew, kurt, n = _sharpe_and_moments(xs)
    assert abs(skew) < 0.1 and abs(kurt - 3.0) < 0.3 and n == 5000


def test_degenerate_series():
    assert deflated_sharpe_ratio([], 5) == 0.0
    assert deflated_sharpe_ratio([0.01], 5) == 0.0          # n<2 -> 0
    # zero variance -> Sharpe 0 -> well below the best-of-5 bar -> low probability
    assert deflated_sharpe_ratio([0.01, 0.01, 0.01], 5) < 0.5


def test_clean_edge_passes_dsr():
    xs = np.random.default_rng(1).normal(0.004, 0.01, 400)  # Sharpe ~0.4, near-normal
    assert deflated_sharpe_ratio(xs, n_trials=1) > 0.95
    assert passes_dsr_gate(xs, n_trials=1) is True


def test_fat_tails_lower_dsr_at_same_sharpe():
    # the upgrade's whole point: at equal Sharpe, fat tails widen the estimator SE,
    # so the moment-adjusted DSR is a HIGHER bar (lower pass probability).
    rng = np.random.default_rng(2)
    n = 400
    normal = rng.normal(0.003, 0.01, n)
    fat = rng.standard_t(3, n)
    fat = fat / fat.std() * normal.std() + normal.mean()   # match mean/std
    assert deflated_sharpe_ratio(fat, 10) < deflated_sharpe_ratio(normal, 10)


def test_noise_fails_dsr():
    xs = np.random.default_rng(3).normal(0.0, 0.01, 300)
    assert passes_dsr_gate(xs, n_trials=50) is False


def test_more_trials_raise_the_bar():
    xs = np.random.default_rng(4).normal(0.003, 0.01, 400)
    assert deflated_sharpe_ratio(xs, n_trials=1) >= deflated_sharpe_ratio(xs, n_trials=200)


def test_backward_compat_plain_deflation_untouched():
    # the simple normal-approx deflation is unchanged (additive upgrade)
    assert deflated_sharpe(0.5, 1, 100) == 0.5            # n_trials<=1 -> bar 0
    assert deflated_sharpe(0.5, 50, 100) < 0.5           # best-of-50 bar > 0
