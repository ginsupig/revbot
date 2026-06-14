import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from reversion_bot.analytics import (
    run_monte_carlo,
    run_bootstrap_monte_carlo,
    extract_trade_returns,
)


def test_monte_carlo_runs():
    out = run_monte_carlo(10000, 0.0005, 0.01, days=50, sims=500, seed=1)
    assert out['sims'] == 500
    assert out['days'] == 50
    assert 'var_95' in out


def test_bootstrap_runs_and_has_parallel_keys():
    returns = [0.012, -0.008, 0.02, -0.015, 0.005, -0.01, 0.03]
    out = run_bootstrap_monte_carlo(10000, returns, sims=2000, seed=1)
    # Same reporting shape as the normal MC, with n_trades replacing days.
    for k in ('initial_invest', 'expected_final_value', 'median_final_value',
              'best_case', 'worst_case', 'percentile_5_value', 'var_95', 'cvar_95'):
        assert k in out
    assert out['sims'] == 2000
    assert out['n_trades'] == len(returns)      # default path length = sample size
    assert out['sample_size'] == len(returns)
    assert out['worst_case'] <= out['median_final_value'] <= out['best_case']
    assert out['cvar_95'] >= out['var_95']      # CVaR is the deeper-tail loss


def test_bootstrap_is_deterministic_under_seed():
    returns = [0.01, -0.02, 0.03, -0.01]
    a = run_bootstrap_monte_carlo(1000, returns, sims=500, seed=7)
    b = run_bootstrap_monte_carlo(1000, returns, sims=500, seed=7)
    assert a == b


def test_bootstrap_all_positive_returns_cannot_lose():
    # If every observed trade won, no resampled path can end below the stake.
    out = run_bootstrap_monte_carlo(1000, [0.01, 0.02, 0.005], sims=500, seed=3)
    assert out['worst_case'] >= 1000
    assert out['var_95'] <= 0          # "loss" is non-positive


def test_bootstrap_captures_fat_left_tail_vs_normal():
    # A skewed book: many small wins, rare large losers. The empirical bootstrap
    # should see a deeper CVaR than a normal fitted to the same mean/std, which
    # smooths the tail away.
    rng = np.random.default_rng(0)
    returns = np.concatenate([rng.normal(0.01, 0.002, 200), np.full(8, -0.15)])
    boot = run_bootstrap_monte_carlo(10000, returns, n_trades=40, sims=5000, seed=1)
    norm = run_monte_carlo(10000, float(returns.mean()), float(returns.std()),
                           days=40, sims=5000, seed=1)
    assert boot['cvar_95'] > norm['cvar_95']


def test_bootstrap_n_trades_override():
    out = run_bootstrap_monte_carlo(1000, [0.01, -0.01], n_trades=100, sims=300, seed=1)
    assert out['n_trades'] == 100


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        run_bootstrap_monte_carlo(0, [0.01])              # bad stake
    with pytest.raises(ValueError):
        run_bootstrap_monte_carlo(1000, [])               # empty population
    with pytest.raises(ValueError):
        run_bootstrap_monte_carlo(1000, [0.01], sims=0)   # no sims


def test_extract_trade_returns_pulls_nonzero():
    df = pd.DataFrame({"strategy_return": [0.0, 0.01, 0.0, -0.02, np.nan, 0.03]})
    arr = extract_trade_returns(df)
    assert list(arr) == [0.01, -0.02, 0.03]
    # Falls back to pnl when strategy_return is absent.
    df2 = pd.DataFrame({"pnl": [0.0, 0.05, -0.01]})
    assert list(extract_trade_returns(df2)) == [0.05, -0.01]


def test_extract_then_bootstrap_pipeline():
    df = pd.DataFrame({"strategy_return": [0.0, 0.02, -0.01, 0.0, 0.015, -0.03]})
    out = run_bootstrap_monte_carlo(10000, extract_trade_returns(df), sims=500, seed=2)
    assert out['sample_size'] == 4
