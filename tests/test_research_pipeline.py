import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.pipeline import evaluate_sweep, sharpe
from research.engines.reference import sweep, flush_bounce_signals


def _good(n=300, seed=1):
    return np.random.default_rng(seed).normal(0.004, 0.01, n)     # Sharpe ~0.4

def _noise(n=300, seed=0):
    return np.random.default_rng(seed).normal(0.0, 0.01, n)       # Sharpe ~0


def test_pipeline_picks_real_edge_over_noise():
    cfgs = {f"noise{i}": _noise(seed=i) for i in range(40)}
    cfgs["EDGE"] = _good(seed=999)
    hold = {k: (_good(seed=123) if k == "EDGE" else _noise(seed=k_i))
            for k_i, k in enumerate(cfgs)}
    res = evaluate_sweep(cfgs, hold)
    assert res.chosen == "EDGE"
    assert res.passed_gate is True and res.robust is True
    assert res.deflated_sharpe > 0
    assert "DEPLOY-CANDIDATE" in res.verdict


def test_pipeline_rejects_all_noise():
    cfgs = {f"noise{i}": _noise(seed=i) for i in range(40)}
    hold = {k: _noise(seed=100 + i) for i, k in enumerate(cfgs)}
    res = evaluate_sweep(cfgs, hold)
    # best-of-40 noise must not clear the deflation bar
    assert res.passed_gate is False
    assert "REJECT" in res.verdict


def test_pipeline_rejects_on_holdout_failure():
    cfgs = {f"noise{i}": _noise(seed=i) for i in range(20)}
    cfgs["EDGE"] = _good(seed=7)
    hold = dict({k: _noise(seed=200 + i) for i, k in enumerate(cfgs)})
    hold["EDGE"] = np.random.default_rng(5).normal(-0.004, 0.01, 200)   # flips negative
    res = evaluate_sweep(cfgs, hold)
    assert res.chosen == "EDGE"
    assert "holdout" in res.verdict.lower() and "REJECT" in res.verdict


def test_sharpe_zero_on_degenerate():
    assert sharpe([0.01]) == 0.0
    assert sharpe([]) == 0.0


# --- reference flush-bounce engine ---------------------------------------

def _bars_with_flush(n=200, seed=3):
    # random walk with periodic sharp flushes that then bounce
    rng = np.random.default_rng(seed)
    close = [100.0]
    for d in range(1, n):
        if d % 25 == 0:
            close.append(close[-1] * 0.94)          # -6% flush
        elif d % 25 in (1, 2, 3):
            close.append(close[-1] * 1.02)          # bounce
        else:
            close.append(close[-1] * (1 + rng.normal(0, 0.004)))
    close = np.array(close)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": np.full(n, 1e6)})


def test_flush_signals_fire_on_flush_only():
    bars = _bars_with_flush()
    sig = flush_bounce_signals(bars, flush_pct=0.05, drop_window=1, rsi_max=100)
    assert sig.sum() > 0                              # the -6% bars trip a 5% flush
    # a 20% flush threshold should fire far less (or never) on the same data
    strict = flush_bounce_signals(bars, flush_pct=0.20, drop_window=1, rsi_max=100)
    assert strict.sum() <= sig.sum()


def test_reference_sweep_shape():
    bars = _bars_with_flush()
    out = sweep(bars, flush_grid=[0.04, 0.06], rsi_grid=[50, 100], drop_window=1)
    assert set(out) == {"flush0.04_rsi50", "flush0.04_rsi100",
                        "flush0.06_rsi50", "flush0.06_rsi100"}
    assert all(isinstance(v, np.ndarray) for v in out.values())
