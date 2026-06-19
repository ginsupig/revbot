import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.regime import label_regime, per_regime_stats
from research.cost import apply_cost, cost_sweep, breakeven_cost_bps


def _ohlc(close):
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": np.full(len(close), 1e6)})


def test_label_regime_downtrend_and_range():
    down = label_regime(_ohlc(np.linspace(150, 90, 160)))
    assert down.iloc[-1] == "downtrend"          # sustained decline + high ADX
    flat = label_regime(_ohlc(100 + np.zeros(160)))
    assert flat.iloc[-1] == "range"               # no trend -> range


def test_per_regime_stats_splits_edge():
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05]
    labels = ["uptrend", "uptrend", "uptrend", "downtrend", "downtrend", "downtrend"]
    out = per_regime_stats(returns, labels)
    assert out["uptrend"]["mean"] > 0 and out["downtrend"]["mean"] < 0
    assert out["uptrend"]["n"] == 3 and out["downtrend"]["n"] == 3


def test_cost_sweep_monotone_and_breakeven():
    returns = np.array([0.01, -0.005, 0.012, -0.008, 0.006])  # gross mean +0.0030 = 30 bps
    sweep = cost_sweep(returns, (0, 10, 20, 40))
    means = [sweep[b]["mean"] for b in (0, 10, 20, 40)]
    assert means == sorted(means, reverse=True)               # cost only lowers net
    be = breakeven_cost_bps(returns)
    assert abs(be - 30.0) < 1e-6                                # 0.0030 * 1e4
    # at the breakeven cost, net mean ~ 0
    assert abs(apply_cost(returns, be).mean()) < 1e-9


def test_cost_empty():
    assert breakeven_cost_bps([]) == 0.0
