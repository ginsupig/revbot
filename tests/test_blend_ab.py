import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import blend_ab_backtest as ab
from reversion_bot.service import ReversionService
from reversion_bot.config import ReversionConfig, RiskConfig, PerformanceConfig


def _bars(n=180):
    idx = pd.date_range("2026-01-02 14:30", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    vol = rng.integers(80_000, 200_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_simulate_returns_both_modes_with_finite_pnl(tmp_path):
    svc = ReversionService(
        ReversionConfig(min_history=60, enable_shorts=True),
        RiskConfig(),
        PerformanceConfig(state_dir=str(tmp_path)),
        log_file=str(tmp_path / "svc.log"),
    )
    pnls = ab._simulate(_bars(), svc)
    assert set(pnls) == {"blended", "routed"}
    for mode in ("blended", "routed"):
        assert all(np.isfinite(p) for p in pnls[mode])


def test_metrics_handles_empty():
    n, net, pf, sr = ab._metrics([])
    assert n == 0 and net == 0.0 and np.isfinite(pf) and np.isfinite(sr)
