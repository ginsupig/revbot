import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reversion_bot.exhaustion import relative_volume, exhaustion_short_signal
from reversion_bot.walkforward import short_exhaustion_strategy, run_exhaustion_walkforward
from reversion_bot.analytics import profit_factor


def _df(highs, volumes):
    highs = np.asarray(highs, dtype=float)
    close = highs - 0.5
    return pd.DataFrame({
        "open": close,
        "high": highs,
        "low": close - 0.5,
        "close": close,
        "volume": np.asarray(volumes, dtype=float),
    })


def test_relative_volume_is_one_at_average():
    rvol = relative_volume(pd.Series([100.0] * 30), lookback=5)
    assert abs(float(rvol.iloc[-1]) - 1.0) < 1e-9


def test_exhaustion_fires_on_higher_high_with_fading_volume():
    n = 40
    df = _df(np.linspace(100, 110, n), np.linspace(200_000, 100_000, n))  # up, vol fading
    sig = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0)
    assert bool(sig.iloc[-1]) is True


def test_no_exhaustion_when_volume_expands():
    n = 40
    df = _df(np.linspace(100, 110, n), np.linspace(100_000, 200_000, n))  # up, vol rising
    sig = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0)
    assert bool(sig.iloc[-1]) is False


def test_no_exhaustion_without_a_new_high():
    n = 40
    df = _df(np.full(n, 100.0) - np.linspace(0, 5, n), np.linspace(200_000, 100_000, n))  # falling
    sig = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0)
    assert bool(sig.iloc[-1]) is False


def test_short_strategy_profits_when_a_failed_push_rolls_over():
    # High volume on the way up (no signal), then ONE final new high on thin
    # volume (the exhaustion), then a sharp drop -> the single short wins.
    highs = np.concatenate([np.linspace(100, 109, 40), [111.0], np.linspace(110, 95, 15)])
    vols = np.concatenate([np.full(40, 200_000.0), [40_000.0], np.full(15, 150_000.0)])
    res = short_exhaustion_strategy(
        _df(highs, vols), hh_lookback=5, rvol_lookback=5, rvol_max=1.0,
        require_divergence=False, cost_pct=0.0,
    )
    assert "pnl" in res.columns and "signal" in res.columns
    assert int((res["signal"] == -1).sum()) >= 1     # the exhaustion short fired
    assert np.isfinite(res["pnl"].to_numpy()).all()
    assert res["pnl"].sum() > 0                       # the rollover paid


def test_short_strategy_no_trades_on_quiet_tape():
    df = _df(np.linspace(100, 101, 60), np.linspace(100_000, 200_000, 60))  # rising vol -> no signal
    res = short_exhaustion_strategy(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0)
    assert int((res["signal"] == -1).sum()) == 0
    assert res["pnl"].abs().sum() == 0.0


def test_run_exhaustion_walkforward_smoke():
    rng = np.random.default_rng(0)
    n = 240
    highs = 100 + np.cumsum(rng.normal(0, 0.3, n))
    vols = rng.uniform(80_000, 200_000, n)
    grid = {"rvol_max": [1.0], "hh_lookback": [10], "require_divergence": [False], "target_atr_multiple": [2.0]}
    _results, oos, best = run_exhaustion_walkforward(_df(highs, vols), param_grid=grid, n_splits=2, verbose=False)
    assert len(oos) == 2
    assert "profit_factor" in oos[0]
    assert isinstance(best, dict)
