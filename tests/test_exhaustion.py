import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reversion_bot.exhaustion import (
    relative_volume,
    exhaustion_short_signal,
    chaikin_money_flow,
)
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


def test_borrow_cost_reduces_short_pnl():
    # Same winning setup, but a borrow haircut must lower the realized pnl.
    highs = np.concatenate([np.linspace(100, 109, 40), [111.0], np.linspace(110, 95, 15)])
    vols = np.concatenate([np.full(40, 200_000.0), [40_000.0], np.full(15, 150_000.0)])
    kw = dict(hh_lookback=5, rvol_lookback=5, rvol_max=1.0, require_divergence=False, cost_pct=0.0)
    base = short_exhaustion_strategy(_df(highs, vols), **kw)["pnl"].sum()
    haircut = short_exhaustion_strategy(_df(highs, vols), borrow_cost_pct=0.0020, **kw)["pnl"].sum()
    assert base > 0                # sanity: the trade won pre-haircut
    assert haircut < base          # borrow cost eats into the edge


def test_fixed_kwargs_flow_through_walkforward():
    # A constant borrow haircut across the sweep must not error and cannot
    # improve PF versus no haircut.
    rng = np.random.default_rng(1)
    n = 240
    highs = 100 + np.cumsum(rng.normal(0, 0.3, n))
    vols = rng.uniform(80_000, 200_000, n)
    grid = {"rvol_max": [1.0], "hh_lookback": [10], "require_divergence": [False], "target_atr_multiple": [2.0]}
    df = _df(highs, vols)
    _r1, oos_plain, _b1 = run_exhaustion_walkforward(df, param_grid=grid, n_splits=2, verbose=False)
    _r2, oos_hair, _b2 = run_exhaustion_walkforward(
        df, param_grid=grid, n_splits=2, verbose=False, fixed_kwargs={"borrow_cost_pct": 0.01}
    )
    plain_pf = np.mean([m["profit_factor"] for m in oos_plain])
    hair_pf = np.mean([m["profit_factor"] for m in oos_hair])
    assert hair_pf <= plain_pf + 1e-9


# --- Chaikin Money Flow + new quality filters --------------------------------

def _ohlc(highs, lows, closes, vols):
    a = lambda x: np.asarray(x, dtype=float)
    return pd.DataFrame({"open": a(closes), "high": a(highs), "low": a(lows),
                         "close": a(closes), "volume": a(vols)})


def test_cmf_sign_matches_close_position():
    n = 30
    high = np.full(n, 101.0); low = np.full(n, 100.0); vol = np.full(n, 1e5)
    at_low = chaikin_money_flow(_ohlc(high, low, low, vol), period=21).iloc[-1]
    at_high = chaikin_money_flow(_ohlc(high, low, high, vol), period=21).iloc[-1]
    assert at_low < -0.9      # closing at the low => distribution
    assert at_high > 0.9      # closing at the high => accumulation


def test_cmf_filter_gates_the_signal():
    # A fixture that fires the base signal; the CMF filter should keep it when
    # the threshold is permissive and drop it when impossible.
    df = _df(np.linspace(100, 110, 40), np.linspace(200_000, 100_000, 40))
    base = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0)
    permissive = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0, cmf_max=1.0)
    impossible = exhaustion_short_signal(df, hh_lookback=5, rvol_lookback=5, rvol_max=1.0, cmf_max=-2.0)
    assert bool(base.iloc[-1]) is True
    assert bool(permissive.iloc[-1]) is True       # CMF <= 1 always true
    assert bool(impossible.iloc[-1]) is False        # CMF <= -2 never true


def test_extension_gate_blocks_unstretched_signals():
    highs = np.concatenate([np.linspace(100, 109, 40), [111.0], np.linspace(110, 95, 15)])
    vols = np.concatenate([np.full(40, 200_000.0), [40_000.0], np.full(15, 150_000.0)])
    kw = dict(hh_lookback=5, rvol_lookback=5, rvol_max=1.0, require_divergence=False, cost_pct=0.0)
    traded = short_exhaustion_strategy(_df(highs, vols), **kw)
    gated = short_exhaustion_strategy(_df(highs, vols), min_extension_atr=100.0, **kw)
    assert int((traded["signal"] == -1).sum()) >= 1   # fires without the gate
    assert int((gated["signal"] == -1).sum()) == 0     # impossible extension blocks all


def test_confirmation_skips_a_high_that_keeps_rising():
    # New high on thin volume, but price KEEPS rising afterwards (no rollover).
    # confirm_mode="none" shorts the high (and would get stopped); "medium" waits
    # for a lower close that never comes, so it correctly takes no trade.
    highs = np.concatenate([np.linspace(100, 109, 40), [111.0], np.linspace(112, 120, 15)])
    vols = np.concatenate([np.full(40, 200_000.0), [40_000.0], np.full(15, 200_000.0)])
    kw = dict(hh_lookback=5, rvol_lookback=5, rvol_max=1.0, require_divergence=False, cost_pct=0.0)
    no_confirm = short_exhaustion_strategy(_df(highs, vols), confirm_mode="none", **kw)
    confirmed = short_exhaustion_strategy(_df(highs, vols), confirm_mode="medium", **kw)
    assert int((no_confirm["signal"] == -1).sum()) >= 1   # naively shorts the high
    assert int((confirmed["signal"] == -1).sum()) == 0      # confirmation saves it


def test_confirmation_still_trades_a_real_rollover():
    # New high on thin volume, then a sharp drop -> confirmation passes and the
    # delayed short still gets taken and wins.
    highs = np.concatenate([np.linspace(100, 109, 40), [111.0], np.linspace(110, 92, 15)])
    vols = np.concatenate([np.full(40, 200_000.0), [40_000.0], np.full(15, 150_000.0)])
    kw = dict(hh_lookback=5, rvol_lookback=5, rvol_max=1.0, require_divergence=False, cost_pct=0.0)
    res = short_exhaustion_strategy(_df(highs, vols), confirm_mode="medium", **kw)
    assert int((res["signal"] == -1).sum()) >= 1
    assert res["pnl"].sum() > 0
