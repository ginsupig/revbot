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


# --- Long-side capitulation features ---------------------------------------

from reversion_bot.exhaustion import (
    selling_velocity_decelerating,
    lower_wick_fraction,
    close_position,
    capitulation_candle,
)


def _close_series_from_per_bar(per_bar_pcts):
    """Build a close series whose successive bar-over-bar returns match the given
    per-bar percentages (so a [-5,-3,-1] list literally produces those moves)."""
    close = [100.0]
    for p in per_bar_pcts:
        close.append(close[-1] * (1.0 + p / 100.0))
    return pd.DataFrame({"close": close})


def test_deceleration_true_when_drop_is_slowing():
    # A steep early decline that eases: -5, -4, -3, -2, -1 (per bar).
    df = _close_series_from_per_bar([-5, -4, -3, -2, -1])
    decel = selling_velocity_decelerating(df, short=2, long=4)
    assert bool(decel.iloc[-1]) is True       # recent pace shallower than overall


def test_deceleration_false_on_accelerating_knife():
    # An accelerating decline: -1, -2, -3, -4, -5 (per bar) — a falling knife.
    df = _close_series_from_per_bar([-1, -2, -3, -4, -5])
    decel = selling_velocity_decelerating(df, short=2, long=4)
    assert bool(decel.iloc[-1]) is False


def test_deceleration_requires_a_decline():
    # Rising tape: recent "pace" may look shallower but there's no pullback to
    # exhaust, so require_decline keeps it False.
    df = _close_series_from_per_bar([1, 2, 1, 2, 1])
    assert bool(selling_velocity_decelerating(df, short=2, long=4).iloc[-1]) is False


def test_lower_wick_fraction_hammer_vs_no_tail():
    df = pd.DataFrame({
        # hammer: open/close near the high, long tail down to the low
        "open":  [10.0, 10.0],
        "close": [10.2, 10.2],
        "high":  [10.3, 10.3],
        "low":   [ 9.0, 10.0],   # row0 long lower wick; row1 no lower wick
    })
    frac = lower_wick_fraction(df)
    assert frac.iloc[0] > 0.7                 # most of the range is tail
    assert abs(frac.iloc[1] - 0.0) < 1e-9     # low == body bottom -> no wick


def test_close_position_and_degenerate_bar():
    df = pd.DataFrame({
        "open":  [10.0, 5.0],
        "close": [10.3, 5.0],
        "high":  [10.3, 5.0],   # row1 high==low (flat bar)
        "low":   [ 9.0, 5.0],
    })
    pos = close_position(df)
    assert abs(pos.iloc[0] - 1.0) < 1e-9      # closed at the high
    assert abs(pos.iloc[1] - 0.5) < 1e-9      # degenerate -> neutral 0.5


def test_capitulation_candle_needs_wick_and_recovery():
    df = pd.DataFrame({
        # row0 capitulation: pushed down to 9.0, bought back, closes near the high
        # row1 falling knife: closes near its low, no lower tail
        # row2 plain green push: closes high but no lower wick
        "open":  [10.00, 10.00, 10.00],
        "high":  [10.30, 10.05, 10.45],
        "low":   [ 9.00,  9.00,  9.98],
        "close": [10.20,  9.05, 10.40],
    })
    cap = capitulation_candle(df, min_wick=0.5, min_close_pos=0.5)
    assert list(cap) == [True, False, False]


def test_range_atr_ratio_and_expansion():
    from reversion_bot.exhaustion import range_atr_ratio, range_expansion
    df = pd.DataFrame({
        "open":  [10.0, 10.0, 10.0],
        "high":  [10.5, 11.0, 12.0],   # ranges: 1.0, 2.0, 3.0
        "low":   [ 9.5,  9.0,  9.0],
        "close": [10.0, 10.0, 10.0],
    })
    atr = pd.Series([1.0, 1.0, 1.0])
    ratio = range_atr_ratio(df, atr)
    assert list(ratio) == [1.0, 2.0, 3.0]
    exp = range_expansion(df, atr, mult=1.5)
    assert list(exp) == [False, True, True]      # 1.0x not expanded; 2x/3x are


def test_range_expansion_handles_zero_atr():
    from reversion_bot.exhaustion import range_expansion
    df = pd.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.0]})
    # Zero ATR -> ratio NaN -> not flagged (fails open to False, never True).
    assert list(range_expansion(df, pd.Series([0.0]), mult=1.5)) == [False]
