"""Validation of the corrected trendfail: detects real fakeouts, ignores real
breakouts, never revises history."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from reversion_bot.trendfail import trend_fail_continue_strategy


def ohlcv(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(len(close), 1e5),
    })


def test_detects_planted_fakeout_on_failure_bar():
    # 30 flat bars ~100, breakout to 102 at bar 30, collapse to 99 at bar 32.
    c = [100.0] * 30 + [102.0, 101.2, 99.0] + [99.5] * 10
    out = trend_fail_continue_strategy(ohlcv(c), window=20, confirmation_window=3, threshold=0.005)
    assert bool(out["breakout_up"].iloc[30])
    assert not bool(out["fakeout_up"].iloc[30])      # never on the breakout bar
    assert bool(out["fakeout_up"].iloc[32])          # fires on the FAILURE bar
    assert int(out["signal"].iloc[32]) == -1


def test_genuine_breakout_is_not_faded():
    # Breakout that holds and keeps climbing: no fakeout, no short.
    c = [100.0] * 30 + [102.0, 102.5, 103.0, 103.5, 104.0] + [104.5] * 10
    out = trend_fail_continue_strategy(ohlcv(c), window=20, confirmation_window=3, threshold=0.005)
    assert bool(out["breakout_up"].iloc[30])
    assert int(out["fakeout_up"].sum()) == 0
    assert int((out["signal"] == -1).sum()) == 0


def test_uptrend_no_longer_net_short():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0006, 0.004, 2000)
    c = 100 * np.cumprod(1 + rets)
    out = trend_fail_continue_strategy(ohlcv(c), window=20, confirmation_window=3, threshold=0.005)
    mean_pos = float(out["position"].mean())
    # Old version: -0.285. Fixed version must not be structurally short a trend.
    assert mean_pos > -0.05, f"still structurally short: {mean_pos:+.3f}"


def test_no_lookahead():
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.005, 600)
    c = 100 * np.cumprod(1 + rets)
    df = ohlcv(c)
    full = trend_fail_continue_strategy(df, window=20, confirmation_window=3, threshold=0.005)
    for cut in (200, 350, 500):
        part = trend_fail_continue_strategy(df.iloc[:cut].copy(), window=20,
                                            confirmation_window=3, threshold=0.005)
        pd.testing.assert_series_equal(
            full["signal"].iloc[:cut].reset_index(drop=True),
            part["signal"].reset_index(drop=True),
            check_names=False,
        )


def test_downside_mirror():
    c = [100.0] * 30 + [98.0, 98.5, 101.0] + [101.0] * 10
    out = trend_fail_continue_strategy(ohlcv(c), window=20, confirmation_window=3, threshold=0.005)
    assert bool(out["breakout_down"].iloc[30])
    assert bool(out["fakeout_down"].iloc[32])
    assert int(out["signal"].iloc[32]) == 1
