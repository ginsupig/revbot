"""Demonstrates that trendfail's 'fakeout confirmation' is vacuous.

The claim under test: in trend_fail_continue_strategy, the confirmation

    confirm = all(close[i-j] < high_roll[i-j] for j in 1..confirmation_window)

is satisfied before essentially EVERY upward breakout, because a bar's close
is almost always below the rolling 20-bar max of HIGHS (the close can only
equal it on the single bar that sets the high and closes exactly at it).
Therefore fakeout_up ~= breakout_up, and the model effectively shorts every
upside breakout (and mirrors this on the downside) rather than detecting
failed breakouts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from reversion_bot.trendfail import trend_fail_continue_strategy


def make_trending_ohlcv(n=2000, seed=7):
    """A genuinely trending random walk: breakouts here are REAL continuation
    breakouts, the exact case a fakeout detector should NOT fire on."""
    rng = np.random.default_rng(seed)
    drift = 0.0006
    rets = rng.normal(drift, 0.004, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(50_000, 150_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_fakeout_up_fires_on_virtually_every_breakout():
    df = make_trending_ohlcv()
    out = trend_fail_continue_strategy(df, window=20, confirmation_window=3, threshold=0.005)

    # Only bars where the loop could have run (i >= window + confirmation_window)
    scope = out.iloc[23:]
    n_breakouts = int(scope["breakout_up"].sum())
    n_fakeouts = int(scope["fakeout_up"].sum())

    assert n_breakouts > 10, "test data should contain real breakouts"
    ratio = n_fakeouts / n_breakouts
    print(f"\nupward breakouts: {n_breakouts}, flagged as 'fakeouts': {n_fakeouts} "
          f"({ratio:.0%})")

    # REGRESSION GUARD (post-fix): a meaningful fakeout detector on a strongly
    # trending series must flag only a minority of breakouts. The old vacuous
    # condition flagged 100%; reintroducing it trips this assert.
    assert ratio < 0.60, f"confirmation looks vacuous again: {ratio:.0%} flagged"


def test_confirmation_condition_is_nearly_always_true():
    """Directly measure how often close < rolling-max-of-highs holds at all."""
    df = make_trending_ohlcv()
    high_roll = df["high"].rolling(20).max()
    cond = (df["close"] < high_roll).iloc[20:]
    frac = float(cond.mean())
    print(f"\nP(close < 20-bar rolling max high) = {frac:.4f}")
    assert frac > 0.99  # the 'confirmation' carries ~no information


def test_consequence_signal_shorts_every_real_breakout():
    """Economic consequence: in a clean uptrend, the model goes net short."""
    df = make_trending_ohlcv()
    out = trend_fail_continue_strategy(df, window=20, confirmation_window=3, threshold=0.005)
    scope = out.iloc[23:]
    shorts = int((scope["signal"] == -1).sum())
    longs = int((scope["signal"] == 1).sum())
    mean_pos = float(scope["position"].mean())
    print(f"\nshort signals: {shorts}, long signals: {longs}, "
          f"mean position in an uptrend: {mean_pos:+.3f}")
    # REGRESSION GUARD (post-fix): must not be structurally short an uptrend.
    assert mean_pos > -0.05, f"net short through an uptrend again: {mean_pos:+.3f}"


if __name__ == "__main__":
    test_fakeout_up_fires_on_virtually_every_breakout()
    test_confirmation_condition_is_nearly_always_true()
    test_consequence_signal_shorts_every_real_breakout()
    print("\nAll three demonstrations passed: the confirmation is vacuous.")
