"""Proof + guard: the 2-row decision window is bit-identical to the full slice,
because ReversionEngine.get_decision reads only iloc[-1] and iloc[-2]."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig


def make_bars(n=1200, seed=11):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, n)
    close = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close,
        "volume": rng.integers(50_000, 150_000, n).astype(float),
    }, index=idx)


def test_two_row_window_identical_to_full_slice():
    cfg = ReversionConfig(min_history=2, band_length=20, min_price=0.0, min_dollar_volume=0.0)
    engine = ReversionEngine(cfg)
    enriched = engine.calculate_indicators(make_bars())

    t0 = time.perf_counter()
    full = [engine.get_decision(enriched.iloc[: i + 1]).signal for i in range(1, len(enriched))]
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    win = [engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal for i in range(1, len(enriched))]
    t_win = time.perf_counter() - t0

    assert full == win, "2-row window diverged from full-slice decisions"
    print(f"\nidentical over {len(full)} bars | full-slice {t_full:.2f}s vs 2-row {t_win:.2f}s "
          f"({t_full / max(t_win, 1e-9):.1f}x faster)")
