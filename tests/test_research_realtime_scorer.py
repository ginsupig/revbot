import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.realtime_scorer import (
    _zscore, _trailing_ret_by_ord, _trend_on_by_ord, _candidates,
    score_sweep, run_realtime_scorer, PROFILES,
)
from research.pipeline import PipelineResult


def _bars(close, dates, high=None, low=None):
    close = np.asarray(close, float)
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01 if high is None else high,
        "low": close * 0.99 if low is None else low,
        "close": close,
        "volume": np.full(len(close), 1e6),
        "date": pd.to_datetime(dates),
    })


def test_zscore_handles_degenerate():
    assert np.allclose(_zscore([5.0]), [0.0])           # single element
    assert np.allclose(_zscore([3.0, 3.0, 3.0]), [0, 0, 0])  # zero variance
    z = _zscore([1.0, 2.0, 3.0])
    assert abs(z.mean()) < 1e-12 and z[2] > z[0]


def test_trailing_ret_is_causal():
    dates = pd.date_range("2025-01-01", periods=30)
    close = np.linspace(100, 130, 30)
    bars = _bars(close, dates)
    r = _trailing_ret_by_ord(bars, lookback=10)
    # the return keyed at day d uses close[d]/close[d-10]-1 — only past bars
    d20 = int(pd.Timestamp(dates[20]).toordinal())
    assert abs(r[d20] - (close[20] / close[10] - 1.0)) < 1e-12
    # the first 10 days have no trailing window -> absent (no peeking backward off-array)
    d5 = int(pd.Timestamp(dates[5]).toordinal())
    assert d5 not in r


def test_regime_gate_tracks_sma():
    dates = pd.date_range("2025-01-01", periods=80)
    close = np.concatenate([np.linspace(100, 60, 40), np.linspace(60, 140, 40)])
    on = _trend_on_by_ord(_bars(close, dates), win=20)
    # falling first half -> below SMA (risk-off); rising second half -> above
    early = int(pd.Timestamp(dates[35]).toordinal())
    late = int(pd.Timestamp(dates[75]).toordinal())
    assert on[early] is False and on[late] is True


def test_candidates_fire_on_oversold_below_band():
    dates = pd.date_range("2025-01-01", periods=80)
    # steady, then a sharp flush that drives price below the lower band + RSI down
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 80, 20)])
    rows = _candidates(_bars(close, dates), rsi_max=45.0)
    assert len(rows) > 0
    # each row: (day_ord, entry_i, setup_depth>0, movement>0, ret)
    for (d, i, setup, movement, ret) in rows:
        assert setup > 0 and movement > 0


def _panel():
    """Two names on the same calendar: STRONG has high relative strength /
    momentum, WEAK is flat. Both periodically dip below band to create candidates."""
    dates = pd.date_range("2024-01-01", periods=160)
    rng = np.random.default_rng(0)
    def series(drift):
        c = [100.0]
        for k in range(1, 160):
            shock = 0.85 if k % 20 == 0 else 1.0          # periodic flush -> candidate
            c.append(c[-1] * (1 + drift + rng.normal(0, 0.005)) * shock)
        return np.array(c)
    strong = _bars(series(0.004), dates)
    weak = _bars(series(0.0), dates)
    spy = _bars(np.linspace(100, 140, 160), dates)         # SPY uptrend -> regime on
    return {"STRONG": strong, "WEAK": weak}, spy, dates


def test_score_sweep_prefers_high_rs_name():
    symbol_bars, spy, _ = _panel()
    research, _ = score_sweep(
        symbol_bars, spy, profiles={"momentum": PROFILES["momentum"]},
        lookbacks=[20], ks=[1], regime_gate=True, holdout_cut_ord=None)
    book = research.get("momentum_lb20_K1", np.array([]))
    assert book.size > 0          # the momentum profile with K=1 traded a name each eligible day


def test_score_sweep_regime_gate_blanks_risk_off():
    symbol_bars, _spy, dates = _panel()
    spy_down = _bars(np.linspace(140, 80, 160), dates)     # SPY downtrend -> gate off
    research, _ = score_sweep(symbol_bars, spy_down, profiles={"balanced": PROFILES["balanced"]},
                              lookbacks=[20], ks=[2], regime_gate=True)
    # everything risk-off -> no longs taken
    assert research.get("balanced_lb20_K2", np.array([])).size == 0


def test_run_realtime_scorer_end_to_end():
    symbol_bars, spy, _ = _panel()
    res = run_realtime_scorer(symbol_bars, spy, lookbacks=[20, 40], ks=[1, 2],
                              regime_gate=True, holdout_frac=0.25)
    assert isinstance(res, PipelineResult)
    # 5 profiles x 2 lookbacks x 2 Ks
    assert res.n_trials == len(PROFILES) * 2 * 2
    assert res.chosen is not None


def test_run_realtime_scorer_empty_without_candidates():
    dates = pd.date_range("2025-01-01", periods=40)
    flat = _bars(np.full(40, 100.0), dates)                # never below band -> no candidates
    spy = _bars(np.linspace(100, 120, 40), dates)
    res = run_realtime_scorer({"X": flat}, spy)
    assert res.chosen is None and "empty" in res.verdict.lower()


def test_sector_factor_injected_optional():
    symbol_bars, spy, dates = _panel()
    sector_of = {"STRONG": "XLK", "WEAK": "XLK"}
    sector_bars = {"XLK": _bars(np.linspace(100, 150, 160), dates)}
    research, _ = score_sweep(symbol_bars, spy, sector_of=sector_of, sector_bars=sector_bars,
                              profiles={"sector_rs": PROFILES["sector_rs"]},
                              lookbacks=[20], ks=[1], regime_gate=True)
    assert research.get("sector_rs_lb20_K1", np.array([])).size > 0
