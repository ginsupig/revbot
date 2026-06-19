import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.daily_ranker import (
    setup_outcomes, day_ordinals, build_setups, _score, rank_sweep,
    run_daily_ranker,
)
from research.pipeline import PipelineResult


# --- synthetic setups: (entry_ord, exit_ord, ret). The ranker never sees bars in
# these unit tests — it only sees setup tuples — so we can engineer edge cleanly.

def _winner_stream(start=0, step=5, n=40, ret=0.02):
    """A symbol whose every setup pays +ret, one entry every `step` days."""
    return [(start + i * step, start + i * step + 1, ret) for i in range(n)]

def _loser_stream(start=0, step=5, n=40, ret=-0.02):
    return [(start + i * step, start + i * step + 1, ret) for i in range(n)]


def test_score_is_causal_trailing_window():
    setups = _winner_stream(ret=0.03)            # exits at 1, 6, 11, ...
    # on day 12, setups closed in [12-10, 12) = exits {6, 11} (the day-1 exit is
    # outside the 10-day window); the not-yet-closed day-10 entry is invisible.
    s = _score(setups, day=12, window=10, min_trades=1)
    assert abs(s - 0.03) < 1e-9
    # a setup that EXITS on day d itself must NOT count (strictly-before).
    edge = [(0, 12, 0.5)]                          # exits exactly on day 12
    assert _score(edge, day=12, window=100, min_trades=1) is None


def test_score_requires_min_trades():
    setups = [(0, 1, 0.02)]                        # only one resolved trade
    assert _score(setups, day=10, window=100, min_trades=3) is None
    assert _score(setups, day=10, window=100, min_trades=1) == 0.02


def test_rank_concentrates_on_the_winner():
    # Winner pays +2% every time; loser bleeds -2%. The selector with K=1 should
    # trade the winner and inherit its positive book; including the loser unranked
    # would drag the mean toward zero.
    setups = {"WIN": _winner_stream(start=0, ret=0.02),
              "LOSE": _loser_stream(start=2, ret=-0.02)}
    research, _ = rank_sweep(setups, windows=[30], ks=[1], thresholds=[0.0], min_trades=2)
    book = research["W30_K1_t0"]
    assert book.size > 0
    assert book.mean() > 0.015                     # ~ the winner's +2% (cost-free synthetic)


def test_threshold_filters_out_negative_scores():
    setups = {"LOSE": _loser_stream(ret=-0.02)}
    research, _ = rank_sweep(setups, windows=[30], ks=[3], thresholds=[0.0], min_trades=2)
    # every score is negative -> threshold 0.0 admits nothing after warmup
    assert research.get("W30_K3_t0", np.array([])).size == 0


def test_rank_sweep_holdout_split_by_date():
    setups = {"WIN": _winner_stream(start=0, step=5, n=40, ret=0.02)}
    research, holdout = rank_sweep(setups, windows=[20], ks=[1], thresholds=[0.0],
                                   min_trades=1, holdout_cut_ord=100)
    # entries before ord 100 -> research; >= 100 -> holdout. Both non-empty.
    assert research["W20_K1_t0"].size > 0
    assert holdout["W20_K1_t0"].size > 0


# --- bar-level helpers (setup_outcomes / day_ordinals / build_setups) ----------

def _bars(close, dates=None):
    close = np.asarray(close, float)
    df = pd.DataFrame({"open": close, "high": close * 1.01,
                       "low": close * 0.99, "close": close,
                       "volume": np.full(len(close), 1e6)})
    if dates is not None:
        df["date"] = pd.to_datetime(dates)
    return df


def test_setup_outcomes_returns_entry_exit_indices():
    close = np.linspace(100, 130, 90)             # steady uptrend -> a long target hits
    bars = _bars(close)
    atr = (bars["high"] - bars["low"]).rolling(14).mean().bfill().to_numpy()
    out = setup_outcomes(bars["high"].to_numpy(), bars["low"].to_numpy(),
                         bars["close"].to_numpy(), atr, signal_idx=[5])
    assert len(out) == 1
    entry_i, exit_i, ret = out[0]
    assert entry_i == 5 and exit_i >= entry_i      # exit never before entry


def test_day_ordinals_from_date_column_and_fallback():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    bars = _bars([10, 11, 12, 13, 14], dates=dates)
    ords = day_ordinals(bars)
    assert list(np.diff(ords)) == [1, 1, 1, 1]     # consecutive days
    plain = _bars([10, 11, 12])                    # no date -> positional fallback
    assert list(day_ordinals(plain)) == [0, 1, 2]


def test_build_setups_one_per_symbol_per_day():
    # inject a setups_fn that returns two setups on the SAME entry bar; only the
    # first should survive the per-day dedup.
    def _two_same_day(bars):
        return [(0, 1, 0.01), (0, 2, 0.02)]
    bars = _bars([10, 11, 12], dates=pd.date_range("2025-01-01", periods=3))
    out = build_setups({"X": bars}, setups_fn=_two_same_day)
    assert len(out["X"]) == 1                       # deduped to one entry that day


def test_run_daily_ranker_end_to_end_returns_result():
    # winner + loser as injected setups_fn keyed off the symbol via bars length.
    def _fn(bars):
        # encode the stream in the bars: positive if 'WIN' marker (close[0] high)
        if bars["close"].iloc[0] > 500:
            return [(i, i + 1, 0.02) for i in range(0, 60, 3)]
        return [(i, i + 1, -0.02) for i in range(1, 60, 3)]
    dates = pd.date_range("2024-01-01", periods=70)
    win_bars = _bars(np.full(70, 1000.0), dates=dates)
    lose_bars = _bars(np.full(70, 100.0), dates=dates)
    res = run_daily_ranker({"WIN": win_bars, "LOSE": lose_bars},
                           windows=[20, 40], ks=[1, 2], thresholds=[0.0],
                           min_trades=2, holdout_frac=0.25, setups_fn=_fn)
    assert isinstance(res, PipelineResult)
    assert res.n_trials == 4                        # 2 windows x 2 Ks x 1 threshold
    assert res.chosen is not None


def test_run_daily_ranker_empty_when_no_setups():
    def _none(bars):
        return []
    bars = _bars([10, 11, 12], dates=pd.date_range("2025-01-01", periods=3))
    res = run_daily_ranker({"X": bars}, windows=[20], ks=[1], thresholds=[0.0],
                           setups_fn=_none)
    assert res.chosen is None and "empty" in res.verdict.lower()
