"""Walkforward PnL accounting + tuning protocol (audit H8/H9).

H8: multi-bar trades used to book mark-to-market returns on every held bar AND
the full entry->exit return again on the exit bar — double-counting that
inflated PF/Sharpe/DD in both directions, and made analytics._trade_count
count every held bar as a "trade". A trade now books exactly once, at exit.

H9: the walkforward used to tune parameters on the same TimeSeriesSplit test
folds it then reported as "OOS" (the reported numbers were the grid maximum
over the evaluation set). It now tunes each fold on its TRAIN window only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reversion_bot.walkforward as wf
from reversion_bot.autotune import AutoTuner


def _random_walk_df(n=500, seed=42):
    np.random.seed(seed)
    close = np.cumsum(np.random.randn(n)) + 100
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.random.randint(1000, 2000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


# --- H8: a trade books exactly once, at exit ---------------------------------

def test_held_bars_book_zero_return():
    res = wf.mean_reversion_strategy(_random_walk_df(), use_trend_filter=False,
                                     rsi_max=55.0)
    # Fixture sanity: it must actually contain multi-bar trades or the
    # assertions below are vacuous.
    entries = int((res["signal"] == 1).sum())
    held_bars = int(((res["position"] == 1) & (res["signal"] == 0)).sum())
    assert entries >= 2 and held_bars >= 5

    # Entry and held bars (position==1) must book NO return — the old MTM line
    # here is exactly the double-count.
    assert res.loc[res["position"] == 1, "strategy_return"].eq(0).all()

    # Every nonzero return is an exit bar (position flattened on that bar).
    nonzero = res["strategy_return"] != 0
    assert res.loc[nonzero, "position"].eq(0).all()

    # One booking per completed trade: nonzero returns <= entries.
    assert int(nonzero.sum()) <= entries


def test_booked_return_is_the_round_trip():
    cost = wf.SLIPPAGE_PCT
    res = wf.mean_reversion_strategy(_random_walk_df(), use_trend_filter=False,
                                     rsi_max=55.0)
    entry_price = None
    checked = 0
    for i in range(len(res)):
        row = res.iloc[i]
        if row["signal"] == 1:
            entry_price = float(row["close"])
        sr = float(row["strategy_return"])
        if sr != 0:
            assert entry_price is not None
            # Booked return implies exit at (sr + 1 + cost) * entry; that price
            # must be attainable within the exit bar (stop/target can sit just
            # outside the bar range when price gapped through them).
            implied_exit = (sr + 1.0 + cost) * entry_price
            assert row["low"] - 0.5 <= implied_exit <= row["high"] + 0.5
            checked += 1
    assert checked >= 2


def test_short_held_bars_book_zero_return():
    df = _random_walk_df(seed=7)
    res = wf.short_exhaustion_strategy(df, rvol_max=2.0, require_divergence=False)
    in_short = res["position"] == -1
    if int(in_short.sum()) == 0:
        pytest.skip("fixture produced no short trades")
    assert res.loc[in_short, "strategy_return"].eq(0).all()
    nonzero = res["strategy_return"] != 0
    assert res.loc[nonzero, "position"].eq(0).all()


# --- H9: per-fold tuning must only see the train window ----------------------

def test_walkforward_tunes_only_on_train_windows(monkeypatch):
    from sklearn.model_selection import TimeSeriesSplit

    tuner_windows = []

    class SpyTuner:
        def __init__(self, strategy_func, data, param_grid):
            tuner_windows.append(data)

        def tune(self, score_func, n_splits=5, verbose=True):
            return {}, 0.0

    monkeypatch.setattr(wf, "AutoTuner", SpyTuner)
    df = _random_walk_df(n=400)
    n_splits = 3
    results, oos_metrics, best = wf.run_walkforward_backtest(df, n_splits=n_splits)

    assert len(results) == n_splits
    assert len(oos_metrics) == n_splits
    # One tuner per fold + one final deployment tuner on the full frame.
    assert len(tuner_windows) == n_splits + 1

    folds = list(TimeSeriesSplit(n_splits=n_splits).split(df))
    for (train_idx, test_idx), window in zip(folds, tuner_windows[:n_splits]):
        assert len(window) == len(train_idx)
        # Everything the tuner saw ends strictly before the fold's test window.
        assert window.index.max() < df.index[test_idx].min()
    assert len(tuner_windows[-1]) == len(df)


def test_autotuner_clamps_splits_and_respects_verbose(capsys):
    def strategy(df, k=1):
        out = df.copy()
        out["pnl"] = out["close"].pct_change().fillna(0) * k
        return out

    def score(res):
        return float(res["pnl"].sum())

    df = pd.DataFrame({"close": np.linspace(100, 110, 8)})
    tuner = AutoTuner(strategy, df, {"k": [1, 2]})
    best, _ = tuner.tune(score, n_splits=50, verbose=False)  # 50 >> len(df)
    assert best == {"k": 2}
    assert capsys.readouterr().out == ""
