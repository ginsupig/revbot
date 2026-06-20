import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.portfolio_sim import (
    build_entries, simulate_portfolio, run_portfolio_sim, format_sim,
)


# entries: (entry_ord, exit_ord, ret, sym, score)

def test_empty_entries_is_flat():
    s = simulate_portfolio([], max_positions=4)
    assert s["n_trades"] == 0 and s["max_drawdown"] == 0.0 and s["equity"] == [1.0]


def test_position_cap_limits_concurrency():
    # 6 names all entering day 0, all exiting day 10; cap = 2 -> only 2 taken.
    entries = [(0, 10, 0.05, f"S{i}", float(i)) for i in range(6)]
    s = simulate_portfolio(entries, max_positions=2)
    assert s["n_trades"] == 2
    assert s["n_skipped"] == 4


def test_higher_score_wins_the_slot():
    # one slot; two candidates same day; the higher score (S_hi=9) should be taken.
    entries = [(0, 5, 0.10, "S_lo", 1.0), (0, 5, -0.10, "S_hi", 9.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 1
    # the taken trade was the high-score one (ret -0.10) -> equity fell
    assert s["final_equity"] < 1.0


def test_one_position_per_symbol():
    # same symbol signals twice while already held -> the 2nd is skipped.
    entries = [(0, 20, 0.02, "AAA", 5.0), (1, 5, 0.02, "AAA", 5.0)]
    s = simulate_portfolio(entries, max_positions=4)
    assert s["n_trades"] == 1 and s["n_skipped"] == 1


def test_slot_frees_after_exit():
    # AAA held day0->day5; BBB enters day6 -> slot is free, both taken.
    entries = [(0, 5, 0.03, "AAA", 5.0), (6, 10, 0.03, "BBB", 5.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 2 and s["n_skipped"] == 0


def test_compounding_and_drawdown_sign():
    # a winner then a loser; equity should rise then fall -> negative max DD.
    entries = [(0, 1, 0.20, "AAA", 5.0), (2, 3, -0.20, "BBB", 5.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["max_drawdown"] < 0.0
    assert s["final_equity"] != 1.0


def test_same_day_exit_realizes_immediately():
    # exit_ord == entry_ord -> realized at once, never occupies a slot, so a 2nd
    # same-day name still fits even at cap 1.
    entries = [(0, 0, 0.05, "AAA", 9.0), (0, 5, 0.05, "BBB", 1.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 2          # AAA realized same-day, BBB took the freed slot


# --- build_entries from bars -------------------------------------------------

def _bars(close, dates):
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.full(len(close), 1e6),
                         "date": pd.to_datetime(dates)})


def test_build_entries_shapes_and_exit_after_entry():
    dates = pd.date_range("2024-01-01", periods=120)
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 80, 60)])
    entries = build_entries({"X": _bars(close, dates)}, rsi_max=45.0)
    assert entries
    for (entry_ord, exit_ord, ret, sym, score) in entries:
        assert exit_ord >= entry_ord and sym == "X" and np.isfinite(ret)


def test_run_and_format_end_to_end():
    dates = pd.date_range("2024-01-01", periods=120)
    flush = np.concatenate([np.full(60, 100.0), np.linspace(100, 82, 60)])
    bars = {"X": _bars(flush, dates), "Y": _bars(flush * 1.5, dates)}
    stats = run_portfolio_sim(bars, max_positions=4)
    assert "max_drawdown" in stats and "sharpe" in stats
    txt = format_sim(stats, 4)
    assert "MAX DRAWDOWN" in txt and "max_positions" in txt
