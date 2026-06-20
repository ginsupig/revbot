import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.portfolio_sim import (
    Entry, build_entries, build_closes, simulate_portfolio, run_portfolio_sim, format_sim,
)


def _E(entry_ord, exit_ord, ret, sym, score, price=100.0, atr_pct=0.02):
    return Entry(entry_ord, exit_ord, ret, sym, score, price, atr_pct)


def test_empty_entries_is_flat():
    s = simulate_portfolio([], max_positions=4)
    assert s["n_trades"] == 0 and s["max_drawdown"] == 0.0 and s["equity"] == [1.0]


def test_position_cap_limits_concurrency():
    entries = [_E(0, 10, 0.05, f"S{i}", float(i)) for i in range(6)]
    s = simulate_portfolio(entries, max_positions=2)
    assert s["n_trades"] == 2 and s["n_skipped"] == 4


def test_higher_score_wins_the_slot():
    entries = [_E(0, 5, 0.10, "S_lo", 1.0), _E(0, 5, -0.10, "S_hi", 9.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 1 and s["final_equity"] < 1.0    # took the high-score loser


def test_one_position_per_symbol():
    entries = [_E(0, 20, 0.02, "AAA", 5.0), _E(1, 5, 0.02, "AAA", 5.0)]
    s = simulate_portfolio(entries, max_positions=4)
    assert s["n_trades"] == 1 and s["n_skipped"] == 1


def test_slot_frees_after_exit():
    entries = [_E(0, 5, 0.03, "AAA", 5.0), _E(6, 10, 0.03, "BBB", 5.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 2 and s["n_skipped"] == 0


def test_compounding_and_drawdown_sign():
    entries = [_E(0, 1, 0.20, "AAA", 5.0), _E(2, 3, -0.20, "BBB", 5.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["max_drawdown"] < 0.0 and s["final_equity"] != 1.0


def test_same_day_exit_realizes_immediately():
    entries = [_E(0, 0, 0.05, "AAA", 9.0), _E(0, 5, 0.05, "BBB", 1.0)]
    s = simulate_portfolio(entries, max_positions=1)
    assert s["n_trades"] == 2          # AAA realized same-day, BBB took the freed slot


def test_risk_sizing_scales_with_atr():
    # tighter ATR -> bigger notional -> bigger equity impact for the same ret.
    tight = [_E(0, 1, 0.10, "AAA", 5.0, price=100.0, atr_pct=0.01)]
    wide = [_E(0, 1, 0.10, "AAA", 5.0, price=100.0, atr_pct=0.04)]
    st = simulate_portfolio(tight, max_positions=4, sizing="risk")
    sw = simulate_portfolio(wide, max_positions=4, sizing="risk")
    assert st["final_equity"] > sw["final_equity"] > 1.0


def test_risk_sizing_caps_position_fraction():
    # tiny ATR would imply huge leverage; the cap (max_pos_frac) bounds it.
    e = [_E(0, 1, 0.10, "AAA", 5.0, price=100.0, atr_pct=0.0005)]
    s = simulate_portfolio(e, max_positions=4, sizing="risk",
                           risk_pct=0.005, stop_atr_mult=1.2, max_pos_frac=0.20)
    # notional capped at 20% of equity -> impact <= 0.20 * 0.10 = 0.02
    assert abs(s["final_equity"] - 1.0) <= 0.02 + 1e-9


# --- mark-to-market drawdown -------------------------------------------------

def test_mtm_drawdown_catches_intra_hold_trough():
    # one position held days 0..4; it dips hard mid-hold then recovers to a small
    # gain by exit. Realized DD sees only the +gain; MTM DD must see the trough.
    entries = [_E(0, 4, 0.02, "AAA", 5.0, price=100.0, atr_pct=0.02)]
    closes = {"AAA": {0: 100.0, 1: 80.0, 2: 75.0, 3: 90.0, 4: 102.0}}
    s = simulate_portfolio(entries, closes_by_ord=closes, max_positions=1, sizing="equal")
    assert s["mtm_max_drawdown"] < s["max_drawdown"]      # MTM strictly deeper
    assert s["mtm_max_drawdown"] < -0.10                  # saw the ~-25% mark


def test_mtm_equals_realized_without_marks():
    entries = [_E(0, 2, 0.05, "AAA", 5.0)]
    s = simulate_portfolio(entries, closes_by_ord=None, max_positions=1)
    assert s["mtm_max_drawdown"] == s["max_drawdown"]


# --- build_entries / build_closes from bars ----------------------------------

def _bars(close, dates):
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.full(len(close), 1e6),
                         "date": pd.to_datetime(dates)})


def test_build_entries_carries_price_and_atr():
    dates = pd.date_range("2024-01-01", periods=120)
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 80, 60)])
    entries = build_entries({"X": _bars(close, dates)}, rsi_max=45.0)
    assert entries
    for e in entries:
        assert isinstance(e, Entry)
        assert e.exit_ord >= e.entry_ord and e.entry_price > 0 and e.atr_pct >= 0.0035


def test_build_closes_maps_ord_to_close():
    dates = pd.date_range("2024-01-01", periods=10)
    bars = _bars(np.arange(100, 110.0), dates)
    closes = build_closes({"X": bars})
    assert len(closes["X"]) == 10 and max(closes["X"].values()) == 109.0


def test_run_and_format_end_to_end():
    dates = pd.date_range("2024-01-01", periods=120)
    flush = np.concatenate([np.full(60, 100.0), np.linspace(100, 82, 60)])
    bars = {"X": _bars(flush, dates), "Y": _bars(flush * 1.5, dates)}
    stats = run_portfolio_sim(bars, max_positions=4, sizing="risk")
    assert "mtm_max_drawdown" in stats and "sharpe" in stats
    txt = format_sim(stats, 4)
    assert "mark-to-mkt" in txt and "max_positions" in txt
