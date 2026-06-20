import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.portfolio_sim import (
    Entry, build_entries, build_closes, simulate_portfolio, run_portfolio_sim, format_sim,
    compare_sizing, format_sizing_comparison, SIZING_GRID, grid_sweep, format_grid,
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


# --- sizing frontier (thread 2: recover return ATR-risk sizing leaves on the table)

def test_compare_sizing_returns_a_row_per_grid_config():
    entries = [_E(0, 1, 0.05, "AAA", 5.0)]
    rows = compare_sizing(entries, None, max_positions=4)
    assert len(rows) == len(SIZING_GRID)
    assert rows[0][0] == "equal" and rows[1][0] == "risk@0.5%"


def test_more_risk_budget_scales_high_vol_winner_until_cap():
    # a HIGH-vol winner (atr_pct=0.05) is underweighted by ATR-risk sizing; raising
    # risk_pct must scale its notional up -> higher return, until the position cap.
    hv = [_E(0, 1, 0.10, "AAA", 5.0, price=100.0, atr_pct=0.05)]
    closes = {"AAA": {0: 100.0, 1: 110.0}}
    rows = compare_sizing(hv, closes, max_positions=4)
    by_label = dict(rows)
    r_small = by_label["risk@0.5%"]["total_return"]
    r_big = by_label["risk@2%"]["total_return"]
    assert r_big > r_small > 0.0                     # more budget -> more of the winner
    # equal weight (1/4 = 25%) is the ceiling here; ATR-risk at 0.5% is well under it
    assert by_label["equal"]["total_return"] >= r_big


def test_low_vol_name_is_capped_so_risk_budget_barely_moves_it():
    # a LOW-vol name (atr_pct=0.005) is already pinned at max_pos_frac, so raising
    # risk_pct should NOT change its return (the cap binds at every level).
    lv = [_E(0, 1, 0.08, "AAA", 5.0, price=100.0, atr_pct=0.005)]
    rows = dict(compare_sizing(lv, None, max_positions=4, max_pos_frac=0.20))
    assert rows["risk@0.5%"]["total_return"] == rows["risk@2%"]["total_return"]


def test_format_sizing_comparison_renders_frontier():
    entries = [_E(0, 1, 0.05, "AAA", 5.0, atr_pct=0.03)]
    txt = format_sizing_comparison(compare_sizing(entries, None, 4), 4)
    assert "sizing frontier" in txt and "RET/DD" in txt and "equal" in txt


# --- 2-D cap x risk-budget grid ----------------------------------------------

def test_grid_sweep_covers_the_cartesian_product():
    entries = [_E(0, 1, 0.04, f"S{i}", float(i), atr_pct=0.03) for i in range(8)]
    caps, rps = [2, 4], [0.005, 0.01, 0.02]
    grid = grid_sweep(entries, None, caps, rps)
    assert set(grid) == {(c, r) for c in caps for r in rps}
    for s in grid.values():
        assert "mtm_max_drawdown" in s and s["sizing"] == "risk"


def test_grid_more_slots_take_more_trades():
    # 8 names entering day 0; cap 4 takes more than cap 2 (occupancy, not sizing).
    entries = [_E(0, 5, 0.02, f"S{i}", float(i)) for i in range(8)]
    grid = grid_sweep(entries, None, [2, 4], [0.01])
    assert grid[(4, 0.01)]["n_trades"] > grid[(2, 0.01)]["n_trades"]


def test_format_grid_renders_three_matrices():
    entries = [_E(0, 1, 0.04, f"S{i}", float(i), atr_pct=0.03) for i in range(6)]
    caps, rps = [2, 4], [0.005, 0.01, 0.02]
    txt = format_grid(grid_sweep(entries, None, caps, rps), caps, rps)
    assert "RET/DD" in txt and "RETURN %" in txt and "MTM DRAWDOWN %" in txt
    assert "cap\\risk" in txt


# --- live governor caps (exposure + heat) ------------------------------------

def test_heat_cap_throttles_high_risk_pct():
    # high-vol names (atr_pct=0.08 -> stop 9.6%) so risk@1.5% does NOT hit the 20%
    # position cap: realized heat ~1.5% apiece. At the 2% heat cap only 1 fits
    # (2nd would be 3% > 2%); without the cap, all 4 fit.
    entries = [_E(0, 10, 0.03, f"S{i}", float(i), price=100.0, atr_pct=0.08) for i in range(4)]
    capped = simulate_portfolio(entries, None, max_positions=4, sizing="risk",
                                risk_pct=0.015, max_portfolio_heat=0.02)
    uncapped = simulate_portfolio(entries, None, max_positions=4, sizing="risk",
                                  risk_pct=0.015, max_portfolio_heat=None)
    assert capped["n_trades"] == 1
    assert uncapped["n_trades"] == 4


def test_heat_cap_allows_more_at_lower_risk_pct():
    # at risk@0.5% (heat ~0.5% each), the 2% cap fits 4 -> validated baseline.
    entries = [_E(0, 10, 0.03, f"S{i}", float(i), price=100.0, atr_pct=0.02) for i in range(4)]
    s = simulate_portfolio(entries, None, max_positions=4, sizing="risk",
                           risk_pct=0.005, max_portfolio_heat=0.02)
    assert s["n_trades"] == 4


def test_total_exposure_cap_blocks_leverage():
    # equal sizing, 8 names @ 1/8 = 12.5% each; a 0.50 exposure cap fits 4 (50%).
    entries = [_E(0, 10, 0.02, f"S{i}", float(i)) for i in range(8)]
    s = simulate_portfolio(entries, None, max_positions=8, sizing="equal",
                           max_total_exposure=0.50)
    assert s["n_trades"] == 4


def test_caps_off_by_default_preserve_behavior():
    entries = [_E(0, 10, 0.02, f"S{i}", float(i)) for i in range(6)]
    s = simulate_portfolio(entries, None, max_positions=6, sizing="equal")
    assert s["n_trades"] == 6          # no caps passed -> all six open


# --- factor-ranked daily experiment ------------------------------------------

def _panel(seed_base=0):
    """Multi-symbol daily panel with periodic flushes (creates oversold candidates)
    + an SPY and one sector ETF for the rs/sector factors."""
    dates = pd.date_range("2024-01-01", periods=200)
    def series(seed, drift):
        r = np.random.default_rng(seed); c = [100.0]
        for k in range(1, 200):
            shock = 0.90 if k % 15 == 0 else 1.0
            c.append(c[-1] * (1 + drift + r.normal(0, 0.008)) * shock)
        return np.asarray(c)
    sb = {f"N{i}": _bars(series(seed_base + i, 0.0003), dates) for i in range(6)}
    spy = _bars(np.linspace(100, 130, 200), dates)
    sector = {"XLK": _bars(np.linspace(100, 140, 200), dates)}
    return sb, spy, sector


def test_build_factor_entries_shapes_and_hold_effect():
    from research.portfolio_sim import build_factor_entries, EXPERIMENT_PROFILES
    sb, spy, sector = _panel()
    sof = {f"N{i}": "XLK" for i in range(6)}
    e1 = build_factor_entries(sb, spy, sof, sector, EXPERIMENT_PROFILES["exhaustion"], hold=1)
    e5 = build_factor_entries(sb, spy, sof, sector, EXPERIMENT_PROFILES["exhaustion"], hold=5)
    assert e1 and e5
    for e in e1:
        assert e.exit_ord >= e.entry_ord and e.entry_price > 0 and np.isfinite(e.score)
    # a longer hold lets the bracket run further -> exits no earlier on average
    span1 = np.mean([e.exit_ord - e.entry_ord for e in e1])
    span5 = np.mean([e.exit_ord - e.entry_ord for e in e5])
    assert span5 >= span1


def test_factor_sweep_covers_grid_and_sorts_by_ret_dd():
    from research.portfolio_sim import factor_sweep, _ret_dd, EXPERIMENT_PROFILES
    sb, spy, sector = _panel()
    sof = {f"N{i}": "XLK" for i in range(6)}
    rows = factor_sweep(sb, spy, sof, sector,
                        profiles={"exhaustion": EXPERIMENT_PROFILES["exhaustion"]},
                        ks=[3, 5], holds=[1, 3], risks=[0.0075, 0.01], heat=0.02)
    assert len(rows) == 1 * 2 * 2 * 2          # profile x hold x K x risk
    rds = [_ret_dd(s) for _, s in rows]
    assert rds == sorted(rds, reverse=True)    # best RET/DD first
    assert all("h" in label and "_K" in label and "_r" in label for label, _ in rows)


def test_format_factor_sweep_renders():
    from research.portfolio_sim import factor_sweep, format_factor_sweep, EXPERIMENT_PROFILES
    sb, spy, sector = _panel()
    sof = {f"N{i}": "XLK" for i in range(6)}
    rows = factor_sweep(sb, spy, sof, sector,
                        profiles={"sector_rs": EXPERIMENT_PROFILES["sector_rs"]},
                        ks=[3], holds=[2], risks=[0.01], heat=0.02)
    txt = format_factor_sweep(rows)
    assert "factor-ranked daily experiment" in txt and "RET/DD" in txt
