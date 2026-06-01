from reversion_bot.churn import (
    round_trips_per_symbol,
    build_churn_report,
    format_churn_report,
)


def _fill(symbol, side, qty, price, t):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price, "time": t}


# --- round_trips_per_symbol -------------------------------------------------

def test_round_trips_long_only_counts_sells():
    fills = [
        _fill("TQQQ", "buy", 10, 50.0, "t1"),
        _fill("TQQQ", "sell", 10, 51.0, "t2"),   # one round trip
        _fill("TQQQ", "buy", 5, 50.0, "t3"),
        _fill("TQQQ", "sell", 5, 49.0, "t4"),    # two
    ]
    assert round_trips_per_symbol(fills) == {"TQQQ": 2}


def test_round_trips_partial_exits_count_each_close():
    fills = [
        _fill("X", "buy", 10, 100.0, "t1"),
        _fill("X", "sell", 4, 101.0, "t2"),      # close
        _fill("X", "sell", 6, 102.0, "t3"),      # close
    ]
    assert round_trips_per_symbol(fills) == {"X": 2}


def test_round_trips_ignores_position_building():
    # two buys then one sell == one exit event, not three
    fills = [
        _fill("X", "buy", 5, 100.0, "t1"),
        _fill("X", "buy", 5, 99.0, "t2"),
        _fill("X", "sell", 10, 101.0, "t3"),
    ]
    assert round_trips_per_symbol(fills) == {"X": 1}


# --- build_churn_report -----------------------------------------------------

def test_churn_flags_below_cost_edge():
    # tiny edge per trip vs a high assumed cost -> churning
    fills = [
        _fill("LOW", "buy", 100, 100.0, "t1"),
        _fill("LOW", "sell", 100, 100.05, "t2"),   # +$5 on ~$10k notional, ~5 bps
    ]
    report = build_churn_report(fills, cost_bps=20.0)
    row = report["rows"][0]
    assert row.round_trips == 1
    assert row.churning is True            # 5 bps edge < 20 bps cost


def test_churn_not_flagged_when_edge_beats_cost():
    fills = [
        _fill("HIGH", "buy", 100, 100.0, "t1"),
        _fill("HIGH", "sell", 100, 101.0, "t2"),   # +$100 on ~$10k, ~100 bps
    ]
    report = build_churn_report(fills, cost_bps=20.0)
    row = report["rows"][0]
    assert row.churning is False
    assert row.pnl_per_trip_bps > 20.0


def test_format_runs_and_marks_churners():
    fills = [
        _fill("LOW", "buy", 100, 100.0, "t1"),
        _fill("LOW", "sell", 100, 100.05, "t2"),
    ]
    text = format_churn_report(build_churn_report(fills, cost_bps=20.0))
    assert "churning" in text
    assert "LOW" in text
