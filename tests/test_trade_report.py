import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.trade_report import (
    build_daily_timeline,
    build_report,
    build_scoreboard,
    build_trades,
    format_csv,
    format_daily_csv,
    format_daily_timeline,
    format_scoreboard,
    format_trades,
    realized_pnl_by_day,
    realized_pnl_fifo,
    round_trips,
    symbol_weights,
    traded_notional,
)


def _fill(symbol, side, qty, price, time):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price, "time": time}


def test_long_roundtrip_pnl():
    fills = [
        _fill("AAPL", "buy", 10, 100.0, "2026-05-01T10:00:00Z"),
        _fill("AAPL", "sell", 10, 110.0, "2026-05-02T10:00:00Z"),
    ]
    pnl = realized_pnl_fifo(fills)
    assert pnl["AAPL"] == 100.0  # (110-100) * 10


def test_short_roundtrip_pnl():
    fills = [
        _fill("TSLA", "sell", 5, 200.0, "2026-05-01T10:00:00Z"),
        _fill("TSLA", "buy", 5, 180.0, "2026-05-02T10:00:00Z"),
    ]
    pnl = realized_pnl_fifo(fills)
    assert pnl["TSLA"] == 100.0  # (200-180) * 5


def test_fifo_partial_and_multiple_lots():
    fills = [
        _fill("X", "buy", 10, 100.0, "2026-05-01T10:00:00Z"),
        _fill("X", "buy", 10, 120.0, "2026-05-01T11:00:00Z"),
        _fill("X", "sell", 15, 130.0, "2026-05-02T10:00:00Z"),
    ]
    # First 10 @100 -> +300; next 5 @120 -> +50. Remaining 5 @120 stays open.
    pnl = realized_pnl_fifo(fills)
    assert pnl["X"] == 350.0


def test_open_position_no_realized_pnl():
    fills = [_fill("Y", "buy", 10, 50.0, "2026-05-01T10:00:00Z")]
    assert realized_pnl_fifo(fills).get("Y", 0.0) == 0.0


def test_traded_notional_and_weights():
    fills = [
        _fill("A", "buy", 10, 100.0, "t1"),   # 1000
        _fill("A", "sell", 10, 100.0, "t2"),  # 1000 -> A total 2000
        _fill("B", "buy", 10, 100.0, "t3"),   # 1000 -> B total 1000
    ]
    notional = traded_notional(fills)
    assert notional["A"] == 2000.0
    assert notional["B"] == 1000.0

    weights = symbol_weights(notional)
    assert abs(weights["A"] - 2 / 3) < 1e-9
    assert abs(weights["B"] - 1 / 3) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_build_report_sorted_and_totals():
    fills = [
        _fill("A", "buy", 10, 100.0, "t1"),
        _fill("A", "sell", 10, 110.0, "t2"),
        _fill("B", "buy", 1, 50.0, "t3"),
    ]
    report = build_report(fills)
    assert report["symbols"] == 2
    assert report["total_fills"] == 3
    assert report["rows"][0].symbol == "A"  # higher notional first
    assert report["total_realized_pnl"] == 100.0


def test_efficiency_is_pnl_over_notional():
    fills = [
        _fill("A", "buy", 10, 100.0, "t1"),   # notional 1000
        _fill("A", "sell", 10, 110.0, "t2"),  # notional 1100 -> 2100 total, pnl +100
    ]
    row = build_report(fills)["rows"][0]
    assert abs(row.efficiency - (100.0 / 2100.0)) < 1e-12


def test_sort_by_pnl_and_efficiency():
    fills = [
        # Big notional, small PnL.
        _fill("BIG", "buy", 100, 100.0, "t1"),
        _fill("BIG", "sell", 100, 100.1, "t2"),   # pnl +10, notional ~20010
        # Small notional, bigger PnL and far better efficiency.
        _fill("SML", "buy", 1, 100.0, "t3"),
        _fill("SML", "sell", 1, 150.0, "t4"),     # pnl +50, notional 250
    ]
    by_notional = build_report(fills, sort_by="notional")["rows"]
    assert by_notional[0].symbol == "BIG"

    by_pnl = build_report(fills, sort_by="pnl")["rows"]
    assert by_pnl[0].symbol == "SML"

    by_eff = build_report(fills, sort_by="efficiency")["rows"]
    assert by_eff[0].symbol == "SML"


def test_format_csv_has_header_and_rows():
    fills = [
        _fill("A", "buy", 10, 100.0, "t1"),
        _fill("A", "sell", 10, 110.0, "t2"),
    ]
    csv_text = format_csv(build_report(fills))
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("symbol,weight,traded_notional,realized_pnl,efficiency")
    assert lines[1].startswith("A,")


def test_empty():
    report = build_report([])
    assert report["rows"] == []
    assert report["total_notional"] == 0
    assert symbol_weights({}) == {}
    assert format_csv(report).strip().startswith("symbol,")


# --- Rolling scoreboard ---------------------------------------------------

def test_realized_pnl_by_day_partitions_by_date():
    fills = [
        # Day 1: AAPL +100
        _fill("AAPL", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("AAPL", "sell", 10, 110.0, "2026-06-01T15:00:00Z"),
        # Day 2: AAPL -50
        _fill("AAPL", "buy", 10, 110.0, "2026-06-02T14:00:00Z"),
        _fill("AAPL", "sell", 10, 105.0, "2026-06-02T15:00:00Z"),
    ]
    by_day = realized_pnl_by_day(fills)
    assert by_day["AAPL"]["2026-06-01"] == 100.0
    assert by_day["AAPL"]["2026-06-02"] == -50.0


def test_realized_pnl_by_day_attributes_overnight_to_closing_day():
    # Bought day 1, sold day 2 -> PnL booked on the closing day, not dropped.
    fills = [
        _fill("X", "buy", 10, 100.0, "2026-06-01T15:00:00Z"),
        _fill("X", "sell", 10, 112.0, "2026-06-02T14:00:00Z"),
    ]
    by_day = realized_pnl_by_day(fills)
    assert "2026-06-01" not in by_day.get("X", {})        # open only -> no realized
    assert abs(by_day["X"]["2026-06-02"] - 120.0) < 1e-9  # (112-100)*10 on close day


def test_daily_breakdown_reconciles_with_global_fifo():
    # The sum of per-day realized must equal the cross-day FIFO total — this is
    # the invariant the old partition-by-day method violated for overnight holds.
    fills = [
        _fill("X", "buy", 10, 100.0, "2026-06-01T15:00:00Z"),
        _fill("X", "sell", 5, 108.0, "2026-06-01T15:30:00Z"),   # same-day partial
        _fill("X", "sell", 5, 115.0, "2026-06-03T14:00:00Z"),   # rest sold 2 days later
        _fill("Y", "sell", 4, 50.0, "2026-06-02T14:00:00Z"),    # short
        _fill("Y", "buy", 4, 44.0, "2026-06-04T14:00:00Z"),     # covered later
    ]
    by_day = realized_pnl_by_day(fills)
    per_day_total = sum(p for days in by_day.values() for p in days.values())
    global_total = sum(realized_pnl_fifo(fills).values())
    assert abs(per_day_total - global_total) < 1e-9


def test_scoreboard_cumulative_equals_sum_of_days():
    fills = [
        _fill("X", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("X", "sell", 10, 110.0, "2026-06-01T15:00:00Z"),   # +100
        _fill("X", "buy", 10, 100.0, "2026-06-02T14:00:00Z"),
        _fill("X", "sell", 10, 97.0, "2026-06-02T15:00:00Z"),    # -30
    ]
    report = build_scoreboard(fills)
    row = report["rows"][0]
    assert row.symbol == "X"
    assert abs(row.realized_pnl - 70.0) < 1e-9
    assert row.days_traded == 2
    assert row.days_positive == 1
    assert row.days_negative == 1
    assert row.best_day == 100.0
    assert row.worst_day == -30.0
    assert row.last_day == -30.0          # most recent day
    assert report["days_in_window"] == 2


def test_scoreboard_flags_consistent_bleeder_not_noisy_winner():
    fills = []
    # WIN: net positive across 2 days -> never flagged.
    fills += [
        _fill("WIN", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("WIN", "sell", 10, 130.0, "2026-06-01T15:00:00Z"),   # +300
        _fill("WIN", "buy", 10, 100.0, "2026-06-02T14:00:00Z"),
        _fill("WIN", "sell", 10, 99.0, "2026-06-02T15:00:00Z"),    # -10
    ]
    # BLEED: small but negative every day, efficiently bad -> flagged.
    fills += [
        _fill("BLEED", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("BLEED", "sell", 10, 99.0, "2026-06-01T15:00:00Z"),  # -10
        _fill("BLEED", "buy", 10, 100.0, "2026-06-02T14:00:00Z"),
        _fill("BLEED", "sell", 10, 98.0, "2026-06-02T15:00:00Z"),  # -20
    ]
    report = build_scoreboard(fills)
    flagged = {r.symbol for r in report["rows"] if r.probation}
    assert "BLEED" in flagged
    assert "WIN" not in flagged
    # Worst first ordering.
    assert report["rows"][0].symbol == "BLEED"


def test_scoreboard_note_when_insufficient_days():
    fills = [
        _fill("Q", "buy", 10, 100.0, "2026-06-03T14:00:00Z"),
        _fill("Q", "sell", 10, 99.0, "2026-06-03T15:00:00Z"),
    ]
    text = format_scoreboard(build_scoreboard(fills), days=5, min_days=5)
    assert "only 1 trading day" in text
    assert "CUT?" in text  # legend present


def test_scoreboard_empty():
    report = build_scoreboard([])
    assert report["rows"] == []
    assert report["days_in_window"] == 0
    assert report["total_realized_pnl"] == 0


# --- Daily timeline -------------------------------------------------------

def test_daily_timeline_groups_by_day_with_tickers():
    fills = [
        # Day 1: A +100, B -20
        _fill("A", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("A", "sell", 10, 110.0, "2026-06-01T15:00:00Z"),
        _fill("B", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("B", "sell", 10, 98.0, "2026-06-01T15:00:00Z"),
        # Day 2: A +50
        _fill("A", "buy", 10, 100.0, "2026-06-02T14:00:00Z"),
        _fill("A", "sell", 10, 105.0, "2026-06-02T15:00:00Z"),
    ]
    report = build_daily_timeline(fills)  # newest first by default
    assert report["days_in_window"] == 2
    assert report["rows"][0]["date"] == "2026-06-02"
    assert report["rows"][1]["date"] == "2026-06-01"

    day1 = report["rows"][1]
    assert abs(day1["realized_pnl"] - 80.0) < 1e-9      # +100 - 20
    assert day1["fills"] == 4
    # Symbols sorted by contribution: A (+100) before B (-20).
    assert [s for s, _ in day1["symbols"]] == ["A", "B"]
    assert abs(report["total_realized_pnl"] - 130.0) < 1e-9


def test_daily_timeline_csv_has_total_rows():
    fills = [
        _fill("A", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("A", "sell", 10, 110.0, "2026-06-01T15:00:00Z"),
    ]
    csv_text = format_daily_csv(build_daily_timeline(fills))
    lines = csv_text.strip().splitlines()
    assert lines[0] == "date,symbol,realized_pnl"
    assert any(line.startswith("2026-06-01,A,") for line in lines)
    assert any(line.startswith("2026-06-01,TOTAL,") for line in lines)


def test_daily_timeline_empty():
    report = build_daily_timeline([])
    assert report["rows"] == []
    assert report["days_in_window"] == 0
    assert "Daily PnL timeline" in format_daily_timeline(report, days=60)


def test_round_trips_long_and_short_pnl_and_dir():
    fills = [
        _fill("L", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("L", "sell", 10, 110.0, "2026-06-01T15:00:00Z"),
        _fill("S", "sell", 5, 200.0, "2026-06-01T14:00:00Z"),
        _fill("S", "buy", 5, 180.0, "2026-06-01T14:30:00Z"),
    ]
    trips = round_trips(fills)
    assert len(trips) == 2
    by_sym = {t["symbol"]: t for t in trips}
    assert by_sym["L"]["direction"] == "long"
    assert by_sym["L"]["pnl"] == 100.0
    assert by_sym["L"]["hold_seconds"] == 3600.0
    assert by_sym["S"]["direction"] == "short"
    assert by_sym["S"]["pnl"] == 100.0
    assert abs(by_sym["S"]["return_pct"] - (100.0 / 1000.0)) < 1e-9


def test_round_trips_split_lots_emit_per_lot_records():
    # One closing fill consumes two entry lots -> two round-trip records.
    fills = [
        _fill("X", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("X", "buy", 10, 120.0, "2026-06-01T14:05:00Z"),
        _fill("X", "sell", 15, 130.0, "2026-06-01T15:00:00Z"),
    ]
    trips = round_trips(fills)
    assert len(trips) == 2
    assert sum(t["pnl"] for t in trips) == 350.0  # 10*(130-100) + 5*(130-120)


def test_round_trips_sum_to_fifo_total():
    fills = [
        _fill("A", "buy", 10, 100.0, "t1"),
        _fill("A", "sell", 4, 90.0, "t2"),
        _fill("A", "sell", 6, 130.0, "t3"),
        _fill("B", "sell", 5, 50.0, "t4"),
        _fill("B", "buy", 5, 55.0, "t5"),
    ]
    trips = round_trips(fills)
    fifo = realized_pnl_fifo(fills)
    assert abs(sum(t["pnl"] for t in trips) - sum(fifo.values())) < 1e-9


def test_open_position_no_round_trip():
    fills = [_fill("Y", "buy", 10, 50.0, "t1")]
    assert round_trips(fills) == []


def test_build_trades_filter_sort_and_stats():
    fills = [
        _fill("WDC", "buy", 10, 100.0, "2026-06-01T14:00:00Z"),
        _fill("WDC", "sell", 10, 95.0, "2026-06-01T15:00:00Z"),   # -50 (worst)
        _fill("WDC", "buy", 10, 100.0, "2026-06-01T16:00:00Z"),
        _fill("WDC", "sell", 10, 102.0, "2026-06-01T17:00:00Z"),  # +20
        _fill("AAPL", "buy", 1, 200.0, "2026-06-01T14:00:00Z"),
        _fill("AAPL", "sell", 1, 210.0, "2026-06-01T15:00:00Z"),  # filtered out
    ]
    report = build_trades(fills, symbol="wdc")  # case-insensitive
    assert report["symbol"] == "WDC"
    assert report["round_trips"] == 2
    assert report["wins"] == 1 and report["losses"] == 1
    assert report["trips"][0]["pnl"] == -50.0   # worst first
    assert abs(report["total_pnl"] - (-30.0)) < 1e-9
    assert abs(report["profit_factor"] - (20.0 / 50.0)) < 1e-9
    assert "WDC" in format_trades(report, days=1)


def test_build_trades_all_symbols_when_unfiltered():
    fills = [
        _fill("A", "buy", 1, 10.0, "t1"),
        _fill("A", "sell", 1, 12.0, "t2"),
        _fill("B", "buy", 1, 10.0, "t3"),
        _fill("B", "sell", 1, 9.0, "t4"),
    ]
    report = build_trades(fills)
    assert report["symbol"] is None
    assert report["round_trips"] == 2
    assert report["profit_factor"] == 2.0  # +2 win / 1 loss


def test_build_trades_empty():
    report = build_trades([], symbol="WDC")
    assert report["round_trips"] == 0
    assert report["win_rate"] == 0.0
    assert "WDC" in format_trades(report, days=1)
