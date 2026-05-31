import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.trade_report import (
    build_report,
    format_csv,
    realized_pnl_fifo,
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
