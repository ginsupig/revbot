import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.trade_report import (
    build_report,
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


def test_empty():
    report = build_report([])
    assert report["rows"] == []
    assert report["total_notional"] == 0
    assert symbol_weights({}) == {}
