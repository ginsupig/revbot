"""Tests for the per-closing-fill realized-PnL event stream that feeds
outcome-based threshold adaptation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reversion_bot.trade_report import realized_pnl_events, realized_pnl_fifo


def fill(symbol, side, qty, price, time):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price, "time": time}


def test_long_roundtrip_emits_one_event_on_the_sell():
    fills = [
        fill("AAA", "buy", 10, 100.0, "2026-06-01T14:30:00Z"),
        fill("AAA", "sell", 10, 102.0, "2026-06-01T15:30:00Z"),
    ]
    evs = realized_pnl_events(fills)
    assert len(evs) == 1
    assert evs[0]["symbol"] == "AAA"
    assert evs[0]["time"] == "2026-06-01T15:30:00Z"   # the closing fill's time
    assert abs(evs[0]["realized_pnl"] - 20.0) < 1e-9  # (102-100)*10


def test_short_roundtrip_pnl_sign():
    fills = [
        fill("BBB", "sell", 5, 50.0, "2026-06-01T14:30:00Z"),   # open short
        fill("BBB", "buy", 5, 48.0, "2026-06-01T15:00:00Z"),    # cover lower -> profit
    ]
    evs = realized_pnl_events(fills)
    assert len(evs) == 1
    assert abs(evs[0]["realized_pnl"] - 10.0) < 1e-9   # (50-48)*5


def test_open_only_fill_emits_no_event():
    evs = realized_pnl_events([fill("CCC", "buy", 3, 10.0, "2026-06-01T14:30:00Z")])
    assert evs == []


def test_multiple_roundtrips_count_separately():
    fills = [
        fill("DDD", "buy", 1, 100.0, "2026-06-01T14:30:00Z"),
        fill("DDD", "sell", 1, 101.0, "2026-06-01T14:40:00Z"),   # +1 win
        fill("DDD", "buy", 1, 100.0, "2026-06-01T15:00:00Z"),
        fill("DDD", "sell", 1, 97.0, "2026-06-01T15:10:00Z"),    # -3 loss
    ]
    evs = realized_pnl_events(fills)
    assert len(evs) == 2
    pnls = [round(e["realized_pnl"], 6) for e in evs]
    assert pnls == [1.0, -3.0]               # one win then one loss, distinct events


def test_events_sum_to_total_fifo():
    fills = [
        fill("EEE", "buy", 10, 20.0, "2026-06-01T14:00:00Z"),
        fill("EEE", "sell", 4, 21.0, "2026-06-01T14:10:00Z"),
        fill("EEE", "sell", 6, 19.0, "2026-06-01T14:20:00Z"),
        fill("FFF", "sell", 2, 30.0, "2026-06-01T14:05:00Z"),
        fill("FFF", "buy", 2, 33.0, "2026-06-01T14:25:00Z"),
    ]
    total_events = sum(e["realized_pnl"] for e in realized_pnl_events(fills))
    total_fifo = sum(realized_pnl_fifo(fills).values())
    assert abs(total_events - total_fifo) < 1e-9
