"""End-to-end-ish test of the FILL -> outcome reconciliation that finally makes
outcome-based threshold adaptation live: entries logged at decision time are
matched to closing fills, realized PnL is attributed to the right style/regime,
and re-running is idempotent."""
import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reversion_bot.performance import (
    PerformanceTracker, TradeRecord, utc_now_iso,
)


def entry(pt, symbol, regime, entry_style, ts, entry_price=100.0, qty=10):
    pt.log_trade(TradeRecord(
        timestamp=ts, symbol=symbol, regime=regime, entry_style=entry_style,
        trade_score=0.5, threshold=0.36, entry_price=entry_price,
        stop_price=entry_price * 0.99, target_price=entry_price * 1.02,
        qty=qty, rr_ratio=2.0, risk_per_share=1.0, reward_per_share=2.0,
        position_value=entry_price * qty,
    ))


def fill(symbol, side, qty, price, time):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price, "time": time}


def test_reconcile_attributes_pnl_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        # Entry logged at decision time, before the fills.
        entry(pt, "AAA", "reversion", "mean_reversion", "2026-06-01T14:25:00Z",
              entry_price=100.0, qty=10)
        fills = [
            fill("AAA", "buy", 10, 100.0, "2026-06-01T14:30:00Z"),
            fill("AAA", "sell", 10, 102.0, "2026-06-01T15:30:00Z"),  # +20 realized
        ]
        n = pt.reconcile_outcomes(fills)
        assert n == 1

        outs = [json.loads(l) for l in pt.outcomes_path.read_text().splitlines()]
        assert len(outs) == 1
        o = outs[0]
        assert o["symbol"] == "AAA"
        assert o["regime"] == "reversion"
        assert o["entry_style"] == "mean_reversion"
        assert abs(o["realized_pnl"] - 20.0) < 1e-9
        assert o["entry_price"] == 100.0 and o["qty"] == 10

        # Re-running over the same (overlapping) window logs nothing more.
        assert pt.reconcile_outcomes(fills) == 0
        assert len(pt.outcomes_path.read_text().splitlines()) == 1


def test_reconcile_then_threshold_adaptation_reacts():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        fills = []
        # Six losing mean_reversion round-trips in the same regime.
        for i in range(6):
            ts = f"2026-06-01T14:{10+i:02d}:00Z"
            entry(pt, f"S{i}", "reversion", "mean_reversion", ts, entry_price=100.0)
            fills.append(fill(f"S{i}", "buy", 10, 100.0, f"2026-06-01T14:{20+i:02d}:00Z"))
            fills.append(fill(f"S{i}", "sell", 10, 99.0, f"2026-06-01T15:{20+i:02d}:00Z"))  # -10
        assert pt.reconcile_outcomes(fills) == 6

        K = dict(entry_style="mean_reversion", regime="reversion",
                 baseline_threshold=0.36, min_samples=5, max_adj=0.05)
        # Losing style/regime -> the bar is raised (demand more conviction).
        assert pt.suggest_threshold_adjustment(**K) == 0.41


def test_close_without_matching_entry_is_skipped():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        # Fills with no logged entry to attribute style/regime -> nothing logged.
        fills = [
            fill("ZZZ", "buy", 1, 10.0, "2026-06-01T14:30:00Z"),
            fill("ZZZ", "sell", 1, 11.0, "2026-06-01T14:40:00Z"),
        ]
        assert pt.reconcile_outcomes(fills) == 0
        assert not pt.outcomes_path.exists() or pt.outcomes_path.read_text() == ""


def test_new_close_after_cursor_is_picked_up_on_next_run():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        entry(pt, "AAA", "reversion", "mean_reversion", "2026-06-01T14:25:00Z")
        first = [
            fill("AAA", "buy", 10, 100.0, "2026-06-01T14:30:00Z"),
            fill("AAA", "sell", 10, 101.0, "2026-06-01T15:30:00Z"),
        ]
        assert pt.reconcile_outcomes(first) == 1

        # A later session: a brand-new round-trip after the cursor.
        entry(pt, "AAA", "reversion", "mean_reversion", "2026-06-02T14:25:00Z")
        second = first + [
            fill("AAA", "buy", 10, 100.0, "2026-06-02T14:30:00Z"),
            fill("AAA", "sell", 10, 103.0, "2026-06-02T15:30:00Z"),
        ]
        assert pt.reconcile_outcomes(second) == 1     # only the new close
        assert len(pt.outcomes_path.read_text().splitlines()) == 2


def test_recent_loss_exit_brake():
    from datetime import datetime, timezone, timedelta
    from reversion_bot.performance import OutcomeRecord

    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        now = datetime(2026, 6, 12, 16, 0, tzinfo=timezone.utc)

        # No outcomes yet -> never braked.
        assert pt.recent_loss_exit("WDC", now, 90) is False

        # A losing close 20 min ago -> braked within a 90-min window...
        pt.log_outcome(OutcomeRecord(
            timestamp=(now - timedelta(minutes=20)).isoformat(),
            symbol="WDC", regime="reversion", entry_style="mean_reversion",
            realized_pnl=-110.0,
        ))
        assert pt.recent_loss_exit("WDC", now, 90) is True
        # ...but not once the window has elapsed, and not when disabled.
        assert pt.recent_loss_exit("WDC", now, 10) is False
        assert pt.recent_loss_exit("WDC", now, 0) is False
        # Different symbol is unaffected.
        assert pt.recent_loss_exit("AAPL", now, 90) is False


def test_recent_loss_exit_uses_latest_outcome_only():
    from datetime import datetime, timezone, timedelta
    from reversion_bot.performance import OutcomeRecord

    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        now = datetime(2026, 6, 12, 16, 0, tzinfo=timezone.utc)
        # Earlier loss, then a more recent WIN -> not braked (winners re-enter).
        pt.log_outcome(OutcomeRecord(
            timestamp=(now - timedelta(minutes=40)).isoformat(),
            symbol="WDC", regime="reversion", entry_style="mean_reversion",
            realized_pnl=-50.0,
        ))
        pt.log_outcome(OutcomeRecord(
            timestamp=(now - timedelta(minutes=5)).isoformat(),
            symbol="WDC", regime="reversion", entry_style="mean_reversion",
            realized_pnl=30.0,
        ))
        assert pt.recent_loss_exit("WDC", now, 90) is False
