"""Performance-layer hardening (audit M1/M3/M4/M5).

- Adaptive-threshold LOOSENING must actually admit trades (it was dead code:
  a baseline-level hard block overrode the adaptive value).
- TradeRecords are logged only after a real submission, not at decision time.
- Timestamp comparisons are parsed, not lexical (Z vs +00:00 vs space formats
  mis-order badly), and naive timestamps can't raise.
- The adaptive threshold learns from a recency window, not all history.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from reversion_bot.config import PerformanceConfig
from reversion_bot.models import ReversionDecision
from reversion_bot.performance import (
    OutcomeRecord,
    PerformanceTracker,
    parse_ts,
)
from reversion_bot.service import ReversionService


# --- parse_ts -----------------------------------------------------------------

def test_parse_ts_formats_agree():
    a = parse_ts("2026-07-28T10:00:00Z")
    b = parse_ts("2026-07-28T10:00:00+00:00")
    c = parse_ts("2026-07-28 10:00:00+00:00")
    naive = parse_ts("2026-07-28T10:00:00")
    assert a == b == c == naive
    assert parse_ts("") is None and parse_ts("garbage") is None


def test_parse_ts_orders_across_formats():
    # str(datetime) uses a space separator, which sorts lexically BEFORE any
    # 'T'-format string regardless of the actual time — the old comparisons
    # would call this 14:00 event "older" than a 10:00 one.
    newer_raw = "2026-07-28 14:00:00+00:00"
    older_raw = "2026-07-28T10:00:00Z"
    assert newer_raw < older_raw                      # the lexical bug
    assert parse_ts(newer_raw) > parse_ts(older_raw)  # parsed: correct order


# --- recent_loss_exit ----------------------------------------------------------

def _tracker(tmp_path):
    return PerformanceTracker(str(tmp_path))


def test_recent_loss_exit_picks_true_latest_across_formats(tmp_path):
    t = _tracker(tmp_path)
    now = datetime.now(timezone.utc)
    older_loss = (now - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_win = (now - timedelta(minutes=5)).isoformat()   # +00:00 format
    t.log_outcome(OutcomeRecord(timestamp=older_loss, symbol="WDC",
                                regime="reversion", entry_style="mean_reversion",
                                realized_pnl=-50.0))
    t.log_outcome(OutcomeRecord(timestamp=recent_win, symbol="WDC",
                                regime="reversion", entry_style="mean_reversion",
                                realized_pnl=+30.0))
    # Lexically the 'Z' loss sorts AFTER the '+00:00' win, so the old code
    # judged the stale loss "latest" and braked a name whose last close WON.
    assert t.recent_loss_exit("WDC", now, within_minutes=90) is False


def test_recent_loss_exit_naive_timestamp_does_not_raise(tmp_path):
    t = _tracker(tmp_path)
    now = datetime.now(timezone.utc)
    naive = (now - timedelta(minutes=10)).replace(tzinfo=None).isoformat()
    t.log_outcome(OutcomeRecord(timestamp=naive, symbol="AAPL",
                                regime="reversion", entry_style="mean_reversion",
                                realized_pnl=-10.0))
    assert t.recent_loss_exit("AAPL", now, within_minutes=90) is True


# --- adaptive threshold recency window ------------------------------------------

def test_threshold_recency_window_forgives_dead_config_losses(tmp_path):
    t = _tracker(tmp_path)
    kw = dict(symbol="X", regime="reversion", entry_style="mean_reversion")
    for i in range(30):    # ancient losing era
        t.log_outcome(OutcomeRecord(timestamp=f"2026-01-01T00:{i:02d}:00+00:00",
                                    realized_pnl=-100.0, **kw))
    for i in range(30):    # current winning era
        t.log_outcome(OutcomeRecord(timestamp=f"2026-07-01T00:{i:02d}:00+00:00",
                                    realized_pnl=+100.0, **kw))
    # Full history would be 50/50 (no adjustment or raise); a 30-outcome
    # recency window sees only the winning era -> loosen.
    out = t.suggest_threshold_adjustment(
        entry_style="mean_reversion", regime="reversion",
        baseline_threshold=0.45, min_samples=20, max_adj=0.05,
        recency_limit=30,
    )
    assert out == 0.40


# --- service: adaptive loosening is live; trades logged post-submit --------------

def _svc(tmp_path):
    return ReversionService(
        performance_config=PerformanceConfig(state_dir=str(tmp_path)),
        log_file=str(tmp_path / "svc.log"),
    )


def _frame(close, ri, rsi, adx):
    return pd.DataFrame({
        "open": [close + 1.0], "high": [close + 2.0], "low": [close - 2.0],
        "close": [close], "volume": [1_000_000.0],
        "ri": [ri], "rsi": [rsi], "adx": [adx],
        "trend_following_signal": [0], "macd_line": [0.0],
        "macd_signal": [0.0], "trend_ema": [close],
    })


def _validated_decision(close, ri, rsi, adx):
    return ReversionDecision(
        signal="LONG_REVERSION", reason="Validated_Long_Reversion", symbol="T",
        close=close, lb1=100.0, lb2=90.0, ub1=120.0, ub2=130.0,
        sma=110.0, ri=ri, rsi=rsi, adx=adx, atr=2.0, vwap=100.0,
    )


def _eval_with_threshold(tmp_path, adaptive_value):
    # A weak-but-validated setup: mr ≈ 0.425 (0.15 base + 0.20 validated +
    # 0.05 shallow depth + 0.02 RSI softness + ~0.005 ADX buffer).
    svc = _svc(tmp_path)
    enriched = _frame(close=97.5, ri=-0.5, rsi=45.0, adx=44.0)
    svc.engine.calculate_indicators = lambda df: enriched
    svc.engine.get_decision = (
        lambda df, symbol=None, short_bias=False:
        _validated_decision(97.5, -0.5, 45.0, 44.0)
    )
    # Pin the trendfail component low: its 1-row-frame fallback of 0.45 would
    # otherwise out-score mr (0.425) and hijack the routing.
    svc._get_trendfail_score = lambda df, engine=None: 0.10
    if adaptive_value is not None:
        svc.perf.suggest_threshold_adjustment = lambda **kw: adaptive_value
    return svc, svc.evaluate_symbol("T", enriched, account_equity=100_000.0)


def test_adaptive_loosening_admits_subbaseline_validated_signal(tmp_path):
    svc, result = _eval_with_threshold(tmp_path, adaptive_value=0.40)
    assert 0.40 <= result["component_scores"]["mean_reversion"] < 0.45
    assert result["go_long"] is True          # was impossible before the fix


def test_baseline_still_blocks_subbaseline_signal(tmp_path):
    svc, result = _eval_with_threshold(tmp_path, adaptive_value=None)
    assert result["go_long"] is False


def test_trade_record_logged_only_after_submit(tmp_path):
    svc = _svc(tmp_path)
    enriched = _frame(close=80.0, ri=-1.5, rsi=5.0, adx=20.0)
    svc.engine.calculate_indicators = lambda df: enriched
    svc.engine.get_decision = (
        lambda df, symbol=None, short_bias=False:
        _validated_decision(80.0, -1.5, 5.0, 20.0)
    )
    result = svc.evaluate_symbol("T", enriched, account_equity=100_000.0)
    assert result["go_long"] is True
    trades_file = Path(tmp_path) / "trades.jsonl"
    # Decision alone books nothing — the governor/broker can still veto it.
    assert not trades_file.exists() or trades_file.read_text().strip() == ""

    svc.record_submitted_trade(result)
    lines = trades_file.read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    rec = json.loads(lines[0])
    assert rec["symbol"] == "T" and rec["qty"] > 0
