"""Low-severity sweep checks (audit L-items + M11)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.performance import EvalRecord, PerformanceTracker
from reversion_bot.trade_report import _fill_day


def test_fill_day_uses_eastern_date():
    # 00:30 UTC on July 29 is 20:30 ET on July 28 — the UTC prefix booked this
    # extended-hours fill to the wrong day.
    assert _fill_day({"time": "2026-07-29T00:30:00Z"}) == "2026-07-28"
    assert _fill_day({"time": "2026-07-28T14:30:00Z"}) == "2026-07-28"
    assert _fill_day({"time": "garbage-but-long"}) == "garbage-but"[:10]
    assert _fill_day({"time": ""}) == ""


def test_evaluations_log_rotates_at_cap(tmp_path, monkeypatch):
    t = PerformanceTracker(str(tmp_path))
    monkeypatch.setattr(PerformanceTracker, "_EVALS_MAX_BYTES", 300)
    rec = EvalRecord(timestamp="2026-07-29T00:00:00+00:00", symbol="AAPL",
                     regime="reversion", entry_style="mean_reversion",
                     decision="WAIT", reason="x", router_reason="y",
                     trade_score=0.1, threshold=0.45, go_long=False)
    for _ in range(10):
        t.log_evaluation(rec)
    rolled = tmp_path / "evaluations.jsonl.1"
    assert rolled.exists()
    assert (tmp_path / "evaluations.jsonl").stat().st_size <= 600


def test_config_defaults_match_live_env_defaults():
    from reversion_bot.config import ExecutionConfig, RiskConfig
    assert RiskConfig().min_rr == 1.5
    assert RiskConfig().max_position_value_pct == 0.15
    assert ExecutionConfig().use_limit_entry is True
    assert not hasattr(RiskConfig(), "round_lot")
