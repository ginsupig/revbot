import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reversion_bot.performance import PerformanceTracker, OutcomeRecord, utc_now_iso

K = dict(entry_style="mean_reversion", regime="reversion",
         baseline_threshold=0.36, min_samples=5, max_adj=0.05)

def rec(pnl):
    return OutcomeRecord(timestamp=utc_now_iso(), symbol="TEST",
                         regime="reversion", entry_style="mean_reversion",
                         realized_pnl=pnl)

def test_no_outcomes_is_strict_noop():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        assert pt.suggest_threshold_adjustment(**K) == 0.36

def test_losing_outcomes_raise_threshold():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        for p in [-10, -8, -12, 5, -7, -9]:           # win rate 1/6, negative expectancy
            pt.log_outcome(rec(p))
        assert pt.suggest_threshold_adjustment(**K) == 0.41

def test_winning_outcomes_lower_threshold():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        for p in [10, 8, 12, -5, 7, 9]:               # win rate 5/6, positive expectancy
            pt.log_outcome(rec(p))
        assert abs(pt.suggest_threshold_adjustment(**K) - 0.31) < 1e-12

def test_mixed_outcomes_hold_baseline():
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        for p in [5, -5, 6, -6, 4, -4]:               # 50/50, ~zero expectancy
            pt.log_outcome(rec(p))
        assert pt.suggest_threshold_adjustment(**K) == 0.36

def test_old_score_loop_is_dead_trades_alone_change_nothing():
    """Trades WITHOUT outcomes must no longer move the threshold (the old
    self-referential behavior)."""
    with tempfile.TemporaryDirectory() as d:
        pt = PerformanceTracker(d)
        import json
        with open(pt.trades_path, "w") as f:
            for _ in range(50):
                f.write(json.dumps({"entry_style": "mean_reversion",
                                    "regime": "reversion", "trade_score": 0.9}) + "\n")
        assert pt.suggest_threshold_adjustment(**K) == 0.36
