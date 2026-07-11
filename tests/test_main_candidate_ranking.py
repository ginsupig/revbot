import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import execute_candidates


class _Gov:
    def __init__(self):
        self.approve_order = []

    def rank_candidates(self, candidates):
        return sorted(candidates, key=lambda c: c["trade_score"], reverse=True)

    def approve(self, candidate, executor, trades_executed_this_cycle):
        self.approve_order.append(candidate["symbol"])
        return True


class _Exec:
    def __init__(self):
        self.submitted = []

    def submit_order(self, candidate):
        self.submitted.append(candidate["symbol"])


class _State:
    def __init__(self):
        self.updated = []

    def update(self, candidate):
        self.updated.append(candidate["symbol"])


def test_execute_candidates_runs_in_ranked_order():
    gov = _Gov()
    ex = _Exec()
    st = _State()
    candidates = [
        {"symbol": "LOW", "trade_score": 0.20, "go_long": True},
        {"symbol": "HIGH", "trade_score": 0.95, "go_long": True},
        {"symbol": "MID", "trade_score": 0.60, "go_long": True},
    ]

    asyncio.run(execute_candidates(gov, ex, st, candidates))

    assert gov.approve_order == ["HIGH", "MID", "LOW"]
    assert ex.submitted == ["HIGH", "MID", "LOW"]
    assert st.updated == ["HIGH", "MID", "LOW"]
