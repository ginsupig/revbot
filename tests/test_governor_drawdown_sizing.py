"""Governor drawdown-scaled sizing: once intraday drawdown crosses
reduce_size_after_drawdown_pct (but before the hard drawdown_pause), new entries
are taken at reduced_risk_multiplier size. This locks in the wiring of
effective_risk_multiplier, which was previously dead code (full-size trades kept
firing right up to the hard pause).
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.config import PortfolioConfig
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.portfolio import PortfolioState


class _FakeAccount:
    portfolio_value = "98500"      # 1.5% below the 100k session peak
    equity = "98500"
    buying_power = "1000000"
    daytrading_buying_power = "1000000"
    trading_blocked = False
    account_blocked = False


class _FakeClient:
    def get_account(self):
        return _FakeAccount()

    def list_positions(self):
        return []


class _FakeExecutor:
    def __init__(self):
        self.client = _FakeClient()

    def open_order_symbols(self):
        return set()


def _candidate(qty=100, entry=50.0, rps=0.5):
    return {
        "symbol": "AAPL",
        "go_long": True,
        "entry_style": "mean_reversion",
        "regime": "reversion",
        "portfolio_heat": 10.0,
        "position_plan": {
            "qty": qty,
            "entry_price": entry,
            "stop_price": entry - 1.0,
            "target_price": entry + 2.0,
            "risk_per_share": rps,
            "reward_per_share": 1.0,
            "rr_ratio": 2.0,
            "position_value": round(qty * entry, 2),
            "max_account_risk": round(qty * rps, 2),
            "side": "long",
        },
    }


def _governor(tmp_path, session_equity=100000.0):
    ps = PortfolioState(state_dir=str(tmp_path / "pf"))
    # Seed today's session high-water mark at 100k; the fake account is at 98.5k,
    # so get_drawdown_pct() == 0.015 -> inside the [1%, 2.5%) reduce band.
    ps.update_equity(session_equity)
    gov = ExecutionGovernor(config=PortfolioConfig(), portfolio_state=ps)
    return gov


def test_full_size_when_no_drawdown(tmp_path):
    # Session peak == current equity -> 0 drawdown -> no reduction.
    gov = _governor(tmp_path, session_equity=98500.0)
    cand = _candidate(qty=100)
    assert gov.approve(cand, _FakeExecutor(), trades_executed_this_cycle=0) is True
    assert cand["position_plan"]["qty"] == 100


def test_size_reduced_in_drawdown_band(tmp_path):
    gov = _governor(tmp_path, session_equity=100000.0)  # 1.5% drawdown vs 98.5k
    cand = _candidate(qty=100, entry=50.0, rps=0.5)
    assert gov.approve(cand, _FakeExecutor(), trades_executed_this_cycle=0) is True
    # reduced_risk_multiplier default 0.60 -> 100 * 0.60 = 60 shares.
    assert cand["position_plan"]["qty"] == 60
    # position_value stays consistent with the reduced qty.
    assert cand["position_plan"]["position_value"] == round(60 * 50.0, 2)


def test_reduced_qty_never_zero(tmp_path):
    # A tiny position (1 share) must not be scaled to 0 -> either 1 share or the
    # trade would be dropped; we keep a floor of 1 so an approved trade still fires.
    gov = _governor(tmp_path, session_equity=100000.0)
    cand = _candidate(qty=1)
    assert gov.approve(cand, _FakeExecutor(), trades_executed_this_cycle=0) is True
    assert cand["position_plan"]["qty"] == 1
