"""Pending (unfilled) entry orders must count toward the portfolio caps.

Regression tests for audit H1: the governor counted only FILLED positions, so
resting entry brackets — invisible to list_positions — let max_open_positions,
exposure, and heat all be overshot (3 positions + 2 approvals in one cycle ->
5 positions once the brackets fill).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.config import PortfolioConfig
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.portfolio import PortfolioState


class FakeOrder:
    def __init__(self, symbol, qty=10, limit_price=None, stop_price=None):
        self.symbol = symbol
        self.qty = qty
        self.limit_price = limit_price
        self.stop_price = stop_price


class FakePosition:
    def __init__(self, symbol, qty=10, market_value=1000.0, unrealized_pl=0.0):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl


class FakeAccount:
    portfolio_value = "100000"
    equity = "100000"
    buying_power = "200000"
    daytrading_buying_power = None
    trading_blocked = False
    account_blocked = False


class FakeClient:
    def __init__(self, positions):
        self._positions = positions

    def get_account(self):
        return FakeAccount()

    def list_positions(self):
        return self._positions


class FakeExecutor:
    def __init__(self, positions, open_orders):
        self.client = FakeClient(positions)
        self._orders = open_orders

    def list_open_orders(self):
        return self._orders


def _gov(tmp_path, **cfg):
    state = PortfolioState(state_dir=str(tmp_path))
    return ExecutionGovernor(PortfolioConfig(**cfg), portfolio_state=state)


def _candidate(symbol, value=1000.0, heat=50.0):
    return {
        "symbol": symbol,
        "go_long": True,
        "entry_style": "mean_reversion",
        "regime": "reversion",
        "trade_score": 0.6,
        "position_plan": {"qty": 10, "entry_price": value / 10,
                          "position_value": value},
        "portfolio_heat": heat,
    }


def test_pending_entry_occupies_a_position_slot(tmp_path):
    gov = _gov(tmp_path, max_open_positions=4)
    ok, reason = gov.can_open(
        symbol="NEW", entry_style="mean_reversion", regime="reversion",
        account_equity=100_000,
        open_symbols=["AAA", "BBB", "CCC"],
        open_styles={}, open_regimes={},
        current_total_exposure=3000.0, current_total_heat=0.0,
        new_position_value=1000.0, new_position_heat=0.0,
        trades_executed_this_cycle=0,
        open_order_symbols={"DDD"},          # resting entry, no position yet
    )
    assert ok is False
    assert reason == "max_open_positions_reached"


def test_bracket_legs_of_live_positions_do_not_double_count(tmp_path):
    gov = _gov(tmp_path, max_open_positions=4)
    ok, reason = gov.can_open(
        symbol="NEW", entry_style="mean_reversion", regime="reversion",
        account_equity=100_000,
        open_symbols=["AAA", "BBB", "CCC"],
        open_styles={}, open_regimes={},
        current_total_exposure=3000.0, current_total_heat=0.0,
        new_position_value=1000.0, new_position_heat=0.0,
        trades_executed_this_cycle=0,
        open_order_symbols={"AAA", "BBB"},   # just the live brackets' legs
    )
    assert ok is True, reason


def test_pending_entry_orders_helper_excludes_position_symbols():
    orders = [
        FakeOrder("AAA", qty=10, limit_price=50.0),    # leg of live position
        FakeOrder("DDD", qty=10, limit_price=20.0),    # pending entry parent
        FakeOrder("DDD", qty=10, stop_price=19.0),     # held leg, same symbol
        FakeOrder("EEE", qty=5),                       # market parent: no price
    ]
    pending = ExecutionGovernor.pending_entry_orders(orders, ["AAA"])
    assert set(pending) == {"DDD", "EEE"}
    assert pending["DDD"]["notional"] == 200.0         # max across DDD orders
    assert pending["EEE"]["notional"] == 0.0           # slot counts, value unknown


def test_approve_counts_pending_notional_toward_exposure(tmp_path):
    # Exposure cap 5% of 100k = 5000. Open position 3000 + pending bracket
    # 10 x 150 = 1500 -> 4500 committed; a 1000 candidate must be rejected
    # (4500 + 1000 > 5000). Without pending accounting it would pass.
    gov = _gov(tmp_path, max_open_positions=10, max_total_exposure_pct=0.05,
               max_portfolio_heat_pct=1.0)
    ex = FakeExecutor(
        positions=[FakePosition("AAA", market_value=3000.0)],
        open_orders=[FakeOrder("DDD", qty=10, limit_price=150.0)],
    )
    assert gov.approve(_candidate("NEW", value=1000.0), ex, 0) is False

    ex_no_pending = FakeExecutor(
        positions=[FakePosition("AAA", market_value=3000.0)],
        open_orders=[],
    )
    assert gov.approve(_candidate("NEW", value=1000.0), ex_no_pending, 0) is True


def test_approve_counts_pending_heat_from_submit_metadata(tmp_path):
    # Heat cap 1% of 100k = 1000. Pending DDD was recorded at submit with
    # stop_distance 9.0 and rests as a 100-share bracket -> 900 heat.
    # A candidate bringing 200 more heat must be rejected (900 + 200 > 1000).
    state = PortfolioState(state_dir=str(tmp_path))
    gov = ExecutionGovernor(
        PortfolioConfig(max_open_positions=10, max_total_exposure_pct=10.0,
                        max_portfolio_heat_pct=0.01),
        portfolio_state=state,
    )
    state.update({"symbol": "DDD", "entry_style": "mean_reversion",
                  "regime": "reversion",
                  "position_plan": {"entry_price": 100.0, "risk_per_share": 9.0}})
    ex = FakeExecutor(
        positions=[],
        open_orders=[FakeOrder("DDD", qty=100, limit_price=100.0)],
    )
    candidate = _candidate("NEW", value=1000.0, heat=200.0)
    assert gov.approve(candidate, ex, 0) is False

    light = _candidate("NEW", value=1000.0, heat=50.0)   # 900 + 50 < 1000
    assert gov.approve(light, ex, 0) is True
