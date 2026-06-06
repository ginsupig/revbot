"""End-to-end: an overbought bar flows signal -> governor approval -> a real
sell-bracket order. This is the closest verification to a live short without
hitting a broker account.
"""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the legacy Alpaca SDK so the executor imports without the dependency.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.config import ReversionConfig, RiskConfig, PerformanceConfig, PortfolioConfig
from reversion_bot.service import ReversionService
from reversion_bot.execution import AlpacaExecutor
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.portfolio import PortfolioState
from test_engine import make_overbought_df


class FakeClock:
    is_open = True


class FakeAccount:
    portfolio_value = "100000"
    equity = "100000"
    buying_power = "100000"
    daytrading_buying_power = "100000"
    trading_blocked = False
    account_blocked = False


class FakeClient:
    def __init__(self):
        self.orders = []
        self._positions = []

    def get_clock(self):
        return FakeClock()

    def get_account(self):
        return FakeAccount()

    def list_positions(self):
        return self._positions

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "fake", **kwargs}


def make_executor():
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = FakeClient()
    ex._order_ids = set()
    ex._tif = "day"
    ex._use_limit_entry = False
    ex._limit_offset_bps = 8.0
    return ex


def test_short_flows_signal_to_sell_bracket(tmp_path):
    # 1. Signal: overbought bar -> go_short candidate with a short plan.
    svc = ReversionService(
        ReversionConfig(min_history=60, enable_shorts=True),
        RiskConfig(),
        PerformanceConfig(state_dir=str(tmp_path / "perf")),
        log_file=str(tmp_path / "svc.log"),
    )
    candidate = svc.evaluate_symbol("TEST", make_overbought_df(), account_equity=100000)
    assert candidate["go_short"] is True
    assert candidate["position_plan"]["side"] == "short"

    # 2. Approval: the governor clears it against a flat, healthy account.
    executor = make_executor()
    governor = ExecutionGovernor(
        config=PortfolioConfig(),
        portfolio_state=PortfolioState(state_dir=str(tmp_path / "pf")),
    )
    assert governor.approve(candidate, executor=executor, trades_executed_this_cycle=0) is True

    # 3. Execution: it lands as a sell bracket with stop above / target below.
    executor.submit_order(candidate)
    order = executor.client.orders[-1]
    assert order["side"] == "sell"
    assert order["order_class"] == "bracket"
    assert order["stop_loss"]["stop_price"] > order["take_profit"]["limit_price"]
