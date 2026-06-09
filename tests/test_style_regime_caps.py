"""The per-style / per-regime caps were dead: the governor compared against
empty dicts. These tests cover the reconstruction from PortfolioState metadata
and that the cap now actually binds in approve().
"""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the legacy Alpaca SDK (governor itself doesn't import it, but the test
# executor mirrors the live one; keep the import path clean regardless).
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.portfolio import PortfolioState
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.config import PortfolioConfig


def _note(ps, symbol, style, regime):
    ps.note_new_position(symbol, style, regime, "2026-06-05T15:00:00+00:00")


def test_counts_only_currently_open_symbols(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    _note(ps, "AAA", "mean_reversion", "reversion")
    _note(ps, "BBB", "mean_reversion", "reversion")
    _note(ps, "CCC", "trend_following", "trend")

    # Only AAA and CCC are still open; BBB has been closed.
    styles, regimes = ps.open_style_regime_counts(["AAA", "CCC"])
    assert styles == {"mean_reversion": 1, "trend_following": 1}
    assert regimes == {"reversion": 1, "trend": 1}


def test_unknown_symbols_are_ignored(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    _note(ps, "AAA", "mean_reversion", "reversion")
    styles, regimes = ps.open_style_regime_counts(["AAA", "ZZZ"])  # ZZZ has no meta
    assert styles == {"mean_reversion": 1}
    assert regimes == {"reversion": 1}


class _Account:
    portfolio_value = "100000"
    equity = "100000"
    buying_power = "100000"
    daytrading_buying_power = "100000"
    trading_blocked = False
    account_blocked = False


class _Pos:
    def __init__(self, symbol, mv):
        self.symbol = symbol
        self.market_value = mv
        self.unrealized_pl = 0.0


class _Client:
    def __init__(self, positions):
        self._positions = positions

    def get_account(self):
        return _Account()

    def list_positions(self):
        return self._positions


class _Executor:
    def __init__(self, positions):
        self.client = _Client(positions)

    def open_order_symbols(self):
        return set()


def test_reversion_style_cap_now_binds(tmp_path):
    # Two mean-reversion positions already open; cap is 2 -> a third is rejected.
    ps = PortfolioState(state_dir=str(tmp_path))
    _note(ps, "AAA", "mean_reversion", "reversion")
    _note(ps, "BBB", "mean_reversion", "reversion")

    gov = ExecutionGovernor(
        config=PortfolioConfig(max_reversion_positions=2, max_positions_per_regime=99),
        portfolio_state=ps,
    )
    executor = _Executor([_Pos("AAA", 1000.0), _Pos("BBB", 1000.0)])
    candidate = {
        "go_long": True,
        "symbol": "CCC",
        "entry_style": "mean_reversion",
        "regime": "reversion",
        "position_plan": {"position_value": 1000.0},
        "portfolio_heat": 1.0,
    }
    assert gov.approve(candidate, executor=executor, trades_executed_this_cycle=0) is False


def test_governor_rejects_when_a_working_order_exists(tmp_path):
    # A working (unfilled) order on the symbol must block a new entry at the
    # governor, closing the cross-cycle double-submit gap.
    ps = PortfolioState(state_dir=str(tmp_path))
    gov = ExecutionGovernor(config=PortfolioConfig(), portfolio_state=ps)

    class _ExecutorWithWorkingOrder(_Executor):
        def open_order_symbols(self):
            return {"CCC"}

    candidate = {
        "go_long": True, "symbol": "CCC", "entry_style": "mean_reversion",
        "regime": "reversion", "position_plan": {"position_value": 1000.0},
        "portfolio_heat": 1.0,
    }
    ex = _ExecutorWithWorkingOrder([])
    assert gov.approve(candidate, executor=ex, trades_executed_this_cycle=0) is False
