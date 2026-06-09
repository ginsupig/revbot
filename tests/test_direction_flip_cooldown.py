"""Direction-flip cooldown: block opposite-side re-entry after an entry, to
stop the short->stop->long->stop whipsaw seen in live trading.
"""
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _note(ps, symbol, side, when):
    ps.note_new_position(symbol, "mean_reversion", "reversion", when.isoformat(), side=side)


# --- PortfolioState.in_direction_flip_cooldown -------------------------------

def test_opposite_side_blocked_within_window(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    t0 = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
    _note(ps, "ARM", "short", t0)
    # 33 min later (past the 30-min plain cooldown) a LONG flip is still blocked.
    assert ps.in_direction_flip_cooldown("ARM", "long", t0 + timedelta(minutes=33), 60) is True


def test_same_side_not_blocked(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    t0 = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
    _note(ps, "ARM", "short", t0)
    assert ps.in_direction_flip_cooldown("ARM", "short", t0 + timedelta(minutes=33), 60) is False


def test_opposite_side_allowed_after_window(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    t0 = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
    _note(ps, "ARM", "short", t0)
    assert ps.in_direction_flip_cooldown("ARM", "long", t0 + timedelta(minutes=61), 60) is False


def test_disabled_when_zero(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    t0 = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
    _note(ps, "ARM", "short", t0)
    assert ps.in_direction_flip_cooldown("ARM", "long", t0 + timedelta(minutes=1), 0) is False


def test_unknown_symbol_not_blocked(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    t0 = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
    assert ps.in_direction_flip_cooldown("ZZZ", "long", t0, 60) is False


# --- Governor end-to-end -----------------------------------------------------

class _Account:
    portfolio_value = "100000"
    equity = "100000"
    buying_power = "100000"
    daytrading_buying_power = "100000"
    trading_blocked = False
    account_blocked = False


class _Client:
    def get_account(self):
        return _Account()

    def list_positions(self):
        return []


class _Executor:
    client = _Client()

    def open_order_symbols(self):
        return set()


def _short_candidate(symbol="ARM"):
    return {"go_short": True, "symbol": symbol, "entry_style": "mean_reversion",
            "regime": "reversion", "position_plan": {"position_value": 1000.0},
            "portfolio_heat": 1.0}


def _long_candidate(symbol="ARM"):
    return {"go_long": True, "symbol": symbol, "entry_style": "mean_reversion",
            "regime": "reversion", "position_plan": {"position_value": 1000.0},
            "portfolio_heat": 1.0}


def test_governor_blocks_the_whipsaw_flip(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    gov = ExecutionGovernor(
        config=PortfolioConfig(direction_flip_cooldown_minutes=60, symbol_cooldown_minutes=30),
        portfolio_state=ps,
    )
    # Just shorted ARM (as if the bot opened it moments ago).
    _note(ps, "ARM", "short", datetime.now(timezone.utc))
    # The opposite-direction long flip is rejected...
    assert gov.approve(_long_candidate("ARM"), executor=_Executor(), trades_executed_this_cycle=0) is False
    # ...but a different symbol with no recent entry is unaffected.
    assert gov.approve(_short_candidate("OXY"), executor=_Executor(), trades_executed_this_cycle=0) is True
