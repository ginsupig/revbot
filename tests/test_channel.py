import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the legacy SDK so importing the executor (for close_long) needs no dep.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.channel import regression_channel_position, select_breakout_exits
from reversion_bot.execution import AlpacaExecutor


# --- regression_channel_position --------------------------------------------

def test_high_when_last_bar_above_trend():
    pos = regression_channel_position([100.0] * 79 + [110.0], lookback=80)
    assert pos is not None and pos > 0.5


def test_low_when_last_bar_below_trend():
    pos = regression_channel_position([100.0] * 79 + [90.0], lookback=80)
    assert pos is not None and pos < 0.5


def test_clamped_to_unit_interval():
    assert regression_channel_position([100.0] * 79 + [1000.0], lookback=80) == 1.0
    assert regression_channel_position([100.0] * 79 + [-1000.0], lookback=80) == 0.0


def test_none_on_flat_series():
    # Zero residual spread -> degenerate channel -> None (don't act).
    assert regression_channel_position([100.0] * 80, lookback=80) is None


def test_none_on_insufficient_or_bad_params():
    assert regression_channel_position([100.0] * 50, lookback=80) is None
    assert regression_channel_position([100.0] * 80, lookback=1) is None
    assert regression_channel_position([100.0] * 80, lookback=80, k=0) is None


# --- select_breakout_exits ---------------------------------------------------

class _Pos:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


def test_selects_only_longs_at_the_upper_channel():
    positions = [
        _Pos("AMD", "10"),    # long, at the top -> exit
        _Pos("SPY", "-5"),    # SHORT at the top -> never (longs only)
        _Pos("MU", "8"),      # long, flat channel (None) -> keep
    ]
    closes = {
        "AMD": [100.0] * 79 + [300.0],
        "SPY": [100.0] * 79 + [300.0],
        "MU": [100.0] * 80,
    }
    assert select_breakout_exits(positions, closes, threshold=0.80, lookback=80) == ["AMD"]


def test_missing_closes_and_below_threshold_are_skipped():
    positions = [_Pos("AAA", "5"), _Pos("BBB", "5")]
    closes = {"AAA": [100.0] * 79 + [90.0]}  # AAA below mid; BBB has no bars
    assert select_breakout_exits(positions, closes, threshold=0.80, lookback=80) == []


# --- executor.close_long -----------------------------------------------------

def test_close_long_closes_position_per_symbol_uppercased():
    class _Client:
        def __init__(self):
            self.closed = []
        def close_position(self, symbol):
            self.closed.append(symbol)
            return {"closed": symbol}

    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = _Client()
    ex.close_long("amd")
    assert ex.client.closed == ["AMD"]   # uppercased, per-symbol (no global cancel)
