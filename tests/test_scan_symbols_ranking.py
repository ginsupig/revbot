"""scan_symbols must rank by measured liquidity, not arbitrary API order.

Audit H4: Alpaca Asset objects have no market_cap attribute, so the old
"sort by market cap" was a no-op — the default live universe was whatever
the API listed first (roughly alphabetical), first-match-wins.
"""
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.execution import AlpacaExecutor


class FakeAsset:
    def __init__(self, symbol):
        self.symbol = symbol
        self.tradable = True
        self.easy_to_borrow = True
        self.status = "active"
        self.exchange = "NASDAQ"


class FakeTrade:
    def __init__(self, price):
        self.price = price


class FakeBars:
    def __init__(self, close, volume):
        self.df = pd.DataFrame({"close": [close] * 5, "volume": [volume] * 5})


class FakeClient:
    """Symbols alphabetically early are LESS liquid than later ones — the old
    first-match-wins scan would pick the illiquid early names."""

    def __init__(self, liquidity_by_symbol, price=50.0):
        self._liq = liquidity_by_symbol
        self._price = price

    def list_assets(self, status=None, asset_class=None):
        return [FakeAsset(s) for s in self._liq]

    def get_latest_trade(self, symbol):
        return FakeTrade(self._price)

    def get_bars(self, symbol, timeframe, limit=None):
        dollar_vol = self._liq[symbol]
        return FakeBars(close=self._price, volume=dollar_vol / self._price)


def _executor(client):
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = client
    return ex


def test_universe_is_top_by_dollar_volume_not_alphabetical():
    liq = {
        "AAAA": 1_000_000,     # early alphabet, modest liquidity
        "BBBB": 2_000_000,
        "MMMM": 50_000_000,    # most liquid
        "ZZZZ": 40_000_000,
    }
    ex = _executor(FakeClient(liq))
    out = ex.scan_symbols(min_price=5.0, min_dollar_volume=750_000, max_count=2)
    assert out == ["MMMM", "ZZZZ"]     # NOT ["AAAA", "BBBB"]


def test_illiquid_and_cheap_names_are_still_filtered():
    liq = {"AAAA": 100_000, "BBBB": 5_000_000}
    ex = _executor(FakeClient(liq))
    out = ex.scan_symbols(min_price=5.0, min_dollar_volume=750_000, max_count=5)
    assert out == ["BBBB"]             # AAAA under the dollar-volume floor

    cheap = _executor(FakeClient({"CCCC": 5_000_000}, price=1.0))
    assert cheap.scan_symbols(min_price=5.0, min_dollar_volume=750_000,
                              max_count=5) == []


def test_deterministic_tiebreak_on_equal_liquidity():
    liq = {"DDDD": 1_000_000, "CCCC": 1_000_000}
    ex = _executor(FakeClient(liq))
    out = ex.scan_symbols(min_price=5.0, min_dollar_volume=750_000, max_count=2)
    assert out == ["CCCC", "DDDD"]     # alphabetical among equals
