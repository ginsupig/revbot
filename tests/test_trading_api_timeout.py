"""ALPACA_HTTP_TIMEOUT must bound TRADING-API calls, not just market data.

Audit H6: the trading REST client had no HTTP timeout, so one hung
get_clock/list_positions/submit_order froze the entire poll loop.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.execution import AlpacaExecutor


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(kwargs)
        return "resp"


class ClientWithSession:
    def __init__(self):
        self._session = FakeSession()


class NestedTradingClient:
    """Shape of the alpaca-py wrapper: session lives on ._trading._session."""
    def __init__(self):
        self._trading = ClientWithSession()


def _executor_with(client):
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = client
    return ex


def test_timeout_injected_when_absent():
    client = ClientWithSession()
    ex = _executor_with(client)
    ex._apply_http_timeout(15.0)
    client._session.request("GET", "https://api")
    assert client._session.calls[-1]["timeout"] == 15.0


def test_explicit_timeout_respected():
    client = ClientWithSession()
    ex = _executor_with(client)
    ex._apply_http_timeout(15.0)
    client._session.request("GET", "https://api", timeout=3.0)
    assert client._session.calls[-1]["timeout"] == 3.0


def test_nested_alpaca_py_session_also_wrapped():
    client = NestedTradingClient()
    ex = _executor_with(client)
    ex._apply_http_timeout(10.0)
    inner = client._trading._session
    inner.request("GET", "https://api")
    assert inner.calls[-1]["timeout"] == 10.0


def test_disabled_or_missing_session_is_harmless():
    ex = _executor_with(object())          # no _session anywhere
    ex._apply_http_timeout(15.0)           # must not raise
    ex2 = _executor_with(ClientWithSession())
    ex2._apply_http_timeout(0)             # disabled: no wrap
    ex2.client._session.request("GET", "https://api")
    assert "timeout" not in ex2.client._session.calls[-1]
