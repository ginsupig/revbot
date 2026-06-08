"""Startup credential preflight: fail fast and clearly on bad creds."""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the legacy Alpaca SDK + the data helper so main.py imports without the
# dependency or any network.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

import main


class _Account:
    def __init__(self, status="ACTIVE", equity="100000", buying_power="100000",
                 trading_blocked=False, account_blocked=False, shorting_enabled=True):
        self.status = status
        self.equity = equity
        self.buying_power = buying_power
        self.trading_blocked = trading_blocked
        self.account_blocked = account_blocked
        self.shorting_enabled = shorting_enabled


class _Client:
    def __init__(self, account=None, error=None):
        self._account = account
        self._error = error

    def get_account(self):
        if self._error is not None:
            raise self._error
        return self._account


class _Executor:
    def __init__(self, account=None, error=None):
        self.client = _Client(account, error)


def test_auth_error_aborts():
    ex = _Executor(error=Exception("request is not authorized"))
    assert main.preflight_check(ex, paper=False) is False


def test_network_error_aborts():
    ex = _Executor(error=Exception("Max retries exceeded: connection refused"))
    assert main.preflight_check(ex, paper=True) is False


def test_healthy_account_proceeds():
    ex = _Executor(account=_Account())
    assert main.preflight_check(ex, paper=True) is True


def test_blocked_account_aborts():
    ex = _Executor(account=_Account(trading_blocked=True))
    assert main.preflight_check(ex, paper=False) is False


def test_shorting_disabled_still_proceeds():
    # A missing short permission is a warning, not a hard stop (longs still work).
    ex = _Executor(account=_Account(shorting_enabled=False))
    assert main.preflight_check(ex, paper=False) is True
