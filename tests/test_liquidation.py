"""EOD/carryover flatten failure handling.

Regression tests for the audit's C2: liquidate_all_positions cancels every
protective bracket leg up front, so a close that then fails leaves a stop-less
position — and reconcile_carryover used to stamp the day "reconciled" anyway,
guaranteeing no retry until tomorrow. Now: liquidation reports success/failure,
a failed close re-arms an emergency GTC stop, the reconcile stamp is only
written on success, and fractional positions are no longer skipped.
"""
import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the legacy SDK so importing the executor needs no network/dependency.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

import main
from reversion_bot.execution import AlpacaExecutor


class FakePosition:
    def __init__(self, symbol, qty, current_price=100.0, avg_entry_price=100.0):
        self.symbol = symbol
        self.qty = qty
        self.current_price = current_price
        self.avg_entry_price = avg_entry_price


class FakeClient:
    def __init__(self, positions=None, fail_close_symbols=(), fail_list=False):
        self._positions = list(positions or [])
        self._fail_close = set(fail_close_symbols)
        self._fail_list = fail_list
        self.closed = []
        self.orders = []           # submit_order kwargs (emergency stops land here)
        self.cancel_all_calls = 0

    def cancel_all_orders(self):
        self.cancel_all_calls += 1

    def list_positions(self):
        if self._fail_list:
            raise RuntimeError("api down")
        return self._positions

    def list_orders(self, status=None, limit=None):
        # liquidate_all_positions is now OWNER-SCOPED: it only flattens symbols
        # whose orders carry revbot's client_order_id prefix. This fixture used
        # to return [], which under the new contract means "revbot owns
        # nothing" and correctly flattens nothing -- so every test here failed.
        # These stubs exist to say "yes, revbot opened these", keeping each
        # test aimed at what it was written for (flatten failure handling)
        # rather than at ownership. Ownership itself is covered in
        # tests/test_order_ownership.py, including a foreign position that must
        # be left alone.
        if status == "open":
            return []
        return [
            types.SimpleNamespace(symbol=p.symbol,
                                  client_order_id=f"revbot-{p.symbol}",
                                  id=f"oid-{p.symbol}")
            for p in self._positions
        ]

    def cancel_order(self, order_id):
        pass

    def close_position(self, symbol):
        if symbol in self._fail_close:
            raise RuntimeError("insufficient qty available")
        self.closed.append(symbol)
        return {"symbol": symbol}

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "stop", **kwargs}


def make_executor(positions=None, fail_close_symbols=(), fail_list=False):
    ex = AlpacaExecutor.__new__(AlpacaExecutor)  # bypass __init__: no real client
    ex.client = FakeClient(positions, fail_close_symbols, fail_list)
    ex._tif = "day"
    return ex


def test_liquidate_all_success_returns_true():
    ex = make_executor([FakePosition("AAPL", "10"), FakePosition("MSFT", "-5")])
    assert asyncio.run(main.liquidate_all_positions(ex)) is True
    assert set(ex.client.closed) == {"AAPL", "MSFT"}
    assert ex.client.orders == []          # no emergency stops needed


def test_liquidate_no_positions_returns_true():
    ex = make_executor([])
    assert asyncio.run(main.liquidate_all_positions(ex)) is True


def test_liquidate_position_fetch_failure_returns_false():
    # Needs a position we OWN, otherwise ownership short-circuits to "nothing to
    # flatten" and we never reach the position read this test is about. The
    # point is unchanged: a blind read must never be reported as success.
    ex = make_executor([FakePosition("AAPL", "10")], fail_list=True)
    assert asyncio.run(main.liquidate_all_positions(ex)) is False


def test_unknown_ownership_is_not_reported_as_success():
    # Ownership lookup down => we cannot tell what is ours => flatten nothing
    # AND return False so the caller retries, rather than stamping the session
    # reconciled. Fail-closed: the old code would have flattened the account.
    ex = make_executor([FakePosition("AAPL", "10")])
    def _boom(status=None, limit=None):
        raise RuntimeError("order lookup down")
    ex.client.list_orders = _boom
    assert asyncio.run(main.liquidate_all_positions(ex)) is False
    assert ex.client.closed == []
    assert ex.client.cancel_all_calls == 0


def test_fractional_position_is_still_closed():
    # int(float("0.5")) == 0 used to skip fractional positions entirely.
    ex = make_executor([FakePosition("AAPL", "0.5")])
    assert asyncio.run(main.liquidate_all_positions(ex)) is True
    assert ex.client.closed == ["AAPL"]


def test_failed_close_returns_false_and_rearms_emergency_stop():
    ex = make_executor(
        [FakePosition("AAPL", "10", current_price=200.0),
         FakePosition("TSLA", "-4", current_price=300.0)],
        fail_close_symbols={"AAPL", "TSLA"},
    )
    assert asyncio.run(main.liquidate_all_positions(ex)) is False

    stops = {o["symbol"]: o for o in ex.client.orders}
    assert set(stops) == {"AAPL", "TSLA"}
    # Long -> GTC sell stop BELOW the reference price.
    assert stops["AAPL"]["side"] == "sell"
    assert stops["AAPL"]["type"] == "stop"
    assert stops["AAPL"]["time_in_force"] == "gtc"
    assert stops["AAPL"]["stop_price"] < 200.0
    # Short -> GTC buy stop ABOVE the reference price.
    assert stops["TSLA"]["side"] == "buy"
    assert stops["TSLA"]["stop_price"] > 300.0
    assert stops["TSLA"]["qty"] == 4


def test_rearm_skips_subunit_fractional_tail():
    ex = make_executor()
    assert ex.rearm_protective_stop("AAPL", 0.4, "long", 100.0) is None
    assert ex.client.orders == []


def _run_reconcile(tmp_path, monkeypatch, executor):
    monkeypatch.setattr(main, "_RECONCILE_STAMP", tmp_path / "last_reconcile.date")
    monkeypatch.delenv("FLATTEN_CARRYOVER_ON_START", raising=False)
    asyncio.run(main.reconcile_carryover(executor))
    return (tmp_path / "last_reconcile.date").exists()


def test_reconcile_stamps_on_successful_flatten(tmp_path, monkeypatch):
    ex = make_executor([FakePosition("NVDA", "3")])
    assert _run_reconcile(tmp_path, monkeypatch, ex) is True
    assert ex.client.closed == ["NVDA"]


def test_reconcile_does_not_stamp_on_failed_flatten(tmp_path, monkeypatch):
    # The close fails after the brackets were cancelled: the day must NOT be
    # stamped reconciled, so the next cycle retries instead of abandoning the
    # orphan until tomorrow.
    ex = make_executor([FakePosition("NVDA", "3")], fail_close_symbols={"NVDA"})
    assert _run_reconcile(tmp_path, monkeypatch, ex) is False
