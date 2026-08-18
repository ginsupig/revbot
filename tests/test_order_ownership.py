"""revbot must only ever flatten what revbot opened.

``liquidate_all_positions`` called ``cancel_all_orders()`` (every open order in
the account) and then ``close_position`` on EVERY row from ``list_positions()``
-- no universe filter, no owner filter, nothing. It runs three times a session
(session end, the EOD window, and reconcile_carryover on the day's first cycle),
and SWING_OVERNIGHT_MODE=false leaves all three live.

It has never done damage only because revbot happens to be the sole occupant of
paper PA3MDZABNKUO. Point it at a shared account -- revbot's .env previously
held ACCOUNT 1 keys -- and the morning carryover sweep cancels every other bot's
brackets and liquidates their positions before the bell.

The fix is the pattern rotation ('rot-') and gapbot ('gapbot-') already use:
tag our own orders and scope every destructive action to that prefix.

The most important test here is the failure mode: when ownership CANNOT be
determined, revbot must flatten NOTHING. An API hiccup that silently reverts to
"close everything" would be worse than the original bug, because it would look
like it was working.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversion_bot.ownership import (COID_PREFIX, is_ours, owned_symbols,
                                     tag_client_order_id)


def _order(symbol, coid, status="filled", oid=None):
    return types.SimpleNamespace(symbol=symbol, client_order_id=coid,
                                 status=status, id=oid or f"id-{symbol}-{coid}")


class _Client:
    """Broker stub. ``raises`` simulates the order-history call failing."""

    def __init__(self, orders=None, raises=False):
        self._orders = orders or []
        self.raises = raises
        self.cancelled = []
        self.cancel_all_called = False

    def list_orders(self, status="open", limit=500):
        if self.raises:
            raise RuntimeError("broker unavailable")
        return list(self._orders)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def cancel_all_orders(self):
        self.cancel_all_called = True


class TestTagging(unittest.TestCase):
    def test_generated_id_carries_the_prefix(self):
        self.assertTrue(tag_client_order_id().startswith(COID_PREFIX))

    def test_ids_are_unique(self):
        self.assertNotEqual(tag_client_order_id(), tag_client_order_id())

    def test_a_caller_supplied_id_is_left_alone(self):
        self.assertEqual(tag_client_order_id("revbot-already"), "revbot-already")

    def test_recognises_our_own_orders(self):
        self.assertTrue(is_ours(_order("AAPL", "revbot-abc")))

    def test_rejects_foreign_and_missing_ids(self):
        self.assertFalse(is_ours(_order("AAPL", "gapbot-AAPL-2026")))
        self.assertFalse(is_ours(_order("AAPL", None)))
        self.assertFalse(is_ours(_order("AAPL", "")))
        self.assertFalse(is_ours(types.SimpleNamespace(symbol="AAPL")))


class TestOwnedSymbols(unittest.TestCase):
    def test_returns_only_symbols_we_opened(self):
        c = _Client([
            _order("AAPL", "revbot-1"),
            _order("NVDA", "gapbot-NVDA-20260817"),      # another bot
            _order("TSLA", "revbot-2"),
            _order("SPY", None),                          # a hand trade
        ])
        self.assertEqual(owned_symbols(c), {"AAPL", "TSLA"})

    def test_is_case_insensitive_on_symbol(self):
        self.assertEqual(owned_symbols(_Client([_order("aapl", "revbot-1")])),
                         {"AAPL"})

    def test_returns_None_not_empty_when_the_lookup_FAILS(self):
        # THE safety property. None means "unknown", which callers must treat as
        # "touch nothing". Returning an empty set here would read as "we own
        # nothing", which is indistinguishable from a real empty account and is
        # exactly how a silent revert to flatten-everything would happen.
        self.assertIsNone(owned_symbols(_Client(raises=True)))

    def test_no_orders_is_an_empty_set_not_None(self):
        self.assertEqual(owned_symbols(_Client([])), set())


class TestScopedCancel(unittest.TestCase):
    def test_cancels_only_our_open_orders(self):
        from reversion_bot.ownership import cancel_our_open_orders
        c = _Client([
            _order("AAPL", "revbot-1", status="open", oid="A1"),
            _order("NVDA", "gapbot-x", status="open", oid="N1"),
            _order("TSLA", "revbot-2", status="open", oid="T1"),
        ])
        cancel_our_open_orders(c)
        self.assertEqual(sorted(c.cancelled), ["A1", "T1"])

    def test_never_calls_the_account_wide_cancel(self):
        from reversion_bot.ownership import cancel_our_open_orders
        c = _Client([_order("AAPL", "revbot-1", status="open", oid="A1")])
        cancel_our_open_orders(c)
        self.assertFalse(c.cancel_all_called,
                         "account-wide cancel strips every other bot's stops")

    def test_a_failed_lookup_cancels_nothing(self):
        from reversion_bot.ownership import cancel_our_open_orders
        c = _Client(raises=True)
        cancel_our_open_orders(c)
        self.assertEqual(c.cancelled, [])
        self.assertFalse(c.cancel_all_called)




class _Pos:
    def __init__(self, symbol, qty, price=100.0):
        self.symbol = symbol
        self.qty = str(qty)
        self.current_price = price
        self.avg_entry_price = price


class _Executor:
    def __init__(self, client, positions):
        self.client = client
        self._positions = positions
        self.closed = []
        self.per_symbol_cancels = []

    def _cancel_orders_for_symbol(self, symbol):
        self.per_symbol_cancels.append(symbol)

    def rearm_protective_stop(self, *a, **k):
        pass


class _LiqClient(_Client):
    def __init__(self, orders=None, positions=None, raises=False):
        super().__init__(orders, raises)
        self._positions = positions or []
        self.closed = []

    def list_positions(self):
        return list(self._positions)

    def close_position(self, symbol):
        self.closed.append(symbol)


class TestLiquidateIsOwnerScoped(unittest.IsolatedAsyncioTestCase):
    """The EOD flatten must never touch a position revbot did not open."""

    def _setup(self, orders, positions, raises=False):
        c = _LiqClient(orders=orders, positions=positions, raises=raises)
        return c, _Executor(c, positions)

    async def test_closes_ours_and_leaves_foreign_positions_alone(self):
        import main
        c, ex = self._setup(
            orders=[_order("AAPL", "revbot-1"), _order("NVDA", "gapbot-NVDA-1")],
            positions=[_Pos("AAPL", 10), _Pos("NVDA", 5), _Pos("SPY", 3)],
        )
        await main.liquidate_all_positions(ex)
        self.assertEqual(c.closed, ["AAPL"],
                         "NVDA is gapbot's and SPY is untagged; neither is ours")

    async def test_never_calls_account_wide_cancel(self):
        import main
        c, ex = self._setup(orders=[_order("AAPL", "revbot-1")],
                            positions=[_Pos("AAPL", 10)])
        await main.liquidate_all_positions(ex)
        self.assertFalse(c.cancel_all_called)

    async def test_unknown_ownership_closes_NOTHING(self):
        # Broker lookup down -> ownership unknown -> touch nothing. Must not
        # fall back to flattening the account.
        import main
        c, ex = self._setup(orders=[], positions=[_Pos("AAPL", 10)], raises=True)
        ok = await main.liquidate_all_positions(ex)
        self.assertEqual(c.closed, [])
        self.assertFalse(c.cancel_all_called)
        self.assertFalse(ok, "unknown ownership is not a successful flatten")




class TestEntriesAreTagged(unittest.TestCase):
    """Tagging is load-bearing now, not cosmetic.

    liquidate_all_positions is fail-closed on ownership: an untagged entry is
    invisible to owned_symbols, so revbot would flatten NOTHING and carry the
    position overnight. Every order it submits must carry the prefix.
    """

    def _build(self, **kw):
        from reversion_bot.alpaca_py_client import _build_order_request
        base = dict(symbol="AAPL", qty=1, side="buy", time_in_force="day",
                    type="market")
        base.update(kw)
        return _build_order_request(base)

    def test_plain_market_order_is_tagged(self):
        req = self._build()
        self.assertTrue(str(req.client_order_id).startswith(COID_PREFIX))

    def test_bracket_entry_is_tagged(self):
        req = self._build(order_class="bracket",
                          take_profit={"limit_price": 110.0},
                          stop_loss={"stop_price": 90.0})
        self.assertTrue(str(req.client_order_id).startswith(COID_PREFIX))

    def test_two_orders_get_distinct_ids(self):
        self.assertNotEqual(self._build().client_order_id,
                            self._build().client_order_id)

    def test_an_explicit_id_is_respected(self):
        req = self._build(client_order_id="revbot-explicit-1")
        self.assertEqual(req.client_order_id, "revbot-explicit-1")


if __name__ == "__main__":
    unittest.main()
