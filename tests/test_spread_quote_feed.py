"""The spread gate's quote feed (SPREAD_QUOTE_FEED).

On the free data plan the default quote is IEX's own book, far wider than the
NBBO, so the fleet sets SPREAD_QUOTE_FEED=delayed_sip. Unset must keep the old
call exactly (no feed argument) for both the legacy REST client and the adapter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from alpaca.data.enums import DataFeed
from tests.test_alpaca_py_client import make_adapter


class _LegacyClient:
    def __init__(self):
        self.calls = []

    def get_latest_quote(self, symbol, feed=None):
        self.calls.append((symbol, feed))
        return "QUOTE"


def test_unset_feed_keeps_default_call(monkeypatch):
    monkeypatch.delenv("SPREAD_QUOTE_FEED", raising=False)
    c = _LegacyClient()
    assert main.fetch_spread_quote(c, "META") == "QUOTE"
    assert c.calls == [("META", None)]


def test_feed_env_is_passed_through(monkeypatch):
    monkeypatch.setenv("SPREAD_QUOTE_FEED", " Delayed_SIP ")
    c = _LegacyClient()
    main.fetch_spread_quote(c, "META")
    assert c.calls == [("META", "delayed_sip")]


def test_adapter_maps_feed_to_request(monkeypatch):
    a = make_adapter()
    seen = []

    def latest_quote(req):
        seen.append(req)
        return {req.symbol_or_symbols: "Q"}

    a._data.get_stock_latest_quote = latest_quote
    assert a.get_latest_quote("META", feed="delayed_sip") == "Q"
    assert a.get_latest_quote("META") == "Q"
    assert seen[0].feed == DataFeed.DELAYED_SIP
    assert seen[1].feed is None


def test_spread_from_nbbo_quote_passes_gate_iex_quote_fails():
    import pandas as pd

    class Q:
        def __init__(self, bid, ask):
            self.bid_price, self.ask_price = bid, ask

    bars = pd.DataFrame({"close": [637.0, 637.0]})
    # 11-13-style META quotes: delayed SIP 637.00/637.05, IEX 607.23/671.85.
    sip = main.attach_quote_spread(bars, Q(637.00, 637.05))["spread_bps"].iloc[-1]
    iex = main.attach_quote_spread(bars, Q(607.23, 671.85))["spread_bps"].iloc[-1]
    assert sip < 40 < iex
