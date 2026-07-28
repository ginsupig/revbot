"""Opt-in adapter: run the live order path on the maintained `alpaca-py` SDK.

The proven trading logic in ``execution.py`` was written against the legacy,
now-EOL ``alpaca_trade_api`` REST client. Rather than rewrite that logic (the
LIVE MONEY PATH), this module exposes an ``AlpacaPyClient`` whose method surface
mirrors the exact subset of the legacy ``REST`` client that the bot actually
calls, backed by ``alpaca-py``'s ``TradingClient`` + ``StockHistoricalDataClient``.

It is wired in behind ``ExecutionConfig.use_alpaca_py`` (env ``USE_ALPACA_PY``),
default OFF — so the default build is byte-for-byte the legacy path. Flip it on
to paper-test the new SDK before it ever becomes the default.

The translation surface (legacy method -> alpaca-py call):
  get_clock()                         -> TradingClient.get_clock()
  get_account()                       -> TradingClient.get_account()
  list_positions()                    -> TradingClient.get_all_positions()
  list_orders(status, limit)          -> TradingClient.get_orders(GetOrdersRequest)
  submit_order(**bracket_or_market)   -> TradingClient.submit_order(OrderRequest)
  cancel_all_orders()                 -> TradingClient.cancel_orders()
  cancel_order(order_id)              -> TradingClient.cancel_order_by_id(order_id)
  list_assets(status, asset_class)    -> TradingClient.get_all_assets(GetAssetsRequest)
  get_latest_trade(symbol)            -> StockHistoricalDataClient.get_stock_latest_trade(...)
  get_bars(symbol, tf, limit).df      -> StockHistoricalDataClient.get_stock_bars(...).df
  replace_order(order_id, limit_price)-> TradingClient.replace_order_by_id(ReplaceOrderRequest)

alpaca-py's status/exchange enums subclass ``str``, so the legacy string
comparisons in ``scan_symbols`` (``a.status == 'active'``, ``a.exchange in {...}``)
keep working against the returned ``Asset`` objects unchanged.
"""
from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
    GetAssetsRequest,
    ReplaceOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
    AssetStatus,
    AssetClass,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame


_ORDER_SIDE = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
_TIME_IN_FORCE = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
    "fok": TimeInForce.FOK,
    "opg": TimeInForce.OPG,
    "cls": TimeInForce.CLS,
}
# The bot only ever asks for daily bars (scan_symbols dollar-volume estimate);
# map the handful of legacy timeframe strings it might pass through.
_TIMEFRAME = {
    "1day": TimeFrame.Day,
    "1hour": TimeFrame.Hour,
    "1min": TimeFrame.Minute,
}


def _build_order_request(kwargs: dict):
    """Translate a legacy ``submit_order(**kwargs)`` dict into an alpaca-py
    ``OrderRequest``.

    Handles both shapes the bot submits:
      * bracket entries (``order_class="bracket"`` with take_profit/stop_loss,
        ``type`` either "market" or marketable "limit"), and
      * plain market exits (EOD liquidation: ``type="market"``, no order_class).
    """
    side = _ORDER_SIDE[str(kwargs["side"]).lower()]
    tif = _TIME_IN_FORCE[str(kwargs.get("time_in_force", "day")).lower()]

    common: dict = {
        "symbol": kwargs["symbol"],
        "qty": kwargs["qty"],
        "side": side,
        "time_in_force": tif,
    }

    if kwargs.get("order_class") == "bracket":
        common["order_class"] = OrderClass.BRACKET
        take_profit = kwargs.get("take_profit")
        stop_loss = kwargs.get("stop_loss")
        if take_profit:
            common["take_profit"] = TakeProfitRequest(
                limit_price=take_profit["limit_price"]
            )
        if stop_loss:
            sl_fields = {"stop_price": stop_loss["stop_price"]}
            if stop_loss.get("limit_price") is not None:
                sl_fields["limit_price"] = stop_loss["limit_price"]
            common["stop_loss"] = StopLossRequest(**sl_fields)

    if str(kwargs.get("type", "market")).lower() == "limit":
        return LimitOrderRequest(limit_price=kwargs["limit_price"], **common)
    return MarketOrderRequest(**common)


class _BarsResult:
    """Mirrors the legacy ``get_bars(...).df`` accessor so callers are unchanged."""

    def __init__(self, df):
        self._df = df

    @property
    def df(self):
        return self._df


class AlpacaPyClient:
    """Legacy-REST-shaped facade over alpaca-py's Trading + Data clients."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "",
        *,
        trading_client=None,
        data_client=None,
    ):
        paper = "paper" in (base_url or "").lower()
        self._trading = trading_client or TradingClient(
            api_key, secret_key, paper=paper
        )
        # Market data is endpoint-agnostic (same host for paper/live keys).
        self._data = data_client or StockHistoricalDataClient(api_key, secret_key)

    # --- account / clock ----------------------------------------------------
    def get_account(self):
        return self._trading.get_account()

    def get_clock(self):
        return self._trading.get_clock()

    # --- positions / orders -------------------------------------------------
    def list_positions(self):
        return self._trading.get_all_positions()

    def list_orders(self, status: str = "open", limit: int = 500):
        status_enum = (
            QueryOrderStatus.OPEN
            if str(status).lower() == "open"
            else QueryOrderStatus(str(status).lower())
        )
        return self._trading.get_orders(
            filter=GetOrdersRequest(status=status_enum, limit=limit)
        )

    def submit_order(self, **kwargs):
        return self._trading.submit_order(_build_order_request(kwargs))

    def cancel_all_orders(self):
        return self._trading.cancel_orders()

    def cancel_order(self, order_id: str):
        # Cancel a single working order (mirrors the legacy REST cancel_order):
        # used to drop a symbol's bracket legs before liquidating its position.
        return self._trading.cancel_order_by_id(order_id)

    def close_position(self, symbol: str):
        # Market-closes the position and cancels its related open orders.
        return self._trading.close_position(symbol)

    def replace_order(self, order_id: str, limit_price: float = None,
                      stop_price: float = None):
        """Replace a working order's limit and/or stop price.

        Mirrors the legacy REST signature: callers pass keyword prices. The
        old version accepted ONLY limit_price, so the trailing stop's
        replace_order(order_id=..., stop_price=...) raised TypeError on every
        trail update under USE_ALPACA_PY — silently swallowed upstream, which
        left every position riding its fixed bracket with the validated
        trailing edge disabled.
        """
        if limit_price is None and stop_price is None:
            raise ValueError("replace_order requires limit_price and/or stop_price")
        fields = {}
        if limit_price is not None:
            fields["limit_price"] = limit_price
        if stop_price is not None:
            fields["stop_price"] = stop_price
        return self._trading.replace_order_by_id(
            order_id, ReplaceOrderRequest(**fields)
        )

    # --- universe / market data --------------------------------------------
    def list_assets(self, status: str = "active", asset_class: str = "us_equity"):
        return self._trading.get_all_assets(
            filter=GetAssetsRequest(
                status=AssetStatus(str(status).lower()),
                asset_class=AssetClass(str(asset_class).lower()),
            )
        )

    def get_latest_trade(self, symbol: str):
        resp = self._data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol)
        )
        # alpaca-py returns {symbol: Trade}; legacy returned the Trade directly.
        return resp[symbol] if isinstance(resp, dict) else resp

    def get_bars(self, symbol: str, timeframe: str, limit: int = 5):
        tf = _TIMEFRAME.get(str(timeframe).lower(), TimeFrame.Day)
        barset = self._data.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
        )
        return _BarsResult(barset.df)
