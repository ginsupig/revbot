from __future__ import annotations

from alpaca_trade_api.rest import REST
from time import sleep
import logging
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError

from .models import PositionPlan


class AlpacaExecutor:
    def __init__(self, api_key: str, secret_key: str, exec_config):
        # Ensure exec_config has a base_url attribute
        if not hasattr(exec_config, 'base_url'):
            raise ValueError("ExecutionConfig must have a 'base_url' attribute.")

        # Order path SDK: opt-in `alpaca-py` adapter (maintained) vs the legacy
        # `alpaca_trade_api` REST client (EOL). Default off = legacy = unchanged.
        if bool(getattr(exec_config, "use_alpaca_py", False)):
            from .alpaca_py_client import AlpacaPyClient
            self.client = AlpacaPyClient(api_key, secret_key, exec_config.base_url)
        else:
            self.client = REST(api_key, secret_key, exec_config.base_url)
        self._order_ids = set()
        self._tif = getattr(exec_config, "tif", "day") or "day"
        # Marketable-limit entries cap entry slippage: the entry leg is priced a
        # few bps through the planned entry so it still crosses and fills, rather
        # than a plain market order that pays whatever the book offers.
        self._use_limit_entry = bool(getattr(exec_config, "use_limit_entry", False))
        self._limit_offset_bps = float(getattr(exec_config, "limit_entry_offset_bps", 8.0))
        self._tune_connection_pool(int(getattr(exec_config, "conn_pool_maxsize", 32)))

    def _entry_order_type(self, entry_price: float, side: str) -> dict:
        """Entry-leg order fields: a marketable limit when enabled, else market.

        For a buy the limit is placed ABOVE the planned entry, for a short sell
        BELOW it, by limit_entry_offset_bps — so the order crosses the spread and
        fills while capping worst-case slippage to roughly that offset.
        """
        if not self._use_limit_entry:
            return {"type": "market"}
        offset = self._limit_offset_bps / 10_000.0
        if side == "buy":
            limit_price = entry_price * (1.0 + offset)
        else:
            limit_price = entry_price * (1.0 - offset)
        return {"type": "limit", "limit_price": round(limit_price, 2)}

    def _tune_connection_pool(self, maxsize: int) -> None:
        """Widen the REST session's HTTP connection pool.

        The bot fans out one request per symbol (plus account/position calls)
        concurrently each cycle, which exceeds the requests default pool of 10
        and spams "Connection pool is full, discarding connection" warnings —
        and silently reopens sockets, adding latency. Mounting a larger adapter
        sizes the pool to the universe. Best-effort: never fatal if the SDK's
        session internals differ.
        """
        if maxsize <= 0:
            return
        session = getattr(self.client, "_session", None)
        if session is None:
            return
        try:
            adapter = HTTPAdapter(pool_connections=maxsize, pool_maxsize=maxsize)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Could not tune connection pool: %s", exc)

    def scan_symbols(self, min_price=5.0, min_dollar_volume=750000.0, max_count=30):
        """
        Fetches a universe of US Stocks & ETFs from Alpaca, best suited for most trading strategies.
        Filters for major US exchanges and tradable, active, easy-to-borrow stocks only.
        """
        import logging
        import time
        retry_attempts = 5
        retry_interval = 5
        logging.debug("Starting Alpaca US Stocks & ETFs universe scan_symbols method.")
        major_exchanges = {"NYSE", "NASDAQ", "ARCA", "BATS"}
        for attempt in range(retry_attempts):
            try:
                logging.debug(f"Fetching all active tradable assets (Attempt {attempt+1})")
                assets = self.client.list_assets(status='active', asset_class='us_equity')
                logging.debug(f"Fetched {len(assets)} assets.")
                tradable = [
                    a for a in assets
                    if a.tradable and a.easy_to_borrow and a.status == 'active'
                    and '.' not in a.symbol and a.symbol.isupper()
                    and a.exchange in major_exchanges
                ]
                logging.debug(f"Tradable, easy_to_borrow, active, major exchange, no dot, all uppercase: {len(tradable)}")
                # Sort by market cap if available, else by symbol
                tradable = sorted(tradable, key=lambda x: getattr(x, 'market_cap', 0), reverse=True)
                symbols = [a.symbol for a in tradable if float(getattr(a, 'min_price', 0) or 0) >= min_price][:max_count*2]
                filtered = []
                for sym in symbols:
                    try:
                        trade = self.client.get_latest_trade(sym)
                        if not trade or not hasattr(trade, 'price'):
                            continue
                        price = float(trade.price)
                        if price < min_price:
                            continue
                        # Use 1-day bar volume for a realistic dollar-volume estimate
                        bars = self.client.get_bars(sym, '1Day', limit=5).df
                        if bars is None or len(bars) == 0:
                            continue
                        avg_dollar_vol = float(
                            (bars['close'] * bars['volume']).mean()
                        )
                        if avg_dollar_vol >= min_dollar_volume:
                            filtered.append(sym)
                    except Exception as e:
                        logging.warning(f"Quote/bar fetch failed for {sym}: {e}")
                        continue
                    if len(filtered) >= max_count:
                        break
                logging.info(f"Alpaca US Stocks & ETFs universe: {filtered[:max_count]}")
                return filtered[:max_count]
            except HTTPError as e:
                if 'rate limit exceeded' in str(e):
                    logging.error("Rate limit exceeded. Retrying in %d seconds...", retry_interval)
                    time.sleep(retry_interval)
                    retry_interval *= 2
                else:
                    logging.exception("HTTPError encountered: %s", e)
                    raise
            except Exception as e:
                logging.error(f"Unexpected error in scan_symbols: {e}")
                time.sleep(retry_interval)
                retry_interval *= 2
        logging.error("Failed to fetch Alpaca US Stocks & ETFs universe after %d attempts", retry_attempts)
        return []

    def open_long_bracket(self, symbol: str, plan: PositionPlan):
        symbol = symbol.strip().upper()
        self._validate_plan(plan)

        clock = self.client.get_clock()
        if not clock.is_open:
            raise RuntimeError("Market is closed. Cannot submit order.")

        if self.has_open_long_position(symbol):
            raise RuntimeError(f"Position for {symbol} already exists.")

        if self.has_open_order(symbol):
            raise RuntimeError(f"Working order already exists for {symbol}.")

        order_key = self._order_key(symbol, plan)
        if order_key in self._order_ids:
            raise RuntimeError(f"Duplicate order detected for {order_key}.")

        order_data = {
            "symbol": symbol,
            "qty": plan.qty,
            "side": "buy",
            "time_in_force": self._tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(plan.target_price, 2)},
            "stop_loss": {"stop_price": round(plan.stop_price, 2)},
            **self._entry_order_type(plan.entry_price, "buy"),
        }

        response = self.client.submit_order(**order_data)
        self._order_ids.add(order_key)
        return response

    def open_short_bracket(self, symbol: str, plan: PositionPlan):
        symbol = symbol.strip().upper()
        self._validate_short_plan(plan)

        clock = self.client.get_clock()
        if not clock.is_open:
            raise RuntimeError("Market is closed. Cannot submit order.")

        if self.has_open_short_position(symbol):
            raise RuntimeError(f"Short position for {symbol} already exists.")

        if self.has_open_order(symbol):
            raise RuntimeError(f"Working order already exists for {symbol}.")

        order_key = self._order_key(symbol, plan)
        if order_key in self._order_ids:
            raise RuntimeError(f"Duplicate order detected for {order_key}.")

        order_data = {
            "symbol": symbol,
            "qty": plan.qty,
            "side": "sell",
            "time_in_force": self._tif,
            "order_class": "bracket",
            # On a short, profit is taken *below* entry and the stop sits *above*.
            "take_profit": {"limit_price": round(plan.target_price, 2)},
            "stop_loss": {"stop_price": round(plan.stop_price, 2)},
            **self._entry_order_type(plan.entry_price, "sell"),
        }

        response = self.client.submit_order(**order_data)
        self._order_ids.add(order_key)
        return response

    def submit_order(self, candidate: dict):
        symbol = candidate["symbol"]
        pd = candidate["position_plan"]
        plan = PositionPlan(
            qty=pd["qty"],
            entry_price=pd["entry_price"],
            stop_price=pd["stop_price"],
            target_price=pd["target_price"],
            risk_per_share=pd["risk_per_share"],
            reward_per_share=pd["reward_per_share"],
            rr_ratio=pd["rr_ratio"],
            position_value=pd["position_value"],
            max_account_risk=pd["max_account_risk"],
            side=pd.get("side", "long"),
        )
        # Prefer the candidate's explicit go_short flag; fall back to plan side.
        if candidate.get("go_short") or plan.side == "short":
            return self.open_short_bracket(symbol, plan)
        return self.open_long_bracket(symbol, plan)

    def replace_limit_order(self, order_id: str, new_limit_price: float):
        if not order_id:
            raise ValueError("order_id is required")
        if new_limit_price <= 0:
            raise ValueError("new_limit_price must be > 0")
        return self.client.replace_order(
            order_id=order_id,
            limit_price=round(float(new_limit_price), 2)
        )

    def get_positions(self):
        return self.client.list_positions()

    def close_long(self, symbol: str):
        """Market-close a LONG position, cancelling ITS working orders (the
        bracket legs) first, per-symbol.

        Used by the breakout exit to take profit on a run before the fixed
        bracket target. The open bracket legs (stop + take-profit) reserve the
        shares as ``held_for_orders``; close_position alone is then rejected with
        "insufficient qty available for order (requested: N, available: 0)" and
        the position never closes — the breakout exit just re-fires the same
        error every cycle. Cancel the symbol's working orders to free the held
        qty, then liquidate. Per-symbol throughout, so other names' brackets are
        untouched (unlike the global cancel_all_orders the EOD path uses).
        """
        symbol = symbol.strip().upper()
        self._cancel_orders_for_symbol(symbol)
        # Cancelled qty can take a beat to free server-side, so the first close
        # may still see held shares; retry a few times before giving up.
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                return self.client.close_position(symbol)
            except Exception as e:  # noqa: BLE001 - retried/propagated below
                last_exc = e
                sleep(0.5)
        raise last_exc

    def _cancel_orders_for_symbol(self, symbol: str) -> None:
        """Best-effort cancel of every working order for one symbol.

        Frees shares the open bracket legs hold so close_position can liquidate.
        Never raises: a cancel hiccup must not block the close path — the close
        retry (and the next cycle) will recover.
        """
        symbol = symbol.strip().upper()
        try:
            orders = self.client.list_orders(status="open", limit=500)
        except Exception as e:  # noqa: BLE001 - logged, close still attempted
            logging.warning("close_long: order list failed for %s: %s", symbol, e)
            return
        for o in orders:
            if str(getattr(o, "symbol", "")).upper() != symbol:
                continue
            order_id = getattr(o, "id", None)
            if order_id is None:
                continue
            try:
                self.client.cancel_order(order_id)
            except Exception as e:  # noqa: BLE001 - one leg failing must not block
                logging.warning(
                    "close_long: cancel order %s for %s failed: %s", order_id, symbol, e
                )

    def open_order_symbols(self) -> set:
        """Symbols with a WORKING (unfilled/active) order right now.

        Closes the double-submit gap: a bracket whose entry leg hasn't filled is
        invisible to list_positions, so without this a second order could stack
        on the same name across cycles (worse with marketable-limit entries that
        sit unfilled). Raises on API error so callers fail CLOSED (skip the
        entry) rather than risk a duplicate.
        """
        orders = self.client.list_orders(status="open", limit=500)
        return {str(o.symbol).upper() for o in orders if getattr(o, "symbol", None)}

    def has_open_order(self, symbol: str) -> bool:
        return symbol.strip().upper() in self.open_order_symbols()

    def has_open_long_position(self, symbol: str) -> bool:
        positions = self.get_positions()
        return any(str(p.symbol).upper() == symbol and float(p.qty) > 0 for p in positions)

    def has_open_short_position(self, symbol: str) -> bool:
        positions = self.get_positions()
        return any(str(p.symbol).upper() == symbol and float(p.qty) < 0 for p in positions)

    def has_open_position(self, symbol: str) -> bool:
        """True if any position (long or short) is open for the symbol.

        Used as the pre-entry guard so the bot never stacks a new trade on top
        of an existing one in either direction.
        """
        positions = self.get_positions()
        return any(str(p.symbol).upper() == symbol and float(p.qty) != 0 for p in positions)

    @staticmethod
    def _validate_plan(plan: PositionPlan) -> None:
        if plan.qty <= 0:
            raise ValueError("Quantity must be positive.")
        if plan.entry_price <= 0:
            raise ValueError("Entry price must be positive.")
        if plan.stop_price <= 0 or plan.target_price <= 0:
            raise ValueError("Stop and target prices must be positive.")
        if plan.stop_price >= plan.entry_price:
            raise ValueError("Stop price must be below entry for long bracket.")
        if plan.target_price <= plan.entry_price:
            raise ValueError("Target price must be above entry for long bracket.")

    @staticmethod
    def _validate_short_plan(plan: PositionPlan) -> None:
        if plan.qty <= 0:
            raise ValueError("Quantity must be positive.")
        if plan.entry_price <= 0:
            raise ValueError("Entry price must be positive.")
        if plan.stop_price <= 0 or plan.target_price <= 0:
            raise ValueError("Stop and target prices must be positive.")
        if plan.stop_price <= plan.entry_price:
            raise ValueError("Stop price must be above entry for short bracket.")
        if plan.target_price >= plan.entry_price:
            raise ValueError("Target price must be below entry for short bracket.")

    @staticmethod
    def _order_key(symbol: str, plan: PositionPlan) -> str:
        return "|".join([
            symbol,
            f"{plan.qty}",
            f"{plan.entry_price:.2f}",
            f"{plan.stop_price:.2f}",
            f"{plan.target_price:.2f}",
        ])
