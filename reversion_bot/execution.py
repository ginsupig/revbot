from __future__ import annotations

import os
from uuid import uuid4

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
        self._tif = getattr(exec_config, "tif", "day") or "day"
        # Warn at construction if the configured TIF can't ride a bracket (the
        # coercion itself happens per-order via the bracket_tif property).
        self._bracket_safe_tif(self._tif)
        # Marketable-limit entries cap entry slippage: the entry leg is priced a
        # few bps through the planned entry so it still crosses and fills, rather
        # than a plain market order that pays whatever the book offers.
        self._use_limit_entry = bool(getattr(exec_config, "use_limit_entry", False))
        self._limit_offset_bps = float(getattr(exec_config, "limit_entry_offset_bps", 8.0))
        self._tune_connection_pool(int(getattr(exec_config, "conn_pool_maxsize", 32)))
        try:
            http_timeout = float(os.getenv("ALPACA_HTTP_TIMEOUT", "15"))
        except (TypeError, ValueError):
            http_timeout = 15.0
        self._apply_http_timeout(http_timeout)

    @property
    def bracket_tif(self) -> str:
        """TIF to put on bracket orders.

        Alpaca accepts only day/gtc on brackets. Swing mode documents pairing
        SWING_OVERNIGHT_MODE with TIF=cls (market-on-close), which the broker
        rejects with a 422 on the bracket parent — silently producing zero
        entries every session. The configured TIF still governs any plain-order
        path; only the bracket legs are coerced.
        """
        return self._bracket_safe_tif(getattr(self, "_tif", "day"))

    @staticmethod
    def _bracket_safe_tif(tif: str) -> str:
        """Map a configured TIF onto one a bracket order can carry.

        Alpaca accepts only ``day`` and ``gtc`` on bracket orders. Anything else
        (notably ``cls``/``opg``) is coerced to ``day`` with a warning so the
        entry is still submitted rather than 422-ing per candidate.
        """
        normalized = str(tif or "day").strip().lower()
        if normalized in ("day", "gtc"):
            return normalized
        logging.warning(
            "TIF %r is not valid on bracket orders (Alpaca allows day/gtc only) "
            "— using 'day' for the bracket legs. Entry timing is still governed "
            "by the MOC entry window when swing mode is on.",
            tif,
        )
        return "day"

    def _apply_http_timeout(self, timeout: float) -> None:
        """Bound every trading-API HTTP call with a default timeout.

        Neither SDK sets one, so a hung get_clock / list_positions /
        submit_order blocks its worker thread FOREVER — freezing the whole
        poll loop (no EOD flatten, no trailing-stop management) with a stale
        heartbeat as the only symptom. ALPACA_HTTP_TIMEOUT already bounded the
        market-data fetchers; this applies the same guard to the trading
        client. Wraps the underlying requests.Session so a timeout is supplied
        only when the caller didn't pass one. Best-effort: if the SDK's
        session internals change this logs and degrades to prior behavior.
        """
        if timeout is None or timeout <= 0:
            return
        # Legacy REST keeps a requests.Session at _session; the alpaca-py
        # wrapper nests its TradingClient at _trading (same _session attr).
        session = getattr(self.client, "_session", None)
        if session is None:
            inner = getattr(self.client, "_trading", None)
            session = getattr(inner, "_session", None)
        if session is None or not callable(getattr(session, "request", None)):
            logging.warning("Trading-API HTTP timeout not applied (no session found).")
            return
        original = session.request

        def _request_with_timeout(method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout
            return original(method, url, **kwargs)

        session.request = _request_with_timeout

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

    # Every order the bot submits carries this client_order_id prefix, so the
    # EOD cancel can target ONLY revbot's orders instead of nuking manual /
    # other-strategy orders sharing the account.
    CLIENT_ORDER_PREFIX = "revbot-"

    @classmethod
    def _make_client_order_id(cls) -> str:
        return f"{cls.CLIENT_ORDER_PREFIX}{uuid4().hex[:20]}"

    def cancel_bot_orders(self, position_symbols=()) -> int:
        """Cancel the bot's own working orders: anything tagged with our
        client_order_id prefix, plus any order on a symbol we hold a position
        in (covers pre-tagging legacy brackets). Returns the cancel count;
        never raises — the per-symbol cancel in the flatten loop is the
        second line of defense."""
        held = {str(s).upper() for s in position_symbols}
        cancelled = 0
        try:
            orders = self.list_open_orders()
        except Exception as e:  # noqa: BLE001
            logging.warning("cancel_bot_orders: order list failed: %s", e)
            return 0
        for o in orders:
            coid = str(getattr(o, "client_order_id", "") or "")
            sym = str(getattr(o, "symbol", "") or "").upper()
            if not (coid.startswith(self.CLIENT_ORDER_PREFIX) or sym in held):
                continue
            oid = getattr(o, "id", None)
            if not oid:
                continue
            try:
                self.client.cancel_order(oid)
                cancelled += 1
            except Exception as e:  # noqa: BLE001
                logging.warning("cancel_bot_orders: cancel failed for %s: %s", sym, e)
        return cancelled

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
                # Alpaca Asset objects carry NO market_cap attribute, so the
                # old "sort by market cap" was a silent no-op (every key 0):
                # the universe was whatever the API returned first — roughly
                # the first N alphabetical liquid names, and the first match
                # won regardless of how liquid it was. Rank by MEASURED
                # liquidity instead: scan a wider (deterministic) candidate
                # pool and keep the max_count highest average-dollar-volume
                # names. The pool slice is still alphabetical — for a curated
                # universe use TRADE_WATCHLIST / TRADE_ALLOWLIST.
                symbols = sorted(a.symbol for a in tradable)[:max_count * 4]
                scored = []
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
                            scored.append((sym, avg_dollar_vol))
                    except Exception as e:
                        logging.warning(f"Quote/bar fetch failed for {sym}: {e}")
                        continue
                scored.sort(key=lambda t: (-t[1], t[0]))
                universe = [s for s, _ in scored[:max_count]]
                logging.info(
                    f"Alpaca universe (top {len(universe)} of {len(scored)} "
                    f"candidates by avg dollar volume): {universe}"
                )
                return universe
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

        order_data = {
            "symbol": symbol,
            "qty": plan.qty,
            "side": "buy",
            "time_in_force": self.bracket_tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(plan.target_price, 2)},
            "stop_loss": {"stop_price": round(plan.stop_price, 2)},
            "client_order_id": self._make_client_order_id(),
            **self._entry_order_type(plan.entry_price, "buy"),
        }

        return self.client.submit_order(**order_data)

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

        order_data = {
            "symbol": symbol,
            "qty": plan.qty,
            "side": "sell",
            "time_in_force": self.bracket_tif,
            "order_class": "bracket",
            # On a short, profit is taken *below* entry and the stop sits *above*.
            "take_profit": {"limit_price": round(plan.target_price, 2)},
            "stop_loss": {"stop_price": round(plan.stop_price, 2)},
            "client_order_id": self._make_client_order_id(),
            **self._entry_order_type(plan.entry_price, "sell"),
        }

        return self.client.submit_order(**order_data)

    def submit_order(self, candidate: dict):
        symbol = candidate["symbol"]
        plan_data = candidate["position_plan"]
        plan = PositionPlan(
            qty=plan_data["qty"],
            entry_price=plan_data["entry_price"],
            stop_price=plan_data["stop_price"],
            target_price=plan_data["target_price"],
            risk_per_share=plan_data["risk_per_share"],
            reward_per_share=plan_data["reward_per_share"],
            rr_ratio=plan_data["rr_ratio"],
            position_value=plan_data["position_value"],
            max_account_risk=plan_data["max_account_risk"],
            side=plan_data.get("side", "long"),
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

    def replace_stop_order(self, order_id: str, new_stop_price: float):
        if not order_id:
            raise ValueError("order_id is required")
        if new_stop_price <= 0:
            raise ValueError("new_stop_price must be > 0")
        return self.client.replace_order(
            order_id=order_id,
            stop_price=round(float(new_stop_price), 2),
        )

    def rearm_protective_stop(self, symbol: str, qty: float, side: str,
                              reference_price: float, stop_pct: float = 0.02):
        """Emergency backstop for a position whose close FAILED after its
        bracket legs were already cancelled: submit a plain GTC stop so the
        position is never carried with zero protection. Long -> sell stop
        below the reference price; short -> buy stop above it. GTC so it
        survives the overnight session; the next liquidation attempt's
        per-symbol cancel clears it before retrying the close, so it never
        blocks a retry. Fractional tails under one share can't carry a GTC
        stop (broker limitation) and are skipped. Never raises."""
        try:
            symbol = symbol.strip().upper()
            whole_qty = int(abs(float(qty)))
            px = float(reference_price)
            if whole_qty < 1 or px <= 0:
                logging.error(
                    "Cannot re-arm stop for %s (qty=%s px=%s) — position is UNPROTECTED.",
                    symbol, qty, reference_price,
                )
                return None
            # The close may have failed *because the position no longer exists*
            # (closed manually, or its stop leg filled just before the cancel).
            # Arming a GTC stop then leaves a naked resting order that nothing
            # owns: a gap through it sells shares the account doesn't hold.
            # Re-check against the broker before arming.
            try:
                if not self._position_exists(symbol):
                    logging.warning(
                        "Skipping emergency stop for %s: broker reports no open "
                        "position (close likely already completed).", symbol,
                    )
                    return None
            except Exception as e:  # noqa: BLE001 - verification is best-effort
                logging.warning(
                    "Could not verify %s position before re-arming stop (%s) — "
                    "arming anyway to avoid leaving it unprotected.", symbol, e,
                )
            if str(side).lower() == "long":
                order_side = "sell"
                stop_price = px * (1.0 - float(stop_pct))
            else:
                order_side = "buy"
                stop_price = px * (1.0 + float(stop_pct))
            order = self.client.submit_order(
                symbol=symbol,
                qty=whole_qty,
                side=order_side,
                type="stop",
                stop_price=round(stop_price, 2),
                time_in_force="gtc",
                client_order_id=self._make_client_order_id(),
            )
            logging.warning(
                "Re-armed emergency %s stop for %s: qty=%d stop=%.2f (close failed after bracket cancel)",
                order_side, symbol, whole_qty, stop_price,
            )
            return order
        except Exception as e:  # noqa: BLE001 - best-effort; failure is logged loudly
            logging.error("FAILED to re-arm protective stop for %s: %s — position is UNPROTECTED.", symbol, e)
            return None

    def open_stop_leg(self, symbol: str):
        """The symbol's open protective STOP order (the bracket's stop leg), as
        ``{"id", "stop_price", "qty"}`` — or None. Identified by carrying a
        stop_price (the take-profit leg is a plain limit). Never raises."""
        symbol = symbol.strip().upper()
        try:
            orders = self.client.list_orders(status="open", limit=500)
        except Exception as e:  # noqa: BLE001 - best-effort; caller treats None as "skip"
            logging.warning("open_stop_leg: list_orders failed for %s: %s", symbol, e)
            return None
        for o in orders:
            if str(getattr(o, "symbol", "")).upper() != symbol:
                continue
            sp = getattr(o, "stop_price", None)
            if sp is None:
                continue
            return {
                "id": getattr(o, "id", None),
                "stop_price": float(sp),
                "qty": abs(float(getattr(o, "qty", 0) or 0)),
            }
        return None

    def update_trailing_stop(self, symbol, entry_price, stop_distance, high_water,
                             last_price, stop_mult, trail_mult):
        """Best-effort: raise the symbol's stop leg toward the trailing stop.

        Returns the new stop price if it was moved up, else None. Never raises —
        a trailing-stop hiccup must not break the poll loop; the fixed bracket
        stop still protects the position and the next cycle retries.
        """
        from .trailing import trailing_stop_price, should_raise_stop
        try:
            leg = self.open_stop_leg(symbol)
            if not leg or not leg.get("id"):
                return None
            desired = trailing_stop_price(entry_price, stop_distance, high_water,
                                          stop_mult, trail_mult)
            if not should_raise_stop(leg["stop_price"], desired, last_price):
                return None
            self.replace_stop_order(leg["id"], desired)
            return round(float(desired), 2)
        except Exception as e:  # noqa: BLE001 - guarded; the fixed stop remains
            logging.warning("update_trailing_stop failed for %s: %s", symbol, e)
            return None

    def update_trailing_stop_short(self, symbol, entry_price, stop_distance,
                                   low_water, last_price, stop_mult, trail_mult):
        """Best-effort: lower the symbol's stop leg toward the short trailing stop.

        Mirror of update_trailing_stop for shorts: ratchets the stop DOWN as
        the short position profits (price falls). Returns the new stop price
        if it was moved down, else None. Never raises.
        """
        from .trailing import trailing_stop_price_short, should_lower_stop
        try:
            leg = self.open_stop_leg(symbol)
            if not leg or not leg.get("id"):
                return None
            desired = trailing_stop_price_short(entry_price, stop_distance,
                                                low_water, stop_mult, trail_mult)
            if not should_lower_stop(leg["stop_price"], desired, last_price):
                return None
            self.replace_stop_order(leg["id"], desired)
            return round(float(desired), 2)
        except Exception as e:  # noqa: BLE001 - guarded; the fixed stop remains
            logging.warning("update_trailing_stop_short failed for %s: %s", symbol, e)
            return None

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

    def close_short(self, symbol: str):
        """Market-cover a SHORT position, cancelling ITS working orders first.

        Mirror of close_long: cancel the bracket legs (which hold qty), then
        submit a buy-to-cover via close_position. Per-symbol so other names'
        brackets are untouched.
        """
        symbol = symbol.strip().upper()
        self._cancel_orders_for_symbol(symbol)
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
            logging.warning("close_position: order list failed for %s: %s", symbol, e)
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
                    "close_position: cancel order %s for %s failed: %s", order_id, symbol, e
                )

    def list_open_orders(self):
        """All WORKING (unfilled/active) orders. Raises on API error so callers
        fail CLOSED rather than act on unknown state."""
        return self.client.list_orders(status="open", limit=500)

    def open_order_symbols(self) -> set:
        """Symbols with a WORKING (unfilled/active) order right now.

        Closes the double-submit gap: a bracket whose entry leg hasn't filled is
        invisible to list_positions, so without this a second order could stack
        on the same name across cycles (worse with marketable-limit entries that
        sit unfilled). Raises on API error so callers fail CLOSED (skip the
        entry) rather than risk a duplicate.
        """
        return {str(o.symbol).upper() for o in self.list_open_orders()
                if getattr(o, "symbol", None)}

    def has_open_order(self, symbol: str) -> bool:
        return symbol.strip().upper() in self.open_order_symbols()

    def has_open_long_position(self, symbol: str) -> bool:
        positions = self.get_positions()
        symbol = symbol.strip().upper()
        return any(str(p.symbol).upper() == symbol and float(p.qty) > 0 for p in positions)

    def has_open_short_position(self, symbol: str) -> bool:
        positions = self.get_positions()
        symbol = symbol.strip().upper()
        return any(str(p.symbol).upper() == symbol and float(p.qty) < 0 for p in positions)

    def has_open_position(self, symbol: str) -> bool:
        """True if any position (long or short) is open for the symbol.

        Used as the pre-entry guard so the bot never stacks a new trade on top
        of an existing one in either direction.
        """
        positions = self.get_positions()
        symbol = symbol.strip().upper()
        return any(str(p.symbol).upper() == symbol and float(p.qty) != 0 for p in positions)

    def latest_spread_bps(self, symbol: str) -> float | None:
        """Current quoted spread in basis points, or None if unavailable.

        The engine's ``Spread_Too_Wide`` guard reads a ``spread_bps`` bar column
        that NOTHING in the live path populates (bar fetches return OHLCV only),
        so ``MAX_SPREAD_BPS`` was a permanent no-op: a 60bps-spread name passed
        is_market_safe and marketable-limit entries filled deep into the book.
        This measures it from the live quote at submit time instead. Returns
        None (rather than raising) when the quote is missing or degenerate, so
        callers can decide whether to fail open.
        """
        try:
            quote = self.client.get_latest_quote(symbol.strip().upper())
        except Exception as e:  # noqa: BLE001
            logging.warning("Spread check: quote fetch failed for %s: %s", symbol, e)
            return None
        bid = getattr(quote, "bid_price", None) or getattr(quote, "bp", None)
        ask = getattr(quote, "ask_price", None) or getattr(quote, "ap", None)
        try:
            bid = float(bid)
            ask = float(ask)
        except (TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid * 10_000.0

    def open_position_symbols(self) -> set:
        """Every symbol with a non-zero position, as one API call.

        The per-symbol ``has_open_position`` guard costs one full positions fetch
        per name per cycle: on the 90-name broad universe at a 30s poll that is
        ~180 trading-API requests/minute before get_clock/get_account/get_orders,
        which trips Alpaca's 200/min limit and makes symbols silently drop out
        (429 -> exception -> candidate skipped). Callers fetch this once per
        cycle and test membership instead. Raises on API error so callers can
        fail CLOSED rather than trade on a stale view.
        """
        return {str(p.symbol).upper() for p in self.get_positions()
                if getattr(p, "symbol", None) and float(p.qty) != 0}

    def _position_exists(self, symbol: str) -> bool:
        """Broker-truth position check that RAISES on API error.

        ``has_open_position`` is the same query but callers there treat failure
        as "no position"; the emergency-stop path must distinguish "confirmed
        flat" from "couldn't ask".
        """
        symbol = symbol.strip().upper()
        return any(str(p.symbol).upper() == symbol and float(p.qty) != 0
                   for p in self.client.list_positions())

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
