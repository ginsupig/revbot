from __future__ import annotations

from alpaca_trade_api.rest import REST
from time import sleep
import logging
from requests.exceptions import HTTPError

from .models import PositionPlan


class AlpacaExecutor:
    def __init__(self, api_key: str, secret_key: str, exec_config):
        # Ensure exec_config has a base_url attribute
        if not hasattr(exec_config, 'base_url'):
            raise ValueError("ExecutionConfig must have a 'base_url' attribute.")

        self.client = REST(api_key, secret_key, exec_config.base_url)
        self._order_ids = set()
        self._tif = getattr(exec_config, "tif", "day") or "day"
        # Limit-entry settings (important for wide-spread leveraged ETFs).
        self._use_limit_entry = bool(getattr(exec_config, "use_limit_entry", False))
        self._limit_entry_offset_bps = float(getattr(exec_config, "limit_entry_offset_bps", 8.0))

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

        order_key = self._order_key(symbol, plan)
        if order_key in self._order_ids:
            raise RuntimeError(f"Duplicate order detected for {order_key}.")

        order_data = {
            "symbol": symbol,
            "qty": plan.qty,
            "side": "buy",
            "type": "market",
            "time_in_force": self._tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(plan.target_price, 2)},
            "stop_loss": {"stop_price": round(plan.stop_price, 2)}
        }

        # Use a marketable limit entry to cap slippage on wide-spread (leveraged) names.
        # Buy limit is set slightly ABOVE the planned entry so it still crosses the
        # spread and fills, while bounding the worst-case fill price.
        if self._use_limit_entry:
            limit_price = round(plan.entry_price * (1.0 + self._limit_entry_offset_bps / 10_000.0), 2)
            order_data["type"] = "limit"
            order_data["limit_price"] = limit_price

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
        )
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

    def has_open_long_position(self, symbol: str) -> bool:
        positions = self.get_positions()
        return any(str(p.symbol).upper() == symbol and float(p.qty) > 0 for p in positions)

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
    def _order_key(symbol: str, plan: PositionPlan) -> str:
        return "|".join([
            symbol,
            f"{plan.qty}",
            f"{plan.entry_price:.2f}",
            f"{plan.stop_price:.2f}",
            f"{plan.target_price:.2f}",
        ])
