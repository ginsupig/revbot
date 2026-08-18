from __future__ import annotations

import logging
from typing import Optional

from .portfolio import PortfolioState

logger = logging.getLogger("PositionManager")


class PositionManager:
    """Manages open positions each cycle: moves stop to breakeven then trails it."""

    def __init__(self, executor, portfolio_state: PortfolioState) -> None:
        self.executor = executor
        self.portfolio_state = portfolio_state

    def manage_all(self) -> None:
        try:
            positions = self.executor.client.list_positions()
        except Exception as e:
            logger.warning("Failed to fetch positions: %s", e)
            return

        pos_meta = self.portfolio_state.get_all_position_metadata()

        for pos in positions:
            symbol = str(pos.symbol).upper()
            meta = pos_meta.get(symbol, {})
            entry_price = float(meta.get("entry_price", 0.0))
            atr = float(meta.get("atr", 0.0))
            tracked_stop = float(meta.get("stop_price", 0.0))
            side = str(meta.get("side") or ("short" if float(pos.qty) < 0 else "long"))

            if entry_price <= 0 or atr <= 0:
                continue

            current_price = float(getattr(pos, "current_price", 0.0))
            if current_price <= 0:
                continue

            new_stop = self._calc_new_stop(current_price, entry_price, atr, tracked_stop, side)
            if new_stop is None:
                continue

            if self._replace_stop(symbol, new_stop):
                logger.info(
                    "symbol=%s stop moved %.2f -> %.2f (current=%.2f, entry=%.2f)",
                    symbol, tracked_stop, new_stop, current_price, entry_price,
                )
                self.portfolio_state.update_stop_price(symbol, new_stop)

    def _calc_new_stop(
        self,
        current_price: float,
        entry_price: float,
        atr: float,
        current_stop: float,
        side: str = "long",
    ) -> Optional[float]:
        if side == "short":
            trail_trigger = entry_price - 2.0 * atr
            breakeven_trigger = entry_price - 1.0 * atr
            if current_price <= trail_trigger:
                candidate = round(current_price + atr, 2)
                if current_stop <= 0 or candidate < current_stop:
                    return candidate
            elif current_price <= breakeven_trigger:
                candidate = round(entry_price * 0.999, 2)
                if current_stop <= 0 or candidate < current_stop:
                    return candidate
            return None

        trail_trigger = entry_price + 2.0 * atr   # lock in gains
        breakeven_trigger = entry_price + 1.0 * atr  # move to breakeven

        if current_price >= trail_trigger:
            # Trail: stop ratchets up to current_price - 1×ATR
            candidate = round(current_price - atr, 2)
            if candidate > current_stop:
                return candidate

        elif current_price >= breakeven_trigger:
            # Breakeven: stop moves to entry + tiny buffer (0.1%)
            candidate = round(entry_price * 1.001, 2)
            if candidate > current_stop:
                return candidate

        return None

    def _replace_stop(self, symbol: str, new_stop: float) -> bool:
        try:
            orders = self.executor.client.list_orders(status="open")
            stop_order = next(
                (
                    o for o in orders
                    if str(o.symbol).upper() == symbol
                    and getattr(o, "type", "") == "stop"
                ),
                None,
            )
            if stop_order is None:
                logger.warning("symbol=%s no open stop order found", symbol)
                return False
            self.executor.client.replace_order(
                order_id=stop_order.id,
                stop_price=round(new_stop, 2),
            )
            return True
        except Exception as e:
            logger.warning("symbol=%s stop replace failed: %s", symbol, e)
            return False
