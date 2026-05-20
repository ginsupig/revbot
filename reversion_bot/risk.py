from __future__ import annotations

from math import floor

from .config import RiskConfig
from .models import PositionPlan, ReversionDecision


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def build_long_plan(
        self,
        account_equity: float,
        decision: ReversionDecision,
        conviction_score: float,
        entry_style: str = "mean_reversion",
    ) -> PositionPlan:
        entry = float(decision.close or 0.0)
        atr = float(decision.atr or 0.0)

        if entry <= 0:
            raise ValueError("Entry price must be positive.")

        atr_floor = entry * self.config.atr_floor_pct
        atr = max(atr, atr_floor)

        stop_mult, target_mult = self._style_multiples(entry_style)

        stop = round(entry - atr * stop_mult, 2)
        target = round(entry + atr * target_mult, 2)

        if stop >= entry or target <= entry:
            raise ValueError(
                f"Invalid stop/entry/target: stop={stop}, entry={entry}, target={target}"
            )

        risk_per_share = round(entry - stop, 4)
        reward_per_share = round(target - entry, 4)
        rr_ratio = reward_per_share / max(risk_per_share, 1e-9)

        if rr_ratio < self.config.min_rr:
            raise ValueError(
                f"Risk/reward below minimum threshold: {rr_ratio:.2f} < {self.config.min_rr:.2f}"
            )

        conviction_boost = min(max(conviction_score - 0.50, 0.0), 0.35)
        risk_budget = account_equity * self.config.risk_per_trade_pct * (1.0 + conviction_boost)

        raw_qty = floor(risk_budget / max(risk_per_share, 1e-9))
        qty = max(self.config.min_qty, raw_qty)

        max_position_value = account_equity * self.config.max_position_value_pct
        qty_cap_by_value = floor(max_position_value / entry)
        qty = min(qty, max(qty_cap_by_value, 0))

        if qty < self.config.min_qty:
            raise ValueError("Calculated quantity below minimum tradeable size.")

        position_value = round(qty * entry, 2)
        if position_value < self.config.min_position_value:
            raise ValueError("Position value below minimum threshold.")

        return PositionPlan(
            qty=int(qty),
            entry_price=round(entry, 2),
            stop_price=round(stop, 2),
            target_price=round(target, 2),
            risk_per_share=round(risk_per_share, 4),
            reward_per_share=round(reward_per_share, 4),
            rr_ratio=round(rr_ratio, 4),
            position_value=position_value,
            max_account_risk=round(risk_budget, 2),
        )

    def _style_multiples(self, entry_style: str) -> tuple[float, float]:
        style = (entry_style or "mean_reversion").lower()

        if style == "trend_following":
            return (
                self.config.trend_stop_atr_multiple,
                self.config.trend_target_atr_multiple,
            )

        if style == "trendfail":
            return (
                self.config.trendfail_stop_atr_multiple,
                self.config.trendfail_target_atr_multiple,
            )

        return (
            self.config.stop_atr_multiple,
            self.config.target_atr_multiple,
        )

    def build_plan_for_style(
        self,
        account_equity: float,
        decision: ReversionDecision,
        conviction_score: float,
        entry_style: str,
    ) -> PositionPlan | None:
        try:
            return self.build_long_plan(
                account_equity=account_equity,
                decision=decision,
                conviction_score=conviction_score,
                entry_style=entry_style,
            )
        except ValueError:
            return None