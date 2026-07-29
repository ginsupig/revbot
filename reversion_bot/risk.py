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
        conviction_score: float = 0.5,
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
        return self._size_plan(
            account_equity=account_equity,
            entry=entry,
            stop=stop,
            target=target,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            conviction_score=conviction_score,
            side="long",
        )

    def build_short_plan(
        self,
        account_equity: float,
        decision: ReversionDecision,
        conviction_score: float = 0.5,
        entry_style: str = "mean_reversion",
    ) -> PositionPlan:
        """Mirror of build_long_plan for a sell-to-open: stop *above* entry,
        target *below*. Sizing (risk budget, value cap) is identical."""
        entry = float(decision.close or 0.0)
        atr = float(decision.atr or 0.0)

        if entry <= 0:
            raise ValueError("Entry price must be positive.")

        atr_floor = entry * self.config.atr_floor_pct
        atr = max(atr, atr_floor)

        stop_mult, target_mult = self._style_multiples(entry_style)

        stop = round(entry + atr * stop_mult, 2)
        target = round(entry - atr * target_mult, 2)

        if stop <= entry or target >= entry:
            raise ValueError(
                f"Invalid short stop/entry/target: stop={stop}, entry={entry}, target={target}"
            )

        risk_per_share = round(stop - entry, 4)
        reward_per_share = round(entry - target, 4)
        return self._size_plan(
            account_equity=account_equity,
            entry=entry,
            stop=stop,
            target=target,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            conviction_score=conviction_score,
            side="short",
        )

    def _size_plan(
        self,
        *,
        account_equity: float,
        entry: float,
        stop: float,
        target: float,
        risk_per_share: float,
        reward_per_share: float,
        conviction_score: float,
        side: str,
    ) -> PositionPlan:
        """Shared position sizing for both directions. Risk/reward are already
        signed positive by the caller, so the math is direction-agnostic."""
        rr_ratio = reward_per_share / max(risk_per_share, 1e-9)

        if rr_ratio < self.config.min_rr:
            raise ValueError(
                f"Risk/reward below minimum threshold: {rr_ratio:.2f} < {self.config.min_rr:.2f}"
            )

        # NOTE: the boost deliberately scales the risk budget UP TO +35% over
        # RISK_PER_TRADE_PCT for high-conviction signals (conviction 0.85+).
        # The min_qty rejection below therefore guards the BOOSTED budget, not
        # the bare risk_per_trade_pct — worst case per trade is
        # 1.35 x risk_per_trade_pct of equity.
        conviction_boost = min(max(conviction_score - 0.50, 0.0), 0.35)
        risk_budget = account_equity * self.config.risk_per_trade_pct * (1.0 + conviction_boost)

        raw_qty = floor(risk_budget / max(risk_per_share, 1e-9))
        # Never floor UP to min_qty: if the (boosted) risk budget can't afford
        # even min_qty shares at this stop distance, forcing min_qty would risk
        # more still (a wide-stop / high-priced name). Reject instead so no
        # position ever exceeds the boosted per-trade risk budget above.
        if raw_qty < self.config.min_qty:
            raise ValueError(
                f"Risk budget {risk_budget:.2f} affords {raw_qty} < min_qty "
                f"{self.config.min_qty} at {risk_per_share:.4f}/share risk; "
                f"min_qty would exceed the per-trade risk cap."
            )
        qty = raw_qty

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
            side=side,
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
        side: str = "long",
    ) -> PositionPlan | None:
        builder = self.build_short_plan if side == "short" else self.build_long_plan
        try:
            return builder(
                account_equity=account_equity,
                decision=decision,
                conviction_score=conviction_score,
                entry_style=entry_style,
            )
        except ValueError:
            return None