from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple
from datetime import datetime, timezone

from .config import PortfolioConfig
from .portfolio import PortfolioState
from .performance import utc_now_iso


DEFAULT_BUCKETS: Dict[str, Set[str]] = {
    "mega_tech": {"AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN"},
    "semi_ai": {"NVDA", "AMD", "SMCI", "MU"},
    "high_beta_growth": {"TSLA", "ASTS", "APP", "PL", "MP", "SMCI"},
}


class ExecutionGovernor:
    def __init__(
        self,
        config: PortfolioConfig | None = None,
        buckets: Dict[str, Set[str]] | None = None,
        portfolio_state: PortfolioState | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.buckets = buckets or DEFAULT_BUCKETS
        self.portfolio_state = portfolio_state or PortfolioState()

    def total_risk_heat(self, positions: Iterable) -> float:
        """Capital at risk = sum of (qty * stop_distance) for each open position.

        Uses the stored stop_distance from position metadata (the distance from
        entry to stop, set at entry time). This measures the *theoretical* max
        loss per position, not just current drawdown — so a fresh-at-entry
        position correctly contributes its full risk, preventing the heat gate
        from allowing too many simultaneous positions.

        Falls back to negative unrealized P&L when no metadata is available
        (carryover positions opened before tracking was added).
        """
        all_meta = self.portfolio_state.get_all_position_metadata() if self.portfolio_state else {}
        total = 0.0
        for p in positions:
            sym = str(getattr(p, "symbol", "")).upper()
            meta = all_meta.get(sym)
            if meta and "stop_distance" in meta:
                qty = abs(float(getattr(p, "qty", 0)))
                total += qty * float(meta["stop_distance"])
            else:
                total += max(0.0, -float(getattr(p, "unrealized_pl", 0.0)))
        return total

    @staticmethod
    def pending_entry_orders(orders, open_symbols) -> Dict[str, Dict[str, float]]:
        """Working ENTRY orders: open orders on symbols with NO position.

        These are invisible to list_positions but are committed capital — a
        resting marketable-limit bracket WILL become a position. Without
        counting them, max_open_positions/exposure/heat can all be overshot
        (3 positions + 2 approvals in one cycle -> 5 positions once they fill).

        Bracket legs of LIVE positions share the position's symbol and are
        excluded by the open_symbols check; multiple working orders on one
        pending symbol (parent + held legs) count once, at the largest
        notional. Notional uses the order's limit/stop price — 0 for a plain
        market parent (no price on the order), which under-counts exposure but
        still counts the SLOT toward max_open_positions.
        """
        open_set = {str(s).upper() for s in open_symbols}
        pending: Dict[str, Dict[str, float]] = {}
        for o in orders:
            sym = str(getattr(o, "symbol", "") or "").upper()
            if not sym or sym in open_set:
                continue
            try:
                qty = abs(float(getattr(o, "qty", 0) or 0))
            except (TypeError, ValueError):
                qty = 0.0
            px = getattr(o, "limit_price", None) or getattr(o, "stop_price", None)
            try:
                notional = qty * float(px) if px else 0.0
            except (TypeError, ValueError):
                notional = 0.0
            prev = pending.get(sym)
            if prev is None or notional > prev["notional"]:
                pending[sym] = {"qty": qty, "notional": notional}
        return pending

    def symbol_buckets(self, symbol: str) -> List[str]:
        symbol = symbol.upper()
        return [name for name, members in self.buckets.items() if symbol in members]

    def current_bucket_counts(self, open_symbols: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in self.buckets}
        for sym in open_symbols:
            for bucket in self.symbol_buckets(sym):
                counts[bucket] += 1
        return counts

    def rank_candidates(self, candidate_payloads: List[dict]) -> List[dict]:
        def priority_key(p: dict):
            trade_score = float(p.get("trade_score", 0.0))
            decision = p.get("decision", {}) or {}
            style = p.get("entry_style", "")
            regime = p.get("regime", "")
            # Validated = both LONG and SHORT reversion signals (not just longs).
            validated = 1 if decision.get("signal") in ("LONG_REVERSION", "SHORT_REVERSION") else 0
            style_bonus = 1 if style in {"trend_following", "mean_reversion", "trendfail"} else 0
            regime_bonus = 1 if regime in {"trend", "reversion", "range"} else 0
            # Tie-breaker: symbol name ensures deterministic ordering when scores match.
            symbol = str(p.get("symbol", "")).upper()
            return (trade_score, validated, style_bonus, regime_bonus, symbol)
        return sorted(candidate_payloads, key=priority_key, reverse=True)

    def effective_risk_multiplier(self, drawdown_pct: float) -> float:
        if drawdown_pct >= self.config.drawdown_pause_pct:
            return 0.0
        if drawdown_pct >= self.config.reduce_size_after_drawdown_pct:
            return self.config.reduced_risk_multiplier
        return 1.0

    def can_open(
        self,
        *,
        symbol: str,
        entry_style: str,
        regime: str,
        account_equity: float,
        open_symbols: Iterable[str],
        open_styles: Dict[str, int],
        open_regimes: Dict[str, int],
        current_total_exposure: float,
        current_total_heat: float,
        new_position_value: float,
        new_position_heat: float,
        trades_executed_this_cycle: int,
        side: str = "long",
        open_order_symbols: Iterable[str] = (),
    ) -> Tuple[bool, str]:
        symbol = symbol.upper()
        open_symbols = {s.upper() for s in open_symbols}

        now = datetime.now(timezone.utc)
        today = now.date().isoformat()

        if symbol in open_symbols:
            return False, "open_position_exists"

        # A working (unfilled) order on this symbol — invisible to list_positions
        # — would otherwise let a second bracket stack on the same name.
        open_order_set = {s.upper() for s in open_order_symbols}
        if symbol in open_order_set:
            return False, "open_order_exists"

        # Pending entries (working orders on symbols with no position) occupy a
        # position slot the moment they're submitted — counting only filled
        # positions let the cap be overshot by however many brackets were
        # resting unfilled.
        pending_entries = open_order_set - open_symbols
        if len(open_symbols) + len(pending_entries) >= self.config.max_open_positions:
            return False, "max_open_positions_reached"

        if trades_executed_this_cycle >= self.config.max_trades_per_cycle:
            return False, "max_trades_per_cycle_reached"

        if self.portfolio_state.daily_new_positions_count(today) >= self.config.max_daily_new_positions:
            return False, "max_daily_new_positions_reached"

        if self.portfolio_state.in_symbol_cooldown(symbol, now, self.config.symbol_cooldown_minutes):
            return False, "symbol_cooldown"

        if self.portfolio_state.in_direction_flip_cooldown(
            symbol, side, now, self.config.direction_flip_cooldown_minutes
        ):
            return False, "direction_flip_cooldown"

        max_exposure = account_equity * self.config.max_total_exposure_pct
        if current_total_exposure + new_position_value > max_exposure:
            return False, "portfolio_exposure_limit"

        max_heat = account_equity * self.config.max_portfolio_heat_pct
        if current_total_heat + new_position_heat > max_heat:
            return False, "portfolio_heat_limit"

        bucket_counts = self.current_bucket_counts(open_symbols)
        for bucket in self.symbol_buckets(symbol):
            if bucket_counts.get(bucket, 0) >= self.config.max_positions_per_bucket:
                return False, f"bucket_limit:{bucket}"

        if regime and open_regimes.get(regime, 0) >= self.config.max_positions_per_regime:
            return False, f"regime_limit:{regime}"

        if entry_style == "mean_reversion" and open_styles.get("mean_reversion", 0) >= self.config.max_reversion_positions:
            return False, "style_limit:mean_reversion"
        if entry_style == "trend_following" and open_styles.get("trend_following", 0) >= self.config.max_trend_positions:
            return False, "style_limit:trend_following"
        if entry_style == "trendfail" and open_styles.get("trendfail", 0) >= self.config.max_trendfail_positions:
            return False, "style_limit:trendfail"

        drawdown_pct = self.portfolio_state.get_drawdown_pct(account_equity)
        risk_mult = self.effective_risk_multiplier(drawdown_pct)
        if risk_mult <= 0.0:
            return False, "drawdown_pause"

        return True, "ok"

    def approve(
        self,
        candidate: dict,
        executor,
        trades_executed_this_cycle: int,
    ) -> bool:
        """High-level approval gate called from the main loop.

        Extracts all required state from the candidate payload + live broker,
        then delegates to can_open().
        """
        if not (candidate.get("go_long") or candidate.get("go_short")):
            return False

        symbol = candidate.get("symbol", "")
        entry_style = candidate.get("entry_style", "mean_reversion")
        regime = candidate.get("regime", "reversion")
        side = "short" if candidate.get("go_short") else "long"
        position_plan = candidate.get("position_plan", {})
        new_position_value = float(position_plan.get("position_value", 0.0))
        new_position_heat = float(candidate.get("portfolio_heat", 0.0))

        try:
            account = executor.client.get_account()
            pv = getattr(account, "portfolio_value", None)
            eq = getattr(account, "equity", None)
            account_equity = float(pv if pv else eq)
            positions = executor.client.list_positions()
            # Working (unfilled) orders — fetched in the same guarded block so an
            # error fails CLOSED (skip the entry) rather than risk a double-submit.
            if hasattr(executor, "list_open_orders"):
                open_orders = list(executor.list_open_orders())
                open_order_syms = {str(getattr(o, "symbol", "") or "").upper()
                                   for o in open_orders} - {""}
            else:
                # Legacy executor API: symbols only — dup/slot checks still
                # work; pending notional/heat accounting is unavailable.
                open_orders = []
                open_order_syms = {s.upper() for s in executor.open_order_symbols()}
        except Exception as e:
            print(f"[GOVERNOR] account/order fetch failed: {e}")
            return False

        # Real-time intraday margin gate (FINRA Notice 26-10, eff. 2026-06-04).
        # The broker computes the intraday margin requirement; we simply respect
        # the buying power Alpaca reports, which already reflects current exposure.
        # Prefer day-trading buying power when available, else regular buying power.
        live_buying_power = None
        for attr in ("daytrading_buying_power", "buying_power"):
            val = getattr(account, attr, None)
            if val is None:
                continue
            try:
                parsed = float(val)
            except (TypeError, ValueError):
                continue
            # Non-PDT accounts (equity < $25k) report daytrading_buying_power
            # as "0" even with ample regular buying power — zero here means
            # "not applicable", not "broke". Fall through to buying_power.
            if attr == "daytrading_buying_power" and parsed <= 0.0:
                continue
            live_buying_power = parsed
            break
        if live_buying_power is not None and new_position_value > live_buying_power:
            print(
                f"[GOVERNOR] {symbol} rejected: insufficient buying power "
                f"(need {new_position_value:.2f}, have {live_buying_power:.2f})"
            )
            return False

        # Hard block if the broker has flagged the account (e.g. unmet intraday
        # margin deficit triggers a trading restriction under the new rules).
        if bool(getattr(account, "trading_blocked", False)) or bool(
            getattr(account, "account_blocked", False)
        ):
            print(f"[GOVERNOR] {symbol} rejected: broker trading_blocked flag set")
            return False

        open_symbols = [str(p.symbol).upper() for p in positions]
        # Reconstruct style/regime counts from persisted metadata (broker
        # positions don't carry these tags) so the per-style and per-regime
        # caps actually bind instead of comparing against empty dicts.
        open_styles, open_regimes = self.portfolio_state.open_style_regime_counts(open_symbols)
        current_total_exposure = sum(abs(float(p.market_value)) for p in positions)
        current_total_heat = self.total_risk_heat(positions)

        # Committed-but-unfilled entries count toward exposure and heat too:
        # a resting bracket's notional (qty x limit price) and its planned risk
        # (qty x stop_distance from the metadata recorded at submit time).
        pending = self.pending_entry_orders(open_orders, open_symbols)
        if pending:
            all_meta = self.portfolio_state.get_all_position_metadata() if self.portfolio_state else {}
            for sym, info in pending.items():
                current_total_exposure += info["notional"]
                meta = all_meta.get(sym)
                if meta and meta.get("stop_distance") is not None:
                    current_total_heat += info["qty"] * float(meta["stop_distance"])

        self.portfolio_state.update_equity(account_equity)

        ok, reason = self.can_open(
            symbol=symbol,
            entry_style=entry_style,
            regime=regime,
            account_equity=account_equity,
            open_symbols=open_symbols,
            open_styles=open_styles,
            open_regimes=open_regimes,
            current_total_exposure=current_total_exposure,
            current_total_heat=current_total_heat,
            new_position_value=new_position_value,
            new_position_heat=new_position_heat,
            trades_executed_this_cycle=trades_executed_this_cycle,
            side=side,
            open_order_symbols=open_order_syms,
        )
        if not ok:
            print(f"[GOVERNOR] {symbol} rejected: {reason}")
            return False

        # Apply drawdown-scaled risk multiplier: reduce position size when
        # drawdown exceeds reduce_size_after_drawdown_pct (was dead code before).
        drawdown_pct = self.portfolio_state.get_drawdown_pct(account_equity)
        risk_mult = self.effective_risk_multiplier(drawdown_pct)
        if risk_mult < 1.0 and risk_mult > 0.0:
            plan = candidate.get("position_plan")
            if plan and "qty" in plan:
                from math import floor
                original_qty = int(plan["qty"])
                scaled_qty = max(1, floor(original_qty * risk_mult))
                if scaled_qty < original_qty:
                    plan["qty"] = scaled_qty
                    plan["position_value"] = round(scaled_qty * float(plan.get("entry_price", 0)), 2)
                    print(f"[GOVERNOR] {symbol} drawdown scaling: qty {original_qty} -> {scaled_qty} "
                          f"(risk_mult={risk_mult:.2f}, drawdown={drawdown_pct:.3f})")

        return True
