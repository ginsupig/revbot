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

    @staticmethod
    def total_drawdown_heat(positions: Iterable) -> float:
        """Capital currently at risk = sum of open *drawdown* only.

        Counts underwater positions (negative unrealized P&L) as positive heat;
        winners contribute 0, so a profitable book doesn't block new entries.
        """
        return sum(
            max(0.0, -float(getattr(p, "unrealized_pl", 0.0))) for p in positions
        )

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
            validated = 1 if decision.get("signal") == "LONG_REVERSION" else 0
            style_bonus = 1 if style in {"trend_following", "mean_reversion", "trendfail"} else 0
            regime_bonus = 1 if regime in {"trend", "reversion", "range"} else 0
            return (trade_score, validated, style_bonus, regime_bonus)
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
    ) -> Tuple[bool, str]:
        symbol = symbol.upper()
        open_symbols = {s.upper() for s in open_symbols}

        now = datetime.now(timezone.utc)
        today = now.date().isoformat()

        if symbol in open_symbols:
            return False, "open_position_exists"

        if len(open_symbols) >= self.config.max_open_positions:
            return False, "max_open_positions_reached"

        if trades_executed_this_cycle >= self.config.max_trades_per_cycle:
            return False, "max_trades_per_cycle_reached"

        if self.portfolio_state.daily_new_positions_count(today) >= self.config.max_daily_new_positions:
            return False, "max_daily_new_positions_reached"

        if self.portfolio_state.in_symbol_cooldown(symbol, now, self.config.symbol_cooldown_minutes):
            return False, "symbol_cooldown"

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
        if drawdown_pct >= self.config.drawdown_pause_pct:
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
        position_plan = candidate.get("position_plan", {})
        new_position_value = float(position_plan.get("position_value", 0.0))
        new_position_heat = float(candidate.get("portfolio_heat", 0.0))

        try:
            account = executor.client.get_account()
            pv = getattr(account, "portfolio_value", None)
            eq = getattr(account, "equity", None)
            account_equity = float(pv if pv else eq)
            positions = executor.client.list_positions()
        except Exception as e:
            print(f"[GOVERNOR] account fetch failed: {e}")
            return False

        # Real-time intraday margin gate (FINRA Notice 26-10, eff. 2026-06-04).
        # The broker computes the intraday margin requirement; we simply respect
        # the buying power Alpaca reports, which already reflects current exposure.
        # Prefer day-trading buying power when available, else regular buying power.
        live_buying_power = None
        for attr in ("daytrading_buying_power", "buying_power"):
            val = getattr(account, attr, None)
            if val is not None:
                try:
                    live_buying_power = float(val)
                    break
                except (TypeError, ValueError):
                    continue
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
        current_total_heat = self.total_drawdown_heat(positions)

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
        )
        if not ok:
            print(f"[GOVERNOR] {symbol} rejected: {reason}")
        return ok
