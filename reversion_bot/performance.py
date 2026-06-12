from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def time_bucket_from_hour(hour_utc: int) -> str:
    """Map a UTC hour to an ET trading-session bucket.
    Market hours in UTC: 14:30–21:00 (9:30–16:00 ET, UTC-5 / UTC-4 DST).
    We use UTC-5 offset as a conservative approximation.
    """
    hour_et = (hour_utc - 5) % 24
    if hour_et < 10:
        return "early_session"   # 9:30–10:00 ET
    if hour_et < 12:
        return "midmorning"      # 10:00–12:00 ET
    if hour_et < 14:
        return "midday"          # 12:00–14:00 ET
    return "late_session"        # 14:00–16:00 ET


@dataclass
class EvalRecord:
    timestamp: str
    symbol: str
    regime: str
    entry_style: str
    decision: str
    reason: str
    router_reason: str
    trade_score: float
    threshold: float
    go_long: bool
    close: float | None = None
    ri: float | None = None
    rsi: float | None = None
    adx: float | None = None
    time_bucket: str | None = None


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    regime: str
    entry_style: str
    trade_score: float
    threshold: float
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    rr_ratio: float
    risk_per_share: float
    reward_per_share: float
    position_value: float
    time_bucket: str | None = None


@dataclass
class OutcomeRecord:
    """A CLOSED trade with its realized result. This — not the entry-time
    trade_score — is what threshold adaptation is allowed to learn from."""
    timestamp: str
    symbol: str
    regime: str
    entry_style: str
    realized_pnl: float
    entry_price: float | None = None
    exit_price: float | None = None
    qty: int | None = None


class PerformanceTracker:
    def __init__(self, state_dir: str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.evals_path = self.state_dir / "evaluations.jsonl"
        self.trades_path = self.state_dir / "trades.jsonl"
        self.outcomes_path = self.state_dir / "outcomes.jsonl"

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def log_evaluation(self, record: EvalRecord) -> None:
        self._append_jsonl(self.evals_path, asdict(record))

    def log_trade(self, record: TradeRecord) -> None:
        self._append_jsonl(self.trades_path, asdict(record))

    def log_outcome(self, record: OutcomeRecord) -> None:
        """Call when a position CLOSES (fill reconciliation in main.py is the
        natural place). Without outcomes, threshold adaptation stays inert."""
        self._append_jsonl(self.outcomes_path, asdict(record))

    def summarize_recent(self, limit: int = 500) -> Dict[str, Any]:
        evals = self._read_jsonl(self.evals_path)[-limit:]
        trades = self._read_jsonl(self.trades_path)[-limit:]

        by_style: Dict[str, int] = {}
        by_regime: Dict[str, int] = {}
        by_router_reason: Dict[str, int] = {}

        for t in trades:
            style = str(t.get("entry_style") or "unknown")
            regime = str(t.get("regime") or "unknown")
            by_style[style] = by_style.get(style, 0) + 1
            by_regime[regime] = by_regime.get(regime, 0) + 1

        for e in evals:
            rr = str(e.get("router_reason") or "unknown")
            by_router_reason[rr] = by_router_reason.get(rr, 0) + 1

        return {
            "recent_evals": len(evals),
            "recent_trades": len(trades),
            "by_style": by_style,
            "by_regime": by_regime,
            "by_router_reason": by_router_reason,
        }

    def suggest_threshold_adjustment(
        self,
        *,
        entry_style: str,
        regime: str,
        baseline_threshold: float,
        min_samples: int,
        max_adj: float,
    ) -> float:
        # REWRITTEN: the old version adapted the threshold from the average
        # entry-time trade_score of past trades — a feedback loop on the
        # system's own opinion (high scores "earned" a lower bar regardless of
        # whether those trades made money). Adaptation now requires REALIZED
        # OUTCOMES; with no outcome evidence it is a strict no-op.
        outcomes = self._read_jsonl(self.outcomes_path)
        matching = [
            o for o in outcomes
            if o.get("entry_style") == entry_style and o.get("regime") == regime
            and o.get("realized_pnl") is not None
        ]
        if len(matching) < min_samples:
            return baseline_threshold

        pnls = [float(o["realized_pnl"]) for o in matching]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        expectancy = sum(pnls) / len(pnls)

        # Losing combination (negative expectancy, sub-coin-flip hit rate):
        # demand MORE conviction before the next entry in this style/regime.
        if expectancy < 0 and win_rate < 0.45:
            return min(baseline_threshold + max_adj, 0.80)
        # Strongly working combination: allow slightly easier entries, floored.
        if expectancy > 0 and win_rate > 0.55:
            return max(baseline_threshold - max_adj, 0.20)
        return baseline_threshold
