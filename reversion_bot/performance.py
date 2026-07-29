from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import threading


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(ts: Any) -> "datetime | None":
    """Parse a timestamp string from ANY of the formats that reach this module.

    Broker fill times arrive as '...Z', our own records as '...+00:00', and
    str(datetime) produces a space separator — lexical comparison across those
    mis-orders badly ('Z' > '+', 'T' > ' '), which is how a 4-hour-older
    outcome could win a "latest" comparison (audit M4). Naive timestamps are
    assumed UTC. Returns None when unparseable.
    """
    if not ts:
        return None
    s = str(ts).strip().replace("Z", "+00:00")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def time_bucket_from_hour(hour_utc: int) -> str:
    """Map a UTC hour to an ET trading-session bucket.

    Uses the actual US/Eastern offset (DST-aware) instead of hardcoded UTC-5,
    so the bucket is correct year-round.
    """
    try:
        from zoneinfo import ZoneInfo
        now_utc = datetime.now(timezone.utc).replace(hour=hour_utc)
        hour_et = now_utc.astimezone(ZoneInfo("America/New_York")).hour
    except Exception:
        # Fallback to EST (UTC-5) if zoneinfo fails
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
        # High-water mark (ISO timestamp of the last closing fill turned into an
        # outcome) so repeated reconciles over an overlapping fill window never
        # double-log the same closed trade.
        self.outcomes_cursor_path = self.state_dir / "outcomes_cursor.txt"
        # In-memory outcomes cache: avoids full-file reads on every symbol eval.
        # Invalidated when new outcomes are logged.
        self._outcomes_cache: List[Dict[str, Any]] | None = None
        # evaluate_symbol runs concurrently across ~90 threads sharing this
        # tracker. Appends previously worked only by the accident of O_APPEND
        # + GIL semantics; the lock makes writes and cache mutation correct by
        # design (audit M5). Cheap: held only around file appends/cache ops.
        self._lock = threading.RLock()

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
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

    _EVALS_MAX_BYTES = 50_000_000

    def log_evaluation(self, record: EvalRecord) -> None:
        # evaluations.jsonl gains ~universe-size records per cycle forever;
        # roll it once per size cap (single .1 generation) so the disk and the
        # summarize_recent full-file read stay bounded. Diagnostics only —
        # trades/outcomes are never rotated (attribution reads them in full).
        with self._lock:
            try:
                if self.evals_path.exists() and self.evals_path.stat().st_size > self._EVALS_MAX_BYTES:
                    self.evals_path.replace(self.evals_path.with_suffix(".jsonl.1"))
            except OSError:
                pass
            self._append_jsonl(self.evals_path, asdict(record))

    def log_trade(self, record: TradeRecord) -> None:
        self._append_jsonl(self.trades_path, asdict(record))

    def _get_outcomes(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._outcomes_cache is None:
                self._outcomes_cache = self._read_jsonl(self.outcomes_path)
            return self._outcomes_cache

    def log_outcome(self, record: OutcomeRecord) -> None:
        """Call when a position CLOSES (fill reconciliation in main.py is the
        natural place). Without outcomes, threshold adaptation stays inert."""
        d = asdict(record)
        with self._lock:
            self._append_jsonl(self.outcomes_path, d)
            if self._outcomes_cache is not None:
                self._outcomes_cache.append(d)

    def recent_loss_exit(self, symbol: str, now: datetime, within_minutes: int) -> bool:
        """True if this symbol's most recent CLOSED trade was a loss within
        ``within_minutes`` of ``now``.

        Drives the loss-aware re-entry brake: after a reversion long stops out,
        re-buying the same name while it is still falling just repeats the loss
        (the WDC failure mode — bought, stopped, bought again lower, stopped
        again). The plain symbol cooldown re-arms on a timer regardless of
        outcome; this fires only after a *losing* close, so winners re-enter
        freely. Reads outcomes.jsonl (written on every close); an empty or
        unparseable log fails open (False) so a bookkeeping gap never freezes
        trading.
        """
        if within_minutes <= 0:
            return False
        sym = str(symbol).upper()
        latest = None
        latest_dt = None
        for r in self._get_outcomes():
            if str(r.get("symbol", "")).upper() != sym:
                continue
            # Parsed comparison: mixed formats ('Z' vs '+00:00' vs space
            # separator) mis-order lexically, so the "latest" outcome could be
            # hours stale (audit M4).
            dt = parse_ts(r.get("timestamp"))
            if dt is None:
                continue
            if latest_dt is None or dt >= latest_dt:
                latest_dt, latest = dt, r
        if latest is None or latest_dt is None:
            return False
        try:
            if float(latest.get("realized_pnl", 0.0)) >= 0:
                return False                    # last close was a win/scratch
            # parse_ts always returns tz-aware, so this comparison can no
            # longer raise on a naive stored timestamp (which used to kill the
            # symbol's whole evaluation for the cycle).
            return now <= latest_dt + timedelta(minutes=within_minutes)
        except (TypeError, ValueError):
            return False

    def _read_outcomes_cursor(self) -> str:
        try:
            return self.outcomes_cursor_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return ""

    def _write_outcomes_cursor(self, cursor: str) -> None:
        # Atomic replace so a crash mid-write can't corrupt the high-water mark.
        tmp = self.outcomes_cursor_path.with_suffix(".tmp")
        tmp.write_text(cursor, encoding="utf-8")
        tmp.replace(self.outcomes_cursor_path)

    def reconcile_outcomes(self, fills: List[Dict[str, Any]]) -> int:
        """Turn broker FILL activities into realized-trade outcomes.

        Pairs the fills into closed round-trips (long/short-aware FIFO), and for
        each closing fill recovers the entry's style/regime from trades.jsonl —
        the most recent logged entry for that symbol at or before the close — so
        the realized PnL is attributed to the (entry_style, regime) it belongs
        to. Idempotent: a persisted cursor (the latest close timestamp already
        booked) means re-running over an overlapping fill window is a no-op.

        ``fills`` are normalized dicts (symbol/side/qty/price/time), exactly the
        shape ``pnl_report.fetch_fills`` produces. Returns the number of new
        outcomes logged. Best-effort by contract — callers pass whatever the
        broker returned; an empty/garbage list simply logs nothing.
        """
        from .trade_report import realized_pnl_events

        events = realized_pnl_events(fills)
        if not events:
            return 0

        cursor = self._read_outcomes_cursor()
        cursor_dt = parse_ts(cursor)
        # Newest entry per symbol wins when several precede a close; entries are
        # logged at submit time so an entry's timestamp <= its exit's time.
        # All comparisons use PARSED datetimes: broker fill times ('Z' / space
        # separated) and our own ISO ('+00:00') mis-order lexically (audit M4).
        entries = []
        for e in self._read_jsonl(self.trades_path):
            if not e.get("symbol"):
                continue
            dt = parse_ts(e.get("timestamp"))
            if dt is not None:
                entries.append((dt, e))
        entries.sort(key=lambda t: t[0])

        def attribute(symbol: str, when_dt):
            sym = str(symbol).upper()
            best = None
            for dt, e in entries:
                if str(e.get("symbol")).upper() != sym:
                    continue
                if when_dt is None or dt <= when_dt:
                    best = e            # entries are time-sorted -> last match is newest
                else:
                    break
            return best

        logged = 0
        newest, newest_dt = cursor, cursor_dt
        for ev in events:
            when = ev["time"]
            when_dt = parse_ts(when)
            if cursor_dt is not None and when_dt is not None and when_dt <= cursor_dt:
                continue                # already booked on a prior reconcile
            if cursor and when_dt is None and str(when) <= cursor:
                continue                # unparseable: legacy lexical fallback
            entry = attribute(ev["symbol"], when_dt)
            if entry is None:
                continue                # no entry to attribute style/regime to
            self.log_outcome(
                OutcomeRecord(
                    timestamp=when or utc_now_iso(),
                    symbol=str(ev["symbol"]).upper(),
                    regime=str(entry.get("regime") or "unknown"),
                    entry_style=str(entry.get("entry_style") or "unknown"),
                    realized_pnl=float(ev["realized_pnl"]),
                    entry_price=entry.get("entry_price"),
                    qty=entry.get("qty"),
                )
            )
            logged += 1
            if when_dt is not None and (newest_dt is None or when_dt > newest_dt):
                newest_dt, newest = when_dt, when

        if newest and newest != cursor:
            self._write_outcomes_cursor(newest)
        return logged

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
        recency_limit: int = 200,
    ) -> float:
        # REWRITTEN: the old version adapted the threshold from the average
        # entry-time trade_score of past trades — a feedback loop on the
        # system's own opinion (high scores "earned" a lower bar regardless of
        # whether those trades made money). Adaptation now requires REALIZED
        # OUTCOMES; with no outcome evidence it is a strict no-op.
        outcomes = self._get_outcomes()
        matching = [
            o for o in outcomes
            if o.get("entry_style") == entry_style and o.get("regime") == regime
            and o.get("realized_pnl") is not None
        ]
        # Recency window: learn from the LAST recency_limit outcomes in this
        # (style, regime) cell, not everything ever logged — 20+ losers from a
        # long-dead config used to pin the cell at +max_adj forever, even
        # after the underlying bug was fixed (audit M6-adaptive).
        if recency_limit > 0:
            matching = matching[-recency_limit:]
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
