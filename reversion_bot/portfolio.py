from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo
import json


# Circuit-breaker drawdown is measured per trading *session*. The bot flattens
# every position at the close, so it's flat overnight; anchoring the breaker to
# a single day's high-water mark (reset each morning) protects against a bad day
# without an immortal all-time peak that can latch the bot off permanently.
_SESSION_TZ = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    market_value: float
    unrealized_pl: float = 0.0
    entry_style: str = "unknown"
    regime: str = "unknown"


class PortfolioState:
    def __init__(self, state_dir: str = "state/portfolio") -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "portfolio_state.json"

    def _default_state(self) -> Dict[str, Any]:
        return {
            "peak_equity": None,          # all-time high, kept for reporting only
            "session_date": None,         # date the session high-water mark belongs to
            "session_peak_equity": None,  # intraday high-water mark (drives the breaker)
            "last_equity": None,
            "daily_new_positions": [],
            "last_trade_ts_by_symbol": {},
        }

    def _load(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_state()

    @staticmethod
    def _session_today() -> str:
        return datetime.now(_SESSION_TZ).date().isoformat()

    def _save(self, data: Dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update_equity(self, account_equity: float, today: str | None = None) -> Dict[str, Any]:
        data = self._load()
        today = today or self._session_today()

        # All-time peak — retained for reporting; no longer drives the breaker.
        peak = data.get("peak_equity")
        if peak is None or account_equity > peak:
            peak = account_equity
        data["peak_equity"] = peak

        # Session high-water mark: reset to current equity on a new trading day,
        # otherwise ratchet up. This is what the drawdown breaker measures against.
        if data.get("session_date") != today:
            data["session_date"] = today
            data["session_peak_equity"] = account_equity
        elif data.get("session_peak_equity") is None or account_equity > data["session_peak_equity"]:
            data["session_peak_equity"] = account_equity

        data["last_equity"] = account_equity
        self._save(data)
        return data

    def get_drawdown_pct(self, account_equity: float, today: str | None = None) -> float:
        """Drawdown from *today's* session high-water mark (0 on a fresh day).

        Resetting per session means a 2.5% intraday slide halts new entries for
        the rest of the day, but the next morning starts flat — so a bad day can
        never permanently latch the bot off, unlike an all-time-peak anchor.
        """
        data = self._load()
        today = today or self._session_today()
        if data.get("session_date") != today:
            return 0.0
        peak = data.get("session_peak_equity")
        if not peak or peak <= 0:
            return 0.0
        return max(0.0, (peak - account_equity) / peak)

    def note_new_position(self, symbol: str, entry_style: str, regime: str, timestamp_iso: str) -> None:
        data = self._load()
        today = timestamp_iso[:10]
        daily = [x for x in data.get("daily_new_positions", []) if str(x).startswith(today)]
        daily.append(timestamp_iso)
        data["daily_new_positions"] = daily
        data.setdefault("last_trade_ts_by_symbol", {})[symbol.upper()] = timestamp_iso
        self._save(data)

    def daily_new_positions_count(self, date_prefix: str) -> int:
        data = self._load()
        daily = data.get("daily_new_positions", [])
        return sum(1 for x in daily if str(x).startswith(date_prefix))

    def update(self, candidate: dict) -> None:
        """Convenience method called by the main loop after a trade is submitted."""
        from datetime import timezone
        symbol = str(candidate.get("symbol", "")).upper()
        entry_style = str(candidate.get("entry_style", "unknown"))
        regime = str(candidate.get("regime", "unknown"))
        ts = datetime.now(timezone.utc).isoformat()
        self.note_new_position(symbol, entry_style, regime, ts)

    def in_symbol_cooldown(self, symbol: str, now: datetime, cooldown_minutes: int) -> bool:
        data = self._load()
        last_ts = data.get("last_trade_ts_by_symbol", {}).get(symbol.upper())
        if not last_ts:
            return False
        try:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        except Exception:
            return False
        return now <= last_dt + timedelta(minutes=cooldown_minutes)
