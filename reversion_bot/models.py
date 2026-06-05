from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any

Signal = Literal['LONG_REVERSION', 'SHORT_REVERSION', 'WAIT']


@dataclass
class SafetyDecision:
    is_safe: bool
    reason: str
    adx: Optional[float] = None
    rsi: Optional[float] = None
    spread_bps: Optional[float] = None
    dollar_volume: Optional[float] = None


@dataclass
class ReversionDecision:
    signal: Signal
    reason: str
    symbol: Optional[str] = None
    close: Optional[float] = None
    lb1: Optional[float] = None
    lb2: Optional[float] = None
    ub1: Optional[float] = None
    ub2: Optional[float] = None
    sma: Optional[float] = None
    ri: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    spread_bps: Optional[float] = None
    dollar_volume: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PositionPlan:
    qty: int
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_share: float
    reward_per_share: float
    rr_ratio: float
    position_value: float
    max_account_risk: float
    # Direction of the trade. "long" buys with a stop below / target above;
    # "short" sells-to-open with a stop above / target below. Defaults to "long"
    # so existing long-only call sites and persisted plans are unaffected.
    side: str = "long"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
