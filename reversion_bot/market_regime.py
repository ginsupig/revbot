from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sectors import sector_for, etf_for_sector


def is_risk_off(bars: pd.DataFrame | None, ema_length: int = 50) -> bool:
    """True when the benchmark's latest close sits below its trend EMA.

    This is the bot's market-regime gate: when a broad benchmark (e.g. SPY) is
    trending down, dip-buying every oversold name is the losing trade, so the
    caller suppresses new longs.

    Fail-open by design: returns False (risk-on) on missing/short/NaN data so a
    benchmark fetch hiccup never halts all trading.
    """
    if bars is None or len(bars) < ema_length or "close" not in bars.columns:
        return False
    close = bars["close"].astype(float)
    ema = close.ewm(span=ema_length, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    # NaN guard (NaN != NaN).
    if last_close != last_close or last_ema != last_ema:
        return False
    return last_close < last_ema


def classify_regime_series(
    bars: pd.DataFrame | None,
    ema_length: int = 50,
    adx_length: int = 14,
    adx_min: float = 20.0,
    atr_length: int = 14,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Per-bar market-regime flags from a benchmark's OHLC bars (intended: daily SPY).

    The scalar ``is_risk_off`` answers only "is the benchmark below trend?". This
    returns the three orthogonal legs of a *hostile* regime so a caller can test
    them as a ladder (which one actually does the work?):

      trend_down : close < EMA(ema_length)            -- below trend
      adx_high   : ADX(adx_length) >= adx_min          -- the move is a real trend, not chop
      vol_up     : ATR%(atr_length) > its rolling mean -- volatility expanding

    Pure / network-free. Warm-up rows are False. Returns a bool DataFrame aligned
    to ``bars.index``; an empty/None frame yields an empty result.
    """
    from .indicators import calculate_adx, calculate_atr, calculate_ema

    if bars is None or len(bars) == 0 or "close" not in bars.columns:
        return pd.DataFrame(columns=["trend_down", "adx_high", "vol_up"])

    close = bars["close"].astype(float)
    ema = calculate_ema(close, ema_length)
    adx = calculate_adx(bars, adx_length)
    atr_pct = calculate_atr(bars, atr_length) / close.replace(0.0, np.nan)
    atr_pct_avg = atr_pct.rolling(vol_lookback, min_periods=vol_lookback).mean()

    out = pd.DataFrame(index=bars.index)
    out["trend_down"] = (close < ema).fillna(False)
    out["adx_high"] = (adx >= adx_min).fillna(False)
    out["vol_up"] = (atr_pct > atr_pct_avg).fillna(False)
    return out.astype(bool)


def suppress_longs_if_risk_off(
    candidates: List[Dict], risk_off: bool
) -> Tuple[List[Dict], List[Dict]]:
    """When risk-off, drop long candidates; shorts pass through untouched.

    Returns (kept, dropped_longs). A candidate is a long if it carries go_long;
    go_short candidates (and anything else) are always kept.
    """
    if not risk_off:
        return candidates, []
    kept = [c for c in candidates if not c.get("go_long")]
    dropped = [c for c in candidates if c.get("go_long")]
    return kept, dropped


def resolve_sector_regime(
    sectors,
    fetch_bars: Callable[[str], Optional[pd.DataFrame]],
    ema_length: int = 50,
) -> Dict[str, bool]:
    """Map each sector -> risk-off bool via its proxy ETF's trend.

    ``sectors`` is an iterable of sector names; ``fetch_bars(etf)`` returns that
    ETF's bars (or None). ETF lookups are deduped (sectors sharing an ETF fetch
    once) and fail-open per ETF: a fetch/parse error is treated as risk-on so a
    single benchmark hiccup never freezes a sector. Sectors with no proxy ETF
    are omitted (the caller falls back to the broad benchmark for them).
    """
    regime: Dict[str, bool] = {}
    etf_eval: Dict[str, bool] = {}
    for sector in sectors:
        etf = etf_for_sector(sector)
        if not etf:
            continue
        if etf not in etf_eval:
            try:
                etf_eval[etf] = is_risk_off(fetch_bars(etf), ema_length)
            except Exception:
                etf_eval[etf] = False  # fail-open
        regime[sector] = etf_eval[etf]
    return regime


def suppress_longs_by_sector(
    candidates: List[Dict],
    sector_regime: Dict[str, bool],
    fallback_risk_off: bool = False,
    combine: str = "or",
) -> Tuple[List[Dict], List[Dict]]:
    """Drop long candidates whose SECTOR is risk-off; shorts pass through.

    ``sector_regime`` maps sector -> risk-off bool (from resolve_sector_regime).
    ``fallback_risk_off`` is the broad-benchmark (SPY) signal, used for symbols
    with no sector mapping and — under ``combine="or"`` — OR'd with every
    symbol's sector signal so a market-wide selloff still suppresses longs.

    combine:
      "or"     -> suppress if the symbol's sector OR the broad benchmark is
                  risk-off (default; catches both sector-specific and market-wide).
      "sector" -> suppress on the sector signal alone; an unmapped symbol falls
                  back to the broad benchmark.

    Returns (kept, dropped_longs).
    """
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for c in candidates:
        if not c.get("go_long"):
            kept.append(c)
            continue
        sector = sector_for(c.get("symbol", ""))
        sector_ro = sector_regime.get(sector) if sector is not None else None
        if combine == "sector":
            risk_off = sector_ro if sector_ro is not None else bool(fallback_risk_off)
        else:  # "or"
            risk_off = bool(sector_ro) or bool(fallback_risk_off)
        (dropped if risk_off else kept).append(c)
    return kept, dropped
