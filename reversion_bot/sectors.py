"""Sector classification + proxy-ETF map for the sector-aware regime gate.

The market-regime filter originally judged every symbol against ONE benchmark
(SPY). That misses a sector-specific selloff: when semis dump but SPY holds, the
bot would keep dip-buying the falling knife (observed live). This maps each
symbol to a primary sector and each sector to a liquid proxy ETF, so longs can
be judged against THEIR sector's trend (see market_regime.resolve_sector_regime).

A symbol with no mapping returns sector ``None`` and the caller falls back to the
broad SPY signal — safe, just not sector-aware until the name is classified here.
"""
from __future__ import annotations

from typing import Optional

# Primary sector per symbol (uppercase keys). One sector each — this drives the
# regime benchmark, not the governor's (overlapping) position-count buckets.
SYMBOL_SECTOR = {
    # Semiconductors / AI hardware
    "NVDA": "semis_ai", "AMD": "semis_ai", "SMCI": "semis_ai", "MU": "semis_ai",
    "ARM": "semis_ai", "LRCX": "semis_ai", "AMAT": "semis_ai", "AVGO": "semis_ai",
    "TSM": "semis_ai", "ASML": "semis_ai", "INTC": "semis_ai", "QCOM": "semis_ai",
    "MRVL": "semis_ai", "KLAC": "semis_ai",
    # Software / mega-cap tech
    "MSFT": "software", "AAPL": "software", "META": "software", "GOOGL": "software",
    "GOOG": "software", "AMZN": "software", "CRWD": "software", "APP": "software",
    "PLTR": "software", "NFLX": "software", "ADBE": "software", "ORCL": "software",
    "NOW": "software", "PANW": "software", "SNOW": "software",
    # Energy
    "OXY": "energy", "XOM": "energy", "CVX": "energy", "SLB": "energy",
    "COP": "energy", "DVN": "energy",
    # Materials / metals & mining
    "FCX": "materials", "CLF": "materials", "MP": "materials", "NEM": "materials",
    "X": "materials", "AA": "materials", "NUE": "materials",
    # Financials
    "JPM": "financials", "BAC": "financials", "GS": "financials", "WFC": "financials",
    "MS": "financials", "C": "financials", "SCHW": "financials",
    # Healthcare / biotech
    "LLY": "healthcare", "HIMS": "healthcare", "TEM": "healthcare", "UNH": "healthcare",
    "PFE": "healthcare", "MRNA": "healthcare", "AMGN": "healthcare", "ISRG": "healthcare",
}

# Liquid proxy ETF per sector. A sector mapped to None (or absent) has no clean
# proxy -> the caller falls back to the broad benchmark.
SECTOR_ETF = {
    "semis_ai": "SMH",
    "software": "XLK",
    "energy": "XLE",
    "materials": "XLB",
    "financials": "XLF",
    "healthcare": "XLV",
}


def sector_for(symbol: str) -> Optional[str]:
    """Primary sector for a symbol, or None if unclassified."""
    if not symbol:
        return None
    return SYMBOL_SECTOR.get(str(symbol).strip().upper())


def etf_for_sector(sector: Optional[str]) -> Optional[str]:
    """Proxy ETF for a sector, or None if the sector has no clean benchmark."""
    if not sector:
        return None
    return SECTOR_ETF.get(sector)


def etf_for_symbol(symbol: str) -> Optional[str]:
    """Convenience: the proxy ETF a symbol's longs should be judged against."""
    return etf_for_sector(sector_for(symbol))
