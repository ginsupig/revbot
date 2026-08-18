"""Neutral large-cap universe for the validated broad-reversion basket.

The research arc (research/portfolio_sim.py) converged on a simple, durable edge:
a broad daily oversold-reversion basket on a *neutral, survivorship-controlled*
large-cap universe — laggards included, not the curated momentum watchlist —
held at the live 4-position cap with ATR-risk sizing. Two non-overlapping OOS
windows, net of cost, came back ~+20%/window at <7% mark-to-market drawdown.

This ships that universe so the live bot can trade it behind the
USE_BROAD_REVERSION_UNIVERSE flag (default OFF) instead of hand-maintaining 90
names in TRADE_WATCHLIST. The occupancy cap, ATR-risk sizing and trailing stop in
the existing risk/governor/execution stack are unchanged — this only swaps the
*universe* the engine scans. The name set is kept in sync with the research
sector map (research.realtime_scorer.NEUTRAL_SECTOR_OF) by a test.
"""
from __future__ import annotations

from typing import List, Sequence

# Broad neutral large-caps across all 11 SPDR sectors, deliberately including
# laggards (INTC/T/VZ/PFE/BA/CLF/F): the edge must come from ranking the dip, not
# from knowing the names. Mirrors research.realtime_scorer.NEUTRAL_SECTOR_OF keys.
NEUTRAL_LARGECAP: List[str] = [
    # info technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "ACN", "ADBE", "AMD",
    "INTC", "QCOM", "TXN", "IBM", "MU", "AMAT", "LRCX", "ADI", "CRWD",
    # ai and memory
    "WDC", "SNDK", "SKHY", "MRVL", "NOW", "SMCI", "ARM",
    # communication services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "F",
    # consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "MO", "CL", "MDLZ",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "BLK",
    # health care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "BMY", "AMGN",
    # energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "GEV",
    # industrials
    "CAT", "BA", "HON", "UPS", "GE", "RTX", "LMT", "DE", "UNP",
    # materials
    "LIN", "FCX", "NEM", "APD", "SHW", "DOW", "CLF",
    # utilities
    "NEE", "DUK", "SO", "D", "AEP",
]


def broad_reversion_universe() -> List[str]:
    """The validated neutral large-cap universe (a fresh copy each call)."""
    return list(NEUTRAL_LARGECAP)


def select_base_universe(scanned: Sequence[str], fallback: Sequence[str],
                         use_broad: bool) -> List[str]:
    """Resolve the base symbol list before watchlist/blocklist/allowlist wrapping:

    * ``use_broad`` -> the validated neutral large-cap basket (scanner overridden),
    * else the scanner output, or the curated ``fallback`` if the scan came back empty.
    """
    if use_broad:
        return broad_reversion_universe()
    return list(scanned) if scanned else list(fallback)
