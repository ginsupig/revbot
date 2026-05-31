"""Per-symbol allowlist gate.

The autotuner scores every symbol out-of-sample. Only names whose OOS metrics
clear a threshold are written to ``TRADE_ALLOWLIST`` in the .env; the live bot
then trades the intersection of its scan/watchlist with that allowlist. This
keeps the bot from auto-trading names that were break-even or losing in the
most recent walk-forward, while still letting a strong name through.

The functions here are intentionally pure (no env / IO) so they can be unit
tested and reused by both ``autotune_run.py`` and ``main.py``.
"""
from __future__ import annotations

from typing import Iterable, Mapping


def select_allowlist(
    scores: Mapping[str, Mapping[str, float]],
    min_pf: float,
    min_sharpe: float = 0.0,
) -> list[str]:
    """Return symbols whose OOS metrics clear the thresholds.

    ``scores`` maps a symbol to its aggregated out-of-sample metrics, e.g.
    ``{"TQQQ": {"profit_factor": 2.53, "sharpe": 0.91}}``. A symbol is kept
    only if its profit factor is at least ``min_pf`` *and* its Sharpe is at
    least ``min_sharpe`` (the Sharpe floor rejects PF-just-above-1 names whose
    risk-adjusted return is still negative). Order is preserved.
    """
    keep = []
    for symbol, metrics in scores.items():
        pf = metrics.get("profit_factor", 0.0)
        sr = metrics.get("sharpe", 0.0)
        if pf >= min_pf and sr >= min_sharpe:
            keep.append(symbol)
    return keep


def parse_allowlist(raw: str | None) -> set[str] | None:
    """Parse a ``TRADE_ALLOWLIST`` env value into a set of symbols.

    Returns ``None`` when the variable is unset, which means *gating disabled*
    (backward compatible — trade everything). A set is returned when the
    variable is present, even if empty: an explicitly empty allowlist means
    "nothing passed the last tune, so trade nothing" (fail safe).
    """
    if raw is None:
        return None
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def parse_symbol_csv(raw: str | None) -> set[str]:
    """Parse a comma-separated symbol list into an uppercased set.

    Unlike :func:`parse_allowlist`, an unset/blank value yields an *empty* set
    (meaning "no change"), which is the right default for the optional
    force-include watchlist and force-exclude blocklist.
    """
    if not raw:
        return set()
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def apply_watchlist(
    symbols: Iterable[str],
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    """Force ``include`` names into ``symbols`` and drop ``exclude`` names.

    Runs *before* the allowlist gate so a curated name reaches the universe
    even when the liquidity scanner misses it (``include``), and a chronic
    loser never trades even when the scanner keeps surfacing it (``exclude``).
    Order is preserved (appended includes go last), duplicates are removed,
    and matching is case-insensitive. ``exclude`` wins over ``include``.
    """
    out: list[str] = []
    seen: set[str] = set()
    ex = {s.upper() for s in (exclude or set())}
    for sym in symbols:
        key = str(sym).upper()
        if key in ex or key in seen:
            continue
        out.append(sym)
        seen.add(key)
    for sym in sorted(include or set()):
        key = sym.upper()
        if key in ex or key in seen:
            continue
        out.append(sym)
        seen.add(key)
    return out



def filter_symbols(
    symbols: Iterable[str],
    allowlist: set[str] | None,
) -> tuple[list[str], list[str]]:
    """Split ``symbols`` into ``(kept, dropped)`` given an allowlist.

    ``allowlist is None`` disables gating (everything kept). An empty set drops
    everything. Comparison is case-insensitive; original symbols are returned.
    """
    if allowlist is None:
        return list(symbols), []
    kept: list[str] = []
    dropped: list[str] = []
    for symbol in symbols:
        (kept if str(symbol).upper() in allowlist else dropped).append(symbol)
    return kept, dropped
