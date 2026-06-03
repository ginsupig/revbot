"""Pure (network-free) helpers for summarizing executed trades.

These functions take a list of normalized *fill* dicts and compute realized
PnL (FIFO, long/short aware) and per-symbol traded-notional weights. Keeping
them free of any Alpaca import means they are unit-testable without creds or
the SDK installed; the CLI in ``pnl_report.py`` does the actual fetching.

A normalized fill is a dict with:
    symbol: str
    side:   "buy" | "sell"
    qty:    float   (always positive)
    price:  float
    time:   str     (ISO8601, used only for ordering)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class SymbolRow:
    symbol: str
    realized_pnl: float
    traded_notional: float
    weight: float          # share of total traded notional, 0..1
    efficiency: float      # realized_pnl / traded_notional (return on notional)
    buy_qty: float
    sell_qty: float
    fills: int


@dataclass
class ScoreRow:
    symbol: str
    realized_pnl: float    # cumulative over the window (sum of daily)
    traded_notional: float
    efficiency: float      # cumulative realized / traded notional
    fills: int
    days_traded: int
    days_positive: int
    days_negative: int
    best_day: float
    worst_day: float
    last_day: float        # realized PnL on the most recent trading day in window
    probation: bool        # cumulative-negative AND bottom-N by return-on-notional


def _signed_qty(fill: dict) -> float:
    qty = abs(float(fill["qty"]))
    return qty if str(fill["side"]).lower() == "buy" else -qty


def realized_pnl_fifo(fills: Iterable[dict]) -> Dict[str, float]:
    """Realized PnL per symbol using FIFO lot matching.

    Handles both long (buy then sell) and short (sell then buy-to-cover)
    round-trips. Open inventory at the end contributes no realized PnL.
    """
    ordered = sorted(fills, key=lambda f: f.get("time") or "")
    # symbol -> deque of [signed_qty, price]; sign denotes long(+)/short(-) lots.
    lots: Dict[str, deque] = defaultdict(deque)
    realized: Dict[str, float] = defaultdict(float)

    for f in ordered:
        sym = f["symbol"]
        price = float(f["price"])
        qty = _signed_qty(f)
        book = lots[sym]

        # Consume opposing inventory first (closing existing position).
        while qty != 0 and book and (book[0][0] > 0) != (qty > 0):
            lot_qty, lot_price = book[0]
            match = min(abs(lot_qty), abs(qty))
            if lot_qty > 0:                      # closing a long
                realized[sym] += (price - lot_price) * match
            else:                                # covering a short
                realized[sym] += (lot_price - price) * match

            # Shrink the front lot toward zero; drop it if fully consumed.
            new_lot_qty = lot_qty - match if lot_qty > 0 else lot_qty + match
            if new_lot_qty == 0:
                book.popleft()
            else:
                book[0][0] = new_lot_qty
            # Shrink the incoming order toward zero.
            qty = qty - match if qty > 0 else qty + match

        # Whatever is left opens (or extends) a position in the same direction.
        if qty != 0:
            book.append([qty, price])

    return dict(realized)


def traded_notional(fills: Iterable[dict]) -> Dict[str, float]:
    """Gross dollar volume traded per symbol (sum of price * qty over fills)."""
    notional: Dict[str, float] = defaultdict(float)
    for f in fills:
        notional[f["symbol"]] += abs(float(f["qty"])) * float(f["price"])
    return dict(notional)


def symbol_weights(notional: Dict[str, float]) -> Dict[str, float]:
    """Each symbol's share of total traded notional (sums to 1.0, or 0 if empty)."""
    total = sum(notional.values())
    if total <= 0:
        return {sym: 0.0 for sym in notional}
    return {sym: val / total for sym, val in notional.items()}


def _fill_day(fill: dict) -> str:
    """Trading-day key from a fill's ISO timestamp (the date portion).

    US regular hours (13:30–20:00 UTC) all fall on a single UTC calendar date,
    so the date prefix is a safe trading-day key without pulling in a timezone
    dependency. Fills with no/short timestamp collapse into one "" bucket.
    """
    t = str(fill.get("time") or "")
    return t[:10] if len(t) >= 10 else ""


def realized_pnl_by_day(fills: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    """Realized PnL per symbol, broken out by trading day.

    The bot flattens every position at EOD, so each day's fills form a
    self-contained set of round-trips: running FIFO within a single day is
    exact and needs no cross-day inventory carry. Returns
    ``{symbol: {day: realized_pnl}}``.
    """
    by_day: Dict[str, List[dict]] = defaultdict(list)
    for f in fills:
        by_day[_fill_day(f)].append(f)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for day, day_fills in by_day.items():
        for sym, pnl in realized_pnl_fifo(day_fills).items():
            out[sym][day] = pnl
    return dict(out)


# Sort keys -> (SymbolRow attribute, descending?). "symbol" sorts A..Z.
SORT_KEYS = {
    "notional": ("traded_notional", True),
    "pnl": ("realized_pnl", True),
    "efficiency": ("efficiency", True),
    "weight": ("weight", True),
    "fills": ("fills", True),
    "symbol": ("symbol", False),
}


def build_report(fills: List[dict], sort_by: str = "notional") -> dict:
    """Assemble a per-symbol report.

    ``sort_by`` is one of ``SORT_KEYS`` (default ``notional``, descending).
    """
    pnl = realized_pnl_fifo(fills)
    notional = traded_notional(fills)
    weights = symbol_weights(notional)

    buys: Dict[str, float] = defaultdict(float)
    sells: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for f in fills:
        counts[f["symbol"]] += 1
        if str(f["side"]).lower() == "buy":
            buys[f["symbol"]] += abs(float(f["qty"]))
        else:
            sells[f["symbol"]] += abs(float(f["qty"]))

    rows = []
    for sym in notional:
        sym_notional = notional.get(sym, 0.0)
        sym_pnl = pnl.get(sym, 0.0)
        rows.append(
            SymbolRow(
                symbol=sym,
                realized_pnl=sym_pnl,
                traded_notional=sym_notional,
                weight=weights.get(sym, 0.0),
                efficiency=(sym_pnl / sym_notional) if sym_notional else 0.0,
                buy_qty=buys.get(sym, 0.0),
                sell_qty=sells.get(sym, 0.0),
                fills=counts.get(sym, 0),
            )
        )

    attr, desc = SORT_KEYS.get(sort_by, SORT_KEYS["notional"])
    rows.sort(key=lambda r: getattr(r, attr), reverse=desc)

    return {
        "rows": rows,
        "total_realized_pnl": sum(pnl.values()),
        "total_notional": sum(notional.values()),
        "total_fills": len(fills),
        "symbols": len(rows),
    }


def format_report(report: dict, days: int) -> str:
    rows: List[SymbolRow] = report["rows"]
    lines = [
        f"Trade report — last {days} days",
        f"  fills: {report['total_fills']}   symbols: {report['symbols']}",
        f"  total realized PnL: ${report['total_realized_pnl']:,.2f}",
        f"  total traded notional: ${report['total_notional']:,.2f}",
        "",
        f"  {'SYMBOL':<8}{'WEIGHT':>9}{'NOTIONAL':>16}{'REALIZED PnL':>16}{'RET/NOT':>10}{'FILLS':>7}",
        f"  {'-'*8:<8}{'-'*8:>9}{'-'*15:>16}{'-'*15:>16}{'-'*9:>10}{'-'*6:>7}",
    ]
    for r in rows:
        lines.append(
            f"  {r.symbol:<8}{r.weight*100:>8.2f}%"
            f"{r.traded_notional:>16,.2f}{r.realized_pnl:>16,.2f}"
            f"{r.efficiency*100:>9.2f}%{r.fills:>7}"
        )
    return "\n".join(lines)


def format_csv(report: dict) -> str:
    """Render the per-symbol rows as CSV (header + one line per symbol)."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["symbol", "weight", "traded_notional", "realized_pnl",
         "efficiency", "buy_qty", "sell_qty", "fills"]
    )
    for r in report["rows"]:
        writer.writerow(
            [r.symbol, f"{r.weight:.6f}", f"{r.traded_notional:.2f}",
             f"{r.realized_pnl:.2f}", f"{r.efficiency:.6f}",
             f"{r.buy_qty:g}", f"{r.sell_qty:g}", r.fills]
        )
    return buf.getvalue()


def build_scoreboard(fills: List[dict], bottom_n: int = 3) -> dict:
    """Per-symbol rolling scoreboard for the cut/keep decision.

    Beyond aggregate PnL, this surfaces *consistency*: how many days a name
    traded, its win/loss day split, worst single day, and most recent day. The
    cumulative total is summed from the daily breakdown so it reconciles
    exactly with the per-day cells.

    A name is flagged ``probation`` (a cut candidate) when it is both
    cumulative-negative over the window AND ranks in the bottom ``bottom_n`` by
    return-on-notional — i.e. it loses *and* loses efficiently per dollar
    churned. The flag is advisory; callers should still require a minimum
    number of trading days before acting.
    """
    by_day = realized_pnl_by_day(fills)
    notional = traded_notional(fills)

    counts: Dict[str, int] = defaultdict(int)
    for f in fills:
        counts[f["symbol"]] += 1

    rows: List[ScoreRow] = []
    for sym in notional:
        days = by_day.get(sym, {})
        daily = [days[d] for d in sorted(days)]   # ascending by date
        total = sum(daily)
        sym_notional = notional.get(sym, 0.0)
        rows.append(
            ScoreRow(
                symbol=sym,
                realized_pnl=total,
                traded_notional=sym_notional,
                efficiency=(total / sym_notional) if sym_notional else 0.0,
                fills=counts.get(sym, 0),
                days_traded=len(daily),
                days_positive=sum(1 for v in daily if v > 0),
                days_negative=sum(1 for v in daily if v < 0),
                best_day=max(daily) if daily else 0.0,
                worst_day=min(daily) if daily else 0.0,
                last_day=daily[-1] if daily else 0.0,
                probation=False,
            )
        )

    # Flag the worst-bleeding names: cumulative-negative, ranked by the most
    # negative return-on-notional (loses the most per dollar traded).
    losers = sorted(
        (r for r in rows if r.realized_pnl < 0), key=lambda r: r.efficiency
    )
    for r in losers[:bottom_n]:
        r.probation = True

    rows.sort(key=lambda r: r.realized_pnl)   # worst first — this is a watch view

    return {
        "rows": rows,
        "total_realized_pnl": sum(r.realized_pnl for r in rows),
        "total_notional": sum(notional.values()),
        "total_fills": len(fills),
        "symbols": len(rows),
        "days_in_window": len({_fill_day(f) for f in fills if _fill_day(f)}),
    }


def format_scoreboard(report: dict, days: int, min_days: int = 5) -> str:
    rows: List[ScoreRow] = report["rows"]
    n_days = report["days_in_window"]
    lines = [
        f"Rolling scoreboard — last {days} days ({n_days} trading day(s) with fills)",
        f"  fills: {report['total_fills']}   symbols: {report['symbols']}",
        f"  total realized PnL: ${report['total_realized_pnl']:,.2f}",
        "",
        f"  {'SYMBOL':<8}{'REALIZED':>13}{'RET/NOT':>10}{'DAYS':>6}{'W-L':>8}"
        f"{'WORST DAY':>13}{'LAST DAY':>13}{'FILLS':>7}  FLAG",
        f"  {'-'*8:<8}{'-'*12:>13}{'-'*9:>10}{'-'*5:>6}{'-'*7:>8}"
        f"{'-'*12:>13}{'-'*12:>13}{'-'*6:>7}  ----",
    ]
    for r in rows:
        wl = f"{r.days_positive}-{r.days_negative}"
        flag = "CUT?" if r.probation else ""
        lines.append(
            f"  {r.symbol:<8}{r.realized_pnl:>13,.2f}{r.efficiency*100:>9.2f}%"
            f"{r.days_traded:>6}{wl:>8}{r.worst_day:>13,.2f}"
            f"{r.last_day:>13,.2f}{r.fills:>7}  {flag}"
        )
    lines.append("")
    lines.append("  CUT? = cumulative-negative AND bottom-3 by return-on-notional.")
    if n_days < min_days:
        lines.append(
            f"  NOTE: only {n_days} trading day(s) of data — wait for >= {min_days} "
            "before acting on a flag (one bad session is noise)."
        )
    return "\n".join(lines)
