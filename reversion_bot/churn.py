"""Per-symbol churn / trade-economics analysis from executed fills.

Answers the question that actually matters on wide-spread leveraged ETFs:
*is the edge per trade big enough to survive the cost per trade?* If a symbol's
average realized PnL per round trip (in bps of trip notional) is smaller than
your round-trip cost, you are paying the spread to lose money — trading less,
or not at all, is the edge.

Pure / network-free: consumes the same normalized fill dicts as
``trade_report`` (symbol, side, qty, price, time). The CLI in
``churn_report.py`` does the fetching.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from reversion_bot.trade_report import _signed_qty, realized_pnl_fifo, traded_notional


@dataclass
class ChurnRow:
    symbol: str
    round_trips: int
    realized_pnl: float
    pnl_per_round_trip: float
    avg_trip_notional: float
    pnl_per_trip_bps: float      # edge per trade, in bps of trip notional
    cost_bps: float              # assumed round-trip cost, for comparison
    churning: bool               # True when edge/trip < cost/trip (net loser to trade)


def round_trips_per_symbol(fills: Iterable[dict]) -> Dict[str, int]:
    """Count exit events per symbol (a fill that reduces/closes/flips inventory).

    Long-only buy→sell counts the sell as one round trip; a position closed in
    two sells counts as two. Works for shorts too (sell→buy-to-cover).
    """
    ordered = sorted(fills, key=lambda f: f.get("time") or "")
    pos: Dict[str, float] = defaultdict(float)
    trips: Dict[str, int] = defaultdict(int)
    for f in ordered:
        sym = f["symbol"]
        qty = _signed_qty(f)
        before = pos[sym]
        # Opposing fill against existing inventory == a closing (exit) event.
        if before != 0 and (before > 0) != (qty > 0):
            trips[sym] += 1
        pos[sym] = before + qty
    return dict(trips)


def build_churn_report(fills: List[dict], cost_bps: float = 5.0) -> dict:
    """Per-symbol trade economics, sorted worst-churn first.

    ``cost_bps`` is your assumed round-trip cost (spread + commission), used only
    as a comparison line — it does not alter the broker-reported realized PnL.
    """
    pnl = realized_pnl_fifo(fills)
    notional = traded_notional(fills)
    trips = round_trips_per_symbol(fills)

    rows: List[ChurnRow] = []
    for sym in notional:
        n_trips = trips.get(sym, 0)
        sym_pnl = pnl.get(sym, 0.0)
        # Trip notional ~ half of total traded notional (buy leg + sell leg).
        avg_trip_notional = (notional[sym] / 2.0 / n_trips) if n_trips else 0.0
        pnl_per_trip = (sym_pnl / n_trips) if n_trips else 0.0
        edge_bps = (pnl_per_trip / avg_trip_notional * 1e4) if avg_trip_notional else 0.0
        rows.append(
            ChurnRow(
                symbol=sym,
                round_trips=n_trips,
                realized_pnl=sym_pnl,
                pnl_per_round_trip=pnl_per_trip,
                avg_trip_notional=avg_trip_notional,
                pnl_per_trip_bps=edge_bps,
                cost_bps=cost_bps,
                churning=(n_trips > 0 and edge_bps < cost_bps),
            )
        )

    rows.sort(key=lambda r: r.pnl_per_trip_bps)  # worst edge-per-trade first
    return {
        "rows": rows,
        "total_round_trips": sum(trips.values()),
        "total_realized_pnl": sum(pnl.values()),
        "cost_bps": cost_bps,
    }


def format_churn_report(report: dict) -> str:
    rows: List[ChurnRow] = report["rows"]
    churners = [r for r in rows if r.churning]
    lines = [
        f"Churn / trade-economics report   (assumed round-trip cost: "
        f"{report['cost_bps']:.1f} bps)",
        f"  round trips: {report['total_round_trips']}   "
        f"realized PnL: ${report['total_realized_pnl']:,.2f}",
        f"  symbols trading below cost (churning): {len(churners)}",
        "",
        f"  {'SYMBOL':<8}{'TRIPS':>7}{'REAL PnL':>12}{'PnL/TRIP':>11}"
        f"{'EDGE bps':>10}{'':>4}",
        f"  {'-'*8:<8}{'-'*6:>7}{'-'*11:>12}{'-'*10:>11}{'-'*9:>10}{'':>4}",
    ]
    for r in rows:
        flag = "  <- churning" if r.churning else ""
        lines.append(
            f"  {r.symbol:<8}{r.round_trips:>7}{r.realized_pnl:>12,.2f}"
            f"{r.pnl_per_round_trip:>11,.2f}{r.pnl_per_trip_bps:>10.1f}{flag}"
        )
    return "\n".join(lines)
