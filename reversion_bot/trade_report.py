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


def realized_pnl_events(fills: Iterable[dict]) -> List[dict]:
    """Per-closing-fill realized-PnL events from a fill stream.

    Runs the same global long/short-aware FIFO as ``realized_pnl_fifo`` but,
    instead of only the per-symbol total, emits one event for every fill that
    realizes PnL (i.e. consumes opposing inventory). Each event is::

        {"symbol": str, "time": str, "realized_pnl": float}

    where ``realized_pnl`` is the PnL booked by that single closing fill (summed
    over every lot it matched). A fill that only opens or extends a position
    produces no event. The events' realized_pnl sum to ``realized_pnl_fifo`` over
    the same fills. This is the per-round-trip granularity that outcome-based
    threshold adaptation needs: one win/loss per closed trade, not a daily net.
    """
    ordered = sorted(fills, key=lambda f: f.get("time") or "")
    lots: Dict[str, deque] = defaultdict(deque)
    events: List[dict] = []

    for f in ordered:
        sym = f["symbol"]
        price = float(f["price"])
        qty = _signed_qty(f)
        book = lots[sym]
        booked = 0.0
        closed_any = False

        # Consume opposing inventory first (closing an existing position).
        while qty != 0 and book and (book[0][0] > 0) != (qty > 0):
            lot_qty, lot_price = book[0]
            match = min(abs(lot_qty), abs(qty))
            if lot_qty > 0:                      # closing a long
                booked += (price - lot_price) * match
            else:                                # covering a short
                booked += (lot_price - price) * match

            new_lot_qty = lot_qty - match if lot_qty > 0 else lot_qty + match
            if new_lot_qty == 0:
                book.popleft()
            else:
                book[0][0] = new_lot_qty
            qty = qty - match if qty > 0 else qty + match
            closed_any = True

        if closed_any:
            events.append(
                {"symbol": sym, "time": str(f.get("time") or ""), "realized_pnl": booked}
            )

        # Whatever is left opens (or extends) a position in the same direction.
        if qty != 0:
            book.append([qty, price])

    return events



def round_trips(fills: Iterable[dict]) -> List[dict]:
    """Per-lot closed round-trips from a fill stream (the diagnosis granularity).

    Like ``realized_pnl_events`` this runs the global long/short-aware FIFO, but
    it emits one record for every *lot match* — i.e. every (entry lot, closing
    fill) pair — carrying the entry and exit prices/times so a loser can be
    localized to specific trades rather than a daily net. Each record is::

        {
            "symbol": str,
            "direction": "long" | "short",
            "qty": float,
            "entry_price": float, "exit_price": float,
            "entry_time": str, "exit_time": str,
            "pnl": float,
            "hold_seconds": float | None,   # None if either timestamp is unparseable
            "return_pct": float,            # pnl / (entry_price * qty), signed
        }

    Records are returned in closing order. Their ``pnl`` sums to
    ``realized_pnl_fifo`` over the same fills. Inventory still open at the end
    produces no record.
    """
    ordered = sorted(fills, key=lambda f: f.get("time") or "")
    # symbol -> deque of [signed_qty, price, time]
    lots: Dict[str, deque] = defaultdict(deque)
    trips: List[dict] = []

    for f in ordered:
        sym = f["symbol"]
        price = float(f["price"])
        qty = _signed_qty(f)
        exit_time = str(f.get("time") or "")
        book = lots[sym]

        while qty != 0 and book and (book[0][0] > 0) != (qty > 0):
            lot_qty, lot_price, lot_time = book[0]
            match = min(abs(lot_qty), abs(qty))
            if lot_qty > 0:                      # closing a long
                pnl = (price - lot_price) * match
                direction = "long"
            else:                                # covering a short
                pnl = (lot_price - price) * match
                direction = "short"

            cost = lot_price * match
            trips.append(
                {
                    "symbol": sym,
                    "direction": direction,
                    "qty": match,
                    "entry_price": lot_price,
                    "exit_price": price,
                    "entry_time": lot_time,
                    "exit_time": exit_time,
                    "pnl": pnl,
                    "hold_seconds": _hold_seconds(lot_time, exit_time),
                    "return_pct": (pnl / cost) if cost else 0.0,
                }
            )

            new_lot_qty = lot_qty - match if lot_qty > 0 else lot_qty + match
            if new_lot_qty == 0:
                book.popleft()
            else:
                book[0][0] = new_lot_qty
            qty = qty - match if qty > 0 else qty + match

        if qty != 0:
            book.append([qty, price, exit_time])

    return trips


def _hold_seconds(entry_time: str, exit_time: str):
    """Seconds between two ISO8601 timestamps, or None if either is unparseable."""
    from datetime import datetime

    def _parse(t: str):
        t = (t or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(t)
        except ValueError:
            return None

    a, b = _parse(entry_time), _parse(exit_time)
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def _fmt_hold(seconds) -> str:
    """Compact human hold time: '45s', '12m', '3.4h', '2.1d', or '-'."""
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def build_trades(fills: List[dict], symbol: str | None = None) -> dict:
    """Per-round-trip drill-down, optionally filtered to one ``symbol``.

    Surfaces the round-trips behind a symbol's realized PnL, sorted worst-first
    so the biggest bleeders are at the top, with win/loss and cost summaries.
    """
    sym = symbol.upper() if symbol else None
    trips = [t for t in round_trips(fills) if sym is None or t["symbol"] == sym]
    trips.sort(key=lambda t: t["pnl"])   # worst first

    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] < 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = sum(t["pnl"] for t in losses)

    return {
        "symbol": sym,
        "trips": trips,
        "total_pnl": sum(t["pnl"] for t in trips),
        "round_trips": len(trips),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(trips)) if trips else 0.0,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / -gross_loss) if gross_loss else float("inf"),
    }


def format_trades(report: dict, days: int) -> str:
    trips = report["trips"]
    scope = report["symbol"] or "ALL SYMBOLS"
    pf = report["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    lines = [
        f"Round-trip drill-down — {scope}, last {days} days",
        f"  round-trips: {report['round_trips']}   "
        f"W-L: {report['wins']}-{report['losses']}   "
        f"win-rate: {report['win_rate']*100:.0f}%",
        f"  realized PnL: ${report['total_pnl']:,.2f}   "
        f"profit factor: {pf_str}",
        f"  avg win: ${report['avg_win']:,.2f}   "
        f"avg loss: ${report['avg_loss']:,.2f}",
        "",
        f"  {'SYMBOL':<8}{'DIR':<6}{'QTY':>8}{'ENTRY':>10}{'EXIT':>10}"
        f"{'RET':>8}{'HOLD':>7}{'PnL':>12}",
        f"  {'-'*8:<8}{'-'*5:<6}{'-'*7:>8}{'-'*9:>10}{'-'*9:>10}"
        f"{'-'*7:>8}{'-'*6:>7}{'-'*11:>12}",
    ]
    for t in trips:
        lines.append(
            f"  {t['symbol']:<8}{t['direction']:<6}{t['qty']:>8.2f}"
            f"{t['entry_price']:>10.2f}{t['exit_price']:>10.2f}"
            f"{t['return_pct']*100:>7.2f}%{_fmt_hold(t['hold_seconds']):>7}"
            f"{t['pnl']:>12,.2f}"
        )
    return "\n".join(lines)


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
    """Trading-day key (US/Eastern date) from a fill's ISO timestamp.

    The old UTC date prefix booked any extended-hours fill at/after 8pm ET
    (= 00:00 UTC in summer) to the NEXT day. Falls back to the raw date
    prefix when the timestamp doesn't parse; no/short timestamps collapse
    into one "" bucket.
    """
    t = str(fill.get("time") or "")
    try:
        from zoneinfo import ZoneInfo
        from .performance import parse_ts
        dt = parse_ts(t)
        if dt is not None:
            return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        pass
    return t[:10] if len(t) >= 10 else ""


def realized_pnl_by_day(fills: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    """Realized PnL per symbol, attributed to the day each position was *closed*.

    Runs a single global FIFO across the whole window (positions may be held
    overnight — the bot does not always flatten at EOD), booking each realized
    chunk on the date of the closing fill. This makes the per-day totals sum
    exactly to ``realized_pnl_fifo`` over the same fills; a naive
    partition-by-day-then-FIFO would silently drop every round-trip that crosses
    a day boundary. Returns ``{symbol: {day: realized_pnl}}``.
    """
    ordered = sorted(fills, key=lambda f: f.get("time") or "")
    lots: Dict[str, deque] = defaultdict(deque)
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for f in ordered:
        sym = f["symbol"]
        price = float(f["price"])
        qty = _signed_qty(f)
        day = _fill_day(f)
        book = lots[sym]

        # Close opposing inventory first; realized PnL is booked on this fill's day.
        while qty != 0 and book and (book[0][0] > 0) != (qty > 0):
            lot_qty, lot_price = book[0]
            match = min(abs(lot_qty), abs(qty))
            if lot_qty > 0:                      # closing a long
                out[sym][day] += (price - lot_price) * match
            else:                                # covering a short
                out[sym][day] += (lot_price - price) * match

            new_lot_qty = lot_qty - match if lot_qty > 0 else lot_qty + match
            if new_lot_qty == 0:
                book.popleft()
            else:
                book[0][0] = new_lot_qty
            qty = qty - match if qty > 0 else qty + match

        if qty != 0:
            book.append([qty, price])

    return {sym: dict(days) for sym, days in out.items()}


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


def build_daily_timeline(fills: List[dict], newest_first: bool = True) -> dict:
    """Per-day realized PnL with the tickers traded each day.

    The other axis from the scoreboard: one entry per trading day, each listing
    that day's total realized PnL, fill count, and the symbols traded (with their
    individual realized PnL, sorted by contribution).
    """
    by_sym_day = realized_pnl_by_day(fills)        # {symbol: {day: pnl}}
    per_day: Dict[str, Dict[str, float]] = defaultdict(dict)
    for sym, day_map in by_sym_day.items():
        for day, pnl in day_map.items():
            per_day[day][sym] = pnl

    fills_per_day: Dict[str, int] = defaultdict(int)
    for f in fills:
        fills_per_day[_fill_day(f)] += 1

    days = sorted(per_day.keys(), reverse=newest_first)
    rows = []
    for day in days:
        sym_pnls = sorted(per_day[day].items(), key=lambda kv: kv[1], reverse=True)
        rows.append(
            {
                "date": day,
                "realized_pnl": sum(v for _, v in sym_pnls),
                "fills": fills_per_day.get(day, 0),
                "symbols": sym_pnls,                # list of (symbol, pnl)
            }
        )

    return {
        "rows": rows,
        "total_realized_pnl": sum(r["realized_pnl"] for r in rows),
        "total_fills": len(fills),
        "days_in_window": len(rows),
    }


def format_daily_timeline(report: dict, days: int) -> str:
    rows = report["rows"]
    lines = [
        f"Daily PnL timeline — last {days} days ({report['days_in_window']} trading day(s))",
        f"  total realized PnL: ${report['total_realized_pnl']:,.2f}   fills: {report['total_fills']}",
        "",
        f"  {'DATE':<12}{'REALIZED':>13}{'FILLS':>7}   TICKERS (realized PnL)",
        f"  {'-'*10:<12}{'-'*12:>13}{'-'*6:>7}   {'-'*40}",
    ]
    for r in rows:
        tickers = ", ".join(f"{sym} {pnl:+,.0f}" for sym, pnl in r["symbols"])
        lines.append(
            f"  {r['date']:<12}{r['realized_pnl']:>13,.2f}{r['fills']:>7}   {tickers}"
        )
    return "\n".join(lines)


def format_daily_csv(report: dict) -> str:
    """Long-format CSV: one row per (date, symbol) plus a daily TOTAL row."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "symbol", "realized_pnl"])
    for r in report["rows"]:
        for sym, pnl in r["symbols"]:
            writer.writerow([r["date"], sym, f"{pnl:.2f}"])
        writer.writerow([r["date"], "TOTAL", f"{r['realized_pnl']:.2f}"])
    return buf.getvalue()
