"""Occupancy-realistic portfolio sim: what the candidate ACTUALLY draws down.

The gated sweep books every selected signal as an independent full-size trade —
occupancy-free, so on a day with 20 dips it holds 20 correlated falling knives at
once and reports a ~63% MC drawdown. The live bot never does that: it caps open
positions, holds one per symbol, and shares capital. This walks the calendar with
those constraints — at most ``max_positions`` open, filled by the scorer's daily
rank when a slot frees — and returns the real equity curve, max drawdown, and
Sharpe. The occupancy-free sweep overstates BOTH the return and the drawdown
(CLAUDE.md occupancy discipline); this is the honest read.

Research-only. CLI (needs alpaca + creds):
    python -m research.portfolio_sim --universe neutral --max-positions 4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np

from .daily_ranker import setup_outcomes, day_ordinals
from .engines.reference import EXIT
from .realtime_scorer import _candidates, UNIVERSES, ALL_SECTOR_OF, SECTOR_ETFS, liquid_filter

from reversion_bot.indicators import calculate_atr


def build_entries(symbol_bars, rsi_max=45.0) -> list:
    """``[(entry_ord, exit_ord, ret, sym, score)]`` across the universe — the
    reversion candidates with their bracket exit DAY and oversold-depth score."""
    out = []
    for sym, bars in symbol_bars.items():
        rows = _candidates(bars, rsi_max)            # (day_ord, entry_i, feat, ret)
        if not rows:
            continue
        ords = day_ordinals(bars)
        high = bars["high"].astype(float).to_numpy()
        low = bars["low"].astype(float).to_numpy()
        close = bars["close"].astype(float).to_numpy()
        atr = calculate_atr(bars).to_numpy()
        idx = [i for (_d, i, _f, _r) in rows]
        exit_ord = {e: int(ords[x]) for (e, x, _r) in setup_outcomes(high, low, close, atr, idx, EXIT)}
        for (d, i, feat, ret) in rows:
            out.append((int(d), exit_ord[i], float(ret), sym, float(feat["setup"])))
    return out


def _curve_stats(curve, n_trades, n_skipped) -> dict:
    eq = np.asarray(curve, float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([0.0])
    sd = rets.std(ddof=0)
    return dict(
        n_trades=int(n_trades), n_skipped=int(n_skipped),
        max_drawdown=float(dd.min()) if dd.size else 0.0,
        total_return=float(eq[-1] - 1.0),
        sharpe=float(rets.mean() / sd) if sd > 0 else 0.0,
        final_equity=float(eq[-1]),
        equity=eq.tolist(),
    )


def simulate_portfolio(entries, max_positions=4) -> dict:
    """Walk the calendar holding <= ``max_positions`` equal-weight positions, one
    per symbol, filling free slots by score (desc). Capital compounds; each position
    is sized equity/max_positions at entry and realized on exit. Returns the equity
    curve + max drawdown + Sharpe — the occupancy-realistic read."""
    if not entries:
        return dict(n_trades=0, n_skipped=0, max_drawdown=0.0, total_return=0.0,
                    sharpe=0.0, final_equity=1.0, equity=[1.0])
    days = sorted({d for (d, _x, _r, _s, _sc) in entries}
                  | {x for (_d, x, _r, _s, _sc) in entries})
    by_entry = {}
    for (d, x, r, s, sc) in entries:
        by_entry.setdefault(d, []).append((sc, x, r, s))

    equity = 1.0
    open_pos = []          # list of dicts: {exit, size, ret, sym}
    curve = []
    n_trades = n_skipped = 0
    for d in days:
        # 1) close positions whose bracket exits on/before today
        keep = []
        for p in open_pos:
            if p["exit"] <= d:
                equity += p["size"] * p["ret"]
            else:
                keep.append(p)
        open_pos = keep
        held = {p["sym"] for p in open_pos}
        # 2) fill free slots from today's candidates, highest score first
        for (sc, xo, r, s) in sorted(by_entry.get(d, []), reverse=True):
            if len(open_pos) >= max_positions:
                n_skipped += 1
                continue
            if s in held:                       # one position per symbol
                n_skipped += 1
                continue
            size = equity / max_positions
            if xo <= d:                         # same-day bracket exit: realize at once
                equity += size * r
            else:
                open_pos.append(dict(exit=xo, size=size, ret=r, sym=s))
                held.add(s)
            n_trades += 1
        curve.append(equity)
    # close any still-open at the last day's marked equity (already in curve)
    return _curve_stats(curve, n_trades, n_skipped)


def run_portfolio_sim(symbol_bars, max_positions=4, rsi_max=45.0) -> dict:
    return simulate_portfolio(build_entries(symbol_bars, rsi_max), max_positions)


def format_sim(stats: dict, max_positions: int) -> str:
    return "\n".join([
        f"  max_positions  : {max_positions} (one per symbol, equal weight, compounding)",
        f"  trades taken   : {stats['n_trades']}  (skipped {stats['n_skipped']} — slots full / dup symbol)",
        f"  total return   : {stats['total_return']*100:+.1f}%   final equity x{stats['final_equity']:.2f}",
        f"  MAX DRAWDOWN   : {stats['max_drawdown']*100:.1f}%",
        f"  daily Sharpe   : {stats['sharpe']:.3f}",
        "",
        "  Compare MAX DRAWDOWN to the occupancy-free MC maxDD_p95 (~-63%): the cap is the",
        "  risk layer the sweep ignored. Tradeable only if this drawdown is survivable.",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Occupancy-realistic portfolio sim (research-only)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    p.add_argument("--universe", choices=list(UNIVERSES), default="neutral")
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--max-positions", type=int, nargs="*", default=[4],
                   help="open-position cap(s) to sweep (live default 4)")
    p.add_argument("--rsi-max", type=float, default=45.0)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=20e6)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from run_real_backtest import fetch_alpaca_bars   # guarded: needs alpaca + creds

    symbols = [s.upper() for s in args.symbols] if args.symbols else UNIVERSES[args.universe]
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    def _fetch(sym):
        try:
            b = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            return b if b is not None and len(b) >= 30 else None
        except Exception as e:
            print(f"  # {sym}: {e}")
            return None

    raw = {s: b for s in symbols if (b := _fetch(s)) is not None}
    symbol_bars = {s: b for s, b in raw.items()
                   if liquid_filter(b, args.min_price, args.min_dollar_vol)}
    print(f"Portfolio sim | universe={args.universe} | {len(symbol_bars)}/{len(symbols)} names | "
          f"{start}->{end_s} | {args.timeframe}")
    entries = build_entries(symbol_bars, args.rsi_max)
    print(f"  {len(entries)} reversion candidates across the window\n")
    for n in args.max_positions:
        print(format_sim(simulate_portfolio(entries, n), n))
        print()


if __name__ == "__main__":
    main()
