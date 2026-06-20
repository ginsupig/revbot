"""Causal daily name selector: trade only today's top-probability setups.

The frozen four-name allowlist (ARM/CRWD/MU/LRCX) is a snapshot of what carried
edge in two past windows; a name whose edge has decayed stays in the book anyway.
This is the adaptive replacement: each day, score every watchlist name on its
*recent, causally-available* setup performance and trade only the top-K above a
threshold — so the book follows live edge instead of a hindsight pick.

Mechanism (no lookahead — see CLAUDE.md / research.leakage):
  * For each symbol find the entry setups (default: the flush-bounce reference
    signal) and each setup's realized return under the tuned ATR exit, tagging
    the entry day and the EXIT day.
  * The score for a symbol on day d is the mean return of its setups that
    *closed strictly before d* within a trailing window of W days. The day being
    traded — and every setup that hasn't resolved yet — is invisible to the score.
  * Each day, among symbols that signal that day, trade the top-K by score above
    a threshold (>= min_trades of history required, else the name is skipped).

The selection knobs (W, K, threshold) are themselves a parameter sweep, so this
returns per-config trade returns split into a research window and an untouched
holdout — feed both to research.pipeline.evaluate_sweep and the ranker is gated
exactly like any entry engine (deflated Sharpe + fold robustness + holdout).
Selecting names this way does not escape multiple testing; it relocates it into
(W, K, threshold), which is why it must clear the same bar. Research-only.

CLI (needs alpaca + creds):
    python -m research.daily_ranker --universe watchlist --days 540
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .engines.reference import EXIT, ATR_FLOOR_PCT, COST_PCT, flush_bounce_signals
from .pipeline import evaluate_sweep, PipelineResult
from .run import format_report

from reversion_bot.indicators import calculate_atr

# Default entry: the flush-bounce reference setup on daily bars. Fixed here — the
# ranker sweeps the SELECTION params, not the entry; swap setups_fn to use the
# live engine's LONG_REVERSION instead.
DEFAULT_FLUSH = 0.05
DEFAULT_RSI_MAX = 40.0
DEFAULT_DROP_WINDOW = 3


def setup_outcomes(high, low, close, atr_arr, signal_idx, exit=EXIT,
                   open_arr=None, entry_lag=0) -> list:
    """Per-setup ``(entry_i, exit_i, net_return)`` — each signal simulated forward
    independently (occupancy-free; this feeds the score, not the traded book) under
    the tuned ATR bracket + trailing stop, net of round-trip cost.

    ``entry_lag`` shifts the fill off the signal bar: 0 = enter at the signal bar's
    close (default); 1 with ``open_arr`` = enter at the NEXT bar's open. The latter
    removes the same-bar fill artifact that makes intraday reversion look tradeable
    when it is really bid-ask bounce off the signal-bar close (see the 5Min PF~10
    mirage). ATR/stops are anchored on the signal bar; only the fill price moves."""
    n = len(close)
    out = []
    for i in signal_idx:
        ei = i + entry_lag
        if ei >= n:
            continue
        entry = float(open_arr[ei]) if (entry_lag and open_arr is not None) else close[ei]
        a = max(float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0, entry * ATR_FLOOR_PCT)
        init_stop = entry - exit["stop"] * a
        target = entry + exit["target"] * a
        trail_dist = exit["trail"] * a if exit.get("trail") else None
        highest = high[ei]
        end = min(ei + exit["hold"], n - 1)
        exit_i, ret = end, close[end] / entry - 1.0
        for j in range(ei + 1, end + 1):
            highest = max(highest, high[j])
            cur_stop = init_stop if trail_dist is None else max(init_stop, highest - trail_dist)
            if low[j] <= cur_stop:
                exit_i, ret = j, cur_stop / entry - 1.0
                break
            if high[j] >= target:
                exit_i, ret = j, target / entry - 1.0
                break
        out.append((int(i), int(exit_i), float(ret - COST_PCT)))
    return out


def default_setups(bars) -> list:
    """Flush-bounce signals + tuned-exit outcomes for one symbol's bars."""
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr_arr = calculate_atr(bars).to_numpy()
    sig = np.flatnonzero(
        flush_bounce_signals(bars, DEFAULT_FLUSH, DEFAULT_DROP_WINDOW, DEFAULT_RSI_MAX))
    return setup_outcomes(high, low, close, atr_arr, sig)


def day_ordinals(bars) -> np.ndarray:
    """Integer day-ordinal per bar for cross-symbol alignment. Uses a ``date``/
    ``timestamp`` column or a DatetimeIndex if present; else each bar is one day."""
    for col in ("date", "timestamp"):
        if col in getattr(bars, "columns", []):
            ts = pd.to_datetime(bars[col]).dt.normalize()
            return ts.map(lambda t: t.toordinal()).to_numpy()
    idx = getattr(bars, "index", None)
    if isinstance(idx, pd.DatetimeIndex):
        return np.array([t.toordinal() for t in idx.normalize()])
    return np.arange(len(bars))


def build_setups(symbol_bars, setups_fn=default_setups) -> dict:
    """``{sym: [(entry_ord, exit_ord, ret), ...]}`` — setups tagged with day ordinals.
    At most one setup per symbol per day (the first), enforcing one shot per name/day."""
    out = {}
    for sym, bars in symbol_bars.items():
        ords = day_ordinals(bars)
        seen_days = set()
        rows = []
        for entry_i, exit_i, ret in setups_fn(bars):
            d = int(ords[entry_i])
            if d in seen_days:
                continue
            seen_days.add(d)
            rows.append((d, int(ords[exit_i]), ret))
        out[sym] = sorted(rows)
    return out


def _score(setups, day, window, min_trades) -> "float | None":
    """Causal trailing-window mean for one symbol on ``day``: setups that closed in
    ``[day - window, day)``. ``None`` if fewer than ``min_trades`` resolved."""
    hist = [r for (eo, xo, r) in setups if day - window <= xo < day]
    if len(hist) < min_trades:
        return None
    return float(np.mean(hist))


def rank_sweep(symbol_setups, windows, ks, thresholds, min_trades=3,
               holdout_cut_ord=None) -> "tuple[dict, dict]":
    """Sweep (W, K, threshold) and return ``(research_returns, holdout_returns)``,
    each ``{config_key: [net_returns]}``. A trade lands in holdout when its entry
    day is >= ``holdout_cut_ord`` (split by date, not by symbol)."""
    # entries_by_day[d] = [(sym, ret), ...] for symbols signalling on day d
    entries_by_day = defaultdict(list)
    for sym, setups in symbol_setups.items():
        for (eo, xo, r) in setups:
            entries_by_day[eo].append((sym, r))
    days = sorted(entries_by_day)

    research, holdout = defaultdict(list), defaultdict(list)
    for W, K, thr in itertools.product(windows, ks, thresholds):
        key = f"W{W}_K{K}_t{thr:g}"
        for d in days:
            scored = []
            for sym, ret in entries_by_day[d]:
                s = _score(symbol_setups[sym], d, W, min_trades)
                if s is not None and s > thr:
                    scored.append((s, sym, ret))
            scored.sort(reverse=True)
            for _s, _sym, ret in scored[:K]:
                bucket = (holdout if holdout_cut_ord is not None and d >= holdout_cut_ord
                          else research)
                bucket[key].append(ret)
    return ({k: np.asarray(v, float) for k, v in research.items()},
            {k: np.asarray(v, float) for k, v in holdout.items()})


def run_daily_ranker(symbol_bars, windows, ks, thresholds, min_trades=3,
                     holdout_frac=0.25, setups_fn=default_setups) -> PipelineResult:
    """End-to-end: build setups, split a date holdout, sweep the selector, gate.
    ``symbol_bars`` maps symbol -> bars (with a date column/index). ``setups_fn``
    injected so the core is testable without a broker."""
    symbol_setups = build_setups(symbol_bars, setups_fn)
    all_entry_ords = sorted(eo for s in symbol_setups.values() for (eo, _x, _r) in s)
    if not all_entry_ords:
        return PipelineResult(None, 0, 0.0, 0.0, False, False, verdict="empty sweep")
    cut = all_entry_ords[int(round(len(all_entry_ords) * (1.0 - holdout_frac)))
                          if holdout_frac < 1 else len(all_entry_ords) - 1]
    research, holdout = rank_sweep(symbol_setups, windows, ks, thresholds,
                                   min_trades, holdout_cut_ord=cut)
    return evaluate_sweep(research, holdout)


UNIVERSES = {
    "watchlist": ["ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "APP", "ARM", "CRWD",
                  "MSFT", "LRCX", "FCX", "OXY", "CLF", "HIMS", "WDC", "COIN", "ADI",
                  "AMAT", "TSLA", "IONQ"],
    "keepers": ["ARM", "CRWD", "MU", "LRCX"],
}


def main() -> None:
    p = argparse.ArgumentParser(description="Causal daily name-ranker sweep (gated)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    p.add_argument("--universe", choices=list(UNIVERSES), default="watchlist")
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--windows", type=int, nargs="*", default=[20, 40, 60],
                   help="trailing score windows in days")
    p.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3],
                   help="names traded per day")
    p.add_argument("--thresholds", type=float, nargs="*", default=[0.0, 0.005],
                   help="min trailing score to trade")
    p.add_argument("--min-trades", type=int, default=3)
    p.add_argument("--holdout", type=float, default=0.25)
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

    symbol_bars = {}
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
        except Exception as e:
            print(f"  # {sym}: {e}")
            continue
        if bars is not None and len(bars) >= 30:
            symbol_bars[sym] = bars

    n_cfg = len(args.windows) * len(args.ks) * len(args.thresholds)
    print(f"Daily name-ranker | {len(symbol_bars)}/{len(symbols)} symbols | {start}->{end_s} | "
          f"{args.timeframe} | W={args.windows} K={args.ks} thr={args.thresholds} "
          f"({n_cfg} configs) | holdout={args.holdout:.0%}")
    res = run_daily_ranker(symbol_bars, args.windows, args.ks, args.thresholds,
                           args.min_trades, args.holdout)
    print("\n" + format_report(res))


if __name__ == "__main__":
    main()
