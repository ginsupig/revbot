"""A/B backtest: current engine vs proven-model entry/exit rules.

Compares two configurations head-to-head on the same universe, same windows,
same occupancy-aware sim (one position per symbol + cooldowns):

  CURRENT  : entry at LB1 (1σ), RSI(14) ≤ 48 OR RI ≤ -0.5
             exit via trailing stop (1.5 ATR) + 3.5 ATR target backstop
             stop at 1.2 ATR

  PROVEN   : entry at LB2 (2σ), RSI(2) < 15 (extreme oversold, Connors-style)
             exit via signal: close > SMA(20) (mean reversion to the mean)
             no hard stop (portfolio-level DD circuit breaker only)
             time stop at 78 bars (same)

Both use the same trend filter (EMA50, 2% band) and safety gates. The only
differences are signal tightness and exit mechanism.

Two non-overlapping windows, cost sweep, per the research discipline.

Usage:
    python signal_exit_ab_backtest.py --days 180
    python signal_exit_ab_backtest.py --days 120 --costs 0 2 4 6 8
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.indicators import calculate_atr, calculate_rsi
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035
HOLD = 78  # time stop (bars)
COOLDOWN_BARS, LOSS_COOLDOWN_BARS = 6, 18


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def _current_signal_bars(bars):
    """Current engine: LB1 entry, RSI(14) ≤ 48 OR RI ≤ -0.5."""
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    idx = []
    for i in range(len(enriched)):
        slc = enriched.iloc[max(0, i - 1): i + 1]
        if engine.get_decision(slc).signal == "LONG_REVERSION":
            idx.append(i)
    return idx, calculate_atr(bars).to_numpy(), enriched


def _proven_signal_bars(bars):
    """Proven model: LB2 entry, RSI(2) < 15, trend filter still on."""
    engine = ReversionEngine(ReversionConfig(
        min_history=60,
        enable_shorts=False,
        band_std_1=2.0,       # use 2σ band as the entry zone (was 1σ)
        ri_threshold=-1.0,    # tighter RI (was -0.5)
        rsi_length=2,         # RSI(2) instead of RSI(14)
        rsi_max=15.0,         # extreme oversold only (was 48)
    ))
    enriched = engine.calculate_indicators(bars)
    idx = []
    for i in range(len(enriched)):
        slc = enriched.iloc[max(0, i - 1): i + 1]
        if engine.get_decision(slc).signal == "LONG_REVERSION":
            idx.append(i)
    return idx, calculate_atr(bars).to_numpy(), enriched


# ---------------------------------------------------------------------------
# Sim: current (trailing stop exit)
# ---------------------------------------------------------------------------

def _sim_current(signal_idx, high, low, close, atr_arr,
                 stop_m=1.20, tp_m=3.50, trail_m=1.50):
    """Live-occupancy sim with trailing stop exit (current system)."""
    sig = set(signal_idx)
    n = len(close)
    out = []
    in_pos = False
    entry = init_stop = target = highest = 0.0
    trail_dist = 0.0
    held = 0
    last_exit = -10**9
    last_loss = False
    for i in range(n):
        if in_pos:
            held += 1
            highest = max(highest, high[i])
            cur_stop = max(init_stop, highest - trail_dist)
            r = None
            if low[i] <= cur_stop:
                r = cur_stop / entry - 1.0
            elif high[i] >= target:
                r = target / entry - 1.0
            elif held >= HOLD:
                r = close[i] / entry - 1.0
            if r is not None:
                out.append(r)
                in_pos = False
                last_exit = i
                last_loss = r < 0
        if not in_pos and i in sig:
            cd = LOSS_COOLDOWN_BARS if last_loss else COOLDOWN_BARS
            if i - last_exit > cd:
                entry = close[i]
                atr = _atr(atr_arr, i, entry)
                init_stop = entry - stop_m * atr
                target = entry + tp_m * atr
                trail_dist = trail_m * atr
                highest = high[i]
                in_pos = True
                held = 0
    return out


# ---------------------------------------------------------------------------
# Sim: proven (signal-based exit — close > SMA20)
# ---------------------------------------------------------------------------

def _sim_proven(signal_idx, high, low, close, sma, atr_arr):
    """Live-occupancy sim with signal-based exit: close > SMA(20).

    No individual hard stop — only the time stop (78 bars) as safety net.
    This matches Connors' finding that stops destroy MR win rate.
    """
    sig = set(signal_idx)
    n = len(close)
    out = []
    in_pos = False
    entry = 0.0
    held = 0
    last_exit = -10**9
    last_loss = False
    for i in range(n):
        if in_pos:
            held += 1
            r = None
            # Signal-based exit: price reverted to the mean
            if close[i] > sma[i] and held >= 2:  # min 2 bars to avoid same-bar exit
                r = close[i] / entry - 1.0
            elif held >= HOLD:
                r = close[i] / entry - 1.0
            if r is not None:
                out.append(r)
                in_pos = False
                last_exit = i
                last_loss = r < 0
        if not in_pos and i in sig:
            cd = LOSS_COOLDOWN_BARS if last_loss else COOLDOWN_BARS
            if i - last_exit > cd:
                entry = close[i]
                in_pos = True
                held = 0
    return out


# ---------------------------------------------------------------------------
# Hybrid: proven entry + trailing stop exit (isolate entry vs exit effect)
# ---------------------------------------------------------------------------

def _sim_hybrid_entry(signal_idx, high, low, close, atr_arr,
                      stop_m=1.20, tp_m=3.50, trail_m=1.50):
    """Proven entry + current exit (trailing stop). Isolates entry effect."""
    return _sim_current(signal_idx, high, low, close, atr_arr,
                        stop_m, tp_m, trail_m)


def _sim_hybrid_exit(signal_idx, high, low, close, sma, atr_arr):
    """Current entry + signal exit. Isolates exit effect."""
    return _sim_proven(signal_idx, high, low, close, sma, atr_arr)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def _sharpe(arr):
    if arr.size < 2:
        return 0.0
    return float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0


def _max_dd(arr):
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B: current engine vs proven-model entry/exit (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 2, 4, 6, 8],
                        help="round-trip costs in bps to sweep (default 0 2 4 6 8)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else CORE
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    fmt = "%Y-%m-%d"
    windows = {
        "in-sample": ((end - timedelta(days=args.days)).strftime(fmt), end.strftime(fmt)),
        "oos":       ((end - timedelta(days=2 * args.days)).strftime(fmt),
                      (end - timedelta(days=args.days)).strftime(fmt)),
    }
    costs = [c / 10000.0 for c in args.costs]

    print(f"Signal-based exit A/B (live occupancy, long) | universe={len(symbols)} | {args.timeframe}")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")
    print()
    print("  CURRENT : LB1 entry, RSI(14)≤48|RI≤-0.5, trailing stop 1.5ATR + 3.5ATR target")
    print("  PROVEN  : LB2 entry, RSI(2)<15, signal exit (close>SMA20), no hard stop")
    print("  HYBRID-E: proven entry + trailing stop exit (isolate entry effect)")
    print("  HYBRID-X: current entry + signal exit (isolate exit effect)")

    configs = ["CURRENT", "PROVEN", "HYBRID-E", "HYBRID-X"]
    results = {(c, w): [] for c in configs for w in windows}

    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, args.timeframe)
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < 60:
                continue

            high = bars["high"].astype(float).to_numpy()
            low = bars["low"].astype(float).to_numpy()
            close = bars["close"].astype(float).to_numpy()

            # Current signals + enriched (for SMA)
            cur_idx, atr_arr, cur_enriched = _current_signal_bars(bars)
            sma_cur = cur_enriched["sma"].astype(float).to_numpy()

            # Proven signals + enriched
            prv_idx, _, prv_enriched = _proven_signal_bars(bars)
            sma_prv = prv_enriched["sma"].astype(float).to_numpy()

            # A: Current (trailing stop)
            results[("CURRENT", w)] += _sim_current(cur_idx, high, low, close, atr_arr)

            # B: Proven (signal exit)
            results[("PROVEN", w)] += _sim_proven(prv_idx, high, low, close, sma_prv, atr_arr)

            # C: Hybrid-entry (proven entry + trailing stop)
            results[("HYBRID-E", w)] += _sim_hybrid_entry(prv_idx, high, low, close, atr_arr)

            # D: Hybrid-exit (current entry + signal exit)
            results[("HYBRID-X", w)] += _sim_hybrid_exit(cur_idx, high, low, close, sma_cur, atr_arr)

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CONFIG':<11}{'WINDOW':<11}{'N':>6}{'grossEXP':>10}{'WIN':>7}{'SHARPE':>8}{'maxDD':>8}  {pf_hdr}")
    print("  " + "-" * (53 + 8 * len(args.costs)))
    for cfg in configs:
        for w in ("in-sample", "oos"):
            arr = np.asarray(results[(cfg, w)], dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<11}{w:<11}{0:>6}  (no trades)")
                continue
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            dd = _max_dd(arr)
            sr = _sharpe(arr)
            print(f"  {cfg:<11}{w:<11}{arr.size:>6}{arr.mean()*100:>9.3f}%"
                  f"{(arr>0).mean()*100:>6.1f}%{sr:>8.3f}{dd*100:>7.2f}%  {pfs}")
        print()

    print("  INTERPRETATION:")
    print("  - Compare CURRENT vs PROVEN across BOTH windows (OOS is the truth).")
    print("  - HYBRID-E isolates the tighter entry (same trailing exit).")
    print("  - HYBRID-X isolates the signal exit (same loose entry).")
    print("  - Higher WIN%, PF, and Sharpe in OOS = real edge, not curve-fit.")
    print("  - Fewer trades (lower N) in PROVEN is expected — tighter filter.")
    print("  - If PROVEN wins on per-trade EXP but loses on N, the edge is real")
    print("    but capacity-constrained — consider relaxing to RSI(2)<20 or LB1.5σ.")


if __name__ == "__main__":
    main()
