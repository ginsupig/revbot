"""Execution-tuning backtest: cost sweep x limit entry x wider target + trailing stop.

The occupancy audit showed the strategy has a thin POSITIVE gross edge
(~+0.035%/trade) that the ~8bps cost assumption eats. So the levers that matter
are execution cost and getting bigger moves per trade. This tests, on the
realistic live-occupancy sim (one position + symbol cooldown + loss-aware
re-entry brake), three execution changes and sweeps the cost so breakeven is
explicit:

  baseline : market entry, stop 1.2 / target 2.0 ATR, no trail (the audit config)
  tuned    : market entry, stop 1.2 / target 3.0 ATR + 1.5 ATR trailing stop
  limit    : tuned, but LIMIT entry 0.1 ATR below the signal close (maker fill;
             skipped if not touched within --fill-window bars)

For each config x window it prints gross expectancy and PF at a sweep of
round-trip costs (0/2/4/6/8 bps), so you can read where it crosses PF 1.0 and
match it to your real fills. Research-only; nothing wired.

Usage:
    python execution_tuning_backtest.py --days 180
    python execution_tuning_backtest.py --costs 0 1 2 3 4 --days 180
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.indicators import calculate_atr
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035
HOLD = 78
COOLDOWN_BARS, LOSS_COOLDOWN_BARS = 6, 18
LIMIT_OFFSET_ATR = 0.10
FILL_WINDOW = 3

# (entry_mode, stop_atr, target_atr, trail_atr or None)
CONFIGS = {
    "baseline": ("market", 1.20, 2.00, None),
    "tuned":    ("market", 1.20, 3.00, 1.50),
    "limit":    ("limit",  1.20, 3.00, 1.50),
}


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


def _signal_bars(bars):
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    idx = [i for i in range(len(enriched))
           if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION"]
    return idx, calculate_atr(bars).to_numpy()


def _sim_gross(signal_idx, high, low, close, atr_arr, entry_mode, stop_m, tp_m, trail_m,
               limit_off=LIMIT_OFFSET_ATR, fill_window=FILL_WINDOW) -> list:
    """Live-occupancy sim (one position + cooldowns) returning GROSS returns (no cost).
    Long only. Trailing stop ratchets the stop up to highest_high - trail_m*ATR."""
    sig = set(signal_idx)
    n = len(close)
    out = []
    in_pos = False
    entry = init_stop = target = trail_dist = highest = 0.0
    held = 0
    last_exit = -10 ** 9
    last_loss = False
    pending = False
    lim_price = 0.0
    lim_expiry = -1

    def _open(at_bar, px):
        nonlocal in_pos, entry, init_stop, target, trail_dist, highest, held
        atr = _atr(atr_arr, at_bar, px)
        entry = px
        init_stop = entry - stop_m * atr
        target = entry + tp_m * atr
        trail_dist = (trail_m * atr) if trail_m is not None else None
        highest = high[at_bar]
        in_pos = True
        held = 0

    for i in range(n):
        if in_pos:
            held += 1
            highest = max(highest, high[i])
            cur_stop = init_stop if trail_dist is None else max(init_stop, highest - trail_dist)
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

        if (not in_pos) and pending:
            if low[i] <= lim_price:
                _open(i, lim_price)
                pending = False
            elif i > lim_expiry:
                pending = False

        if (not in_pos) and (not pending) and (i in sig):
            cd = LOSS_COOLDOWN_BARS if last_loss else COOLDOWN_BARS
            if i - last_exit > cd:
                if entry_mode == "market":
                    _open(i, close[i])
                else:
                    lim_price = close[i] - limit_off * _atr(atr_arr, i, close[i])
                    lim_expiry = i + fill_window
                    pending = True
    return out


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution-tuning backtest (research-only)")
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
    print(f"Execution tuning (live occupancy, long) | universe={len(symbols)} | {args.timeframe}")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")
    print(f"  configs: baseline=stop1.2/tp2.0  tuned=stop1.2/tp3.0/trail1.5  "
          f"limit=tuned+limit{LIMIT_OFFSET_ATR}ATR(fill<={FILL_WINDOW})")

    gross = {(c, w): [] for c in CONFIGS for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, args.timeframe)
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < 60:
                continue
            idx, atr_arr = _signal_bars(bars)
            high = bars["high"].astype(float).to_numpy()
            low = bars["low"].astype(float).to_numpy()
            close = bars["close"].astype(float).to_numpy()
            for cfg, (mode, sm, tm, trm) in CONFIGS.items():
                gross[(cfg, w)] += _sim_gross(idx, high, low, close, atr_arr, mode, sm, tm, trm)

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CONFIG':<10}{'WINDOW':<11}{'N':>6}{'grossEXP':>10}{'WIN':>7}  {pf_hdr}")
    print("  " + "-" * (44 + 8 * len(args.costs)))
    for cfg in CONFIGS:
        for w in ("in-sample", "oos"):
            arr = np.asarray(gross[(cfg, w)], dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<10}{w:<11}{0:>6}  (no trades)")
                continue
            gexp = arr.mean()
            win = (arr > 0).mean()
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {cfg:<10}{w:<11}{arr.size:>6}{gexp*100:>9.3f}%{win*100:>6.1f}%  {pfs}")
        print()

    print("  Read across the PF@<bps> columns for where each config crosses 1.0. tuned/limit")
    print("  should push breakeven to a higher cost (bigger moves + maker fills absorb cost).")


if __name__ == "__main__":
    main()
