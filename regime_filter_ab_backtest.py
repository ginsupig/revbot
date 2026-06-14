"""A/B backtest: a market-regime filter on mean-reversion entries.

Base PF swung 0.86 -> 0.79 -> 0.95 across recent windows: the *market regime*
moves the whole book far more than any per-symbol entry feature. The engine's
risk-off check is symbol-local; this tests a MARKET-WIDE gate built from a daily
SPY regime, applied to every LONG_REVERSION entry.

Two design choices are left to the data, not pre-picked:

  Action  -- base (take all) vs SUPPRESS (skip in hostile regime) vs HALF (0.5x).
  Trigger -- a LADDER, so we see which leg does the work, like the rvol sweep:
               trend          : SPY < 50-day EMA
               trend+adx      : + SPY ADX >= 20 (a real down-trend, not chop)
               trend+vol      : + SPY ATR% expanding
               trend+adx+vol  : all three

The daily regime is shifted one day (known at the OPEN of the entry day) so there
is no look-ahead. Per-setup expectancy on the standard bracket; research-only,
nothing wired into the engine. The eventual home is the governor / market_regime
layer (suppress_longs_*), never get_decision. Validate OOS (--end).

Usage:
    python regime_filter_ab_backtest.py --days 180
    python regime_filter_ab_backtest.py --end 2025-12-15 --days 180
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.market_regime import classify_regime_series
from reversion_bot.analytics import ulcer_index, worst_losing_streak
from run_real_backtest import fetch_alpaca_bars

CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]
BENCHMARK = "SPY"

ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
STD = dict(stop=1.20, target=2.00, hold=78)
WARMUP_DAYS = 120          # extra daily history so the 50-EMA/ADX are warm at window start

VARIANTS = {
    "trend": ("trend_down",),
    "trend+adx": ("trend_down", "adx_high"),
    "trend+vol": ("trend_down", "vol_up"),
    "trend+adx+vol": ("trend_down", "adx_high", "vol_up"),
}


def _outcome(high, low, close, i, entry, atr, br) -> float:
    stop = entry - atr * br["stop"]
    target = entry + atr * br["target"]
    end = min(i + br["hold"], len(close) - 1)
    for j in range(i + 1, end + 1):
        if low[j] <= stop:
            return (stop / entry - 1.0) - COST_PCT
        if high[j] >= target:
            return (target / entry - 1.0) - COST_PCT
    return (close[end] / entry - 1.0) - COST_PCT


def _daily_regime_by_date(start, end_s, timeframe_warmup_start) -> dict:
    """{date -> (trend_down, adx_high, vol_up)} from daily SPY, shifted one day
    (so the flag for day D reflects data through D-1: no look-ahead)."""
    spy = fetch_alpaca_bars(BENCHMARK, timeframe_warmup_start, end_s, "1Day")
    if spy is None or len(spy) < 60:
        return {}
    reg = classify_regime_series(spy).shift(1).fillna(False).astype(bool)
    idx = pd.to_datetime(spy.index)
    out = {}
    for ts, (td, ah, vu) in zip(idx, reg[["trend_down", "adx_high", "vol_up"]].to_numpy()):
        out[ts.date()] = (bool(td), bool(ah), bool(vu))
    return out


def _simulate(bars, regime_by_date) -> list:
    """Per-entry records: (base_return, trend_down, adx_high, vol_up)."""
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr_col = enriched["atr"].to_numpy()
    index = pd.to_datetime(bars.index)

    out = []
    for i in range(len(enriched)):
        if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal != "LONG_REVERSION":
            continue
        c = close[i]
        atr = max(float(atr_col[i]) if not np.isnan(atr_col[i]) else 0.0, c * ATR_FLOOR_PCT)
        r = _outcome(high, low, close, i, c, atr, STD)
        td, ah, vu = regime_by_date.get(index[i].date(), (False, False, False))
        out.append((r, td, ah, vu))
    return out


def _metrics(stream) -> dict:
    arr = np.asarray(stream, dtype=float)
    if arr.size == 0:
        return dict(n=0, net=0.0, pf=0.0, sharpe=0.0, max_dd=0.0, ulcer=0.0, streak=0)
    wins, losses = arr[arr > 0].sum(), -arr[arr < 0].sum()
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    std = arr.std(ddof=0)
    return dict(
        n=int(arr.size), net=float(arr.sum()),
        pf=float(wins / losses) if losses > 0 else (10.0 if wins > 0 else 0.0),
        sharpe=float(arr.mean() / std) if std > 0 else 0.0,
        max_dd=float(((eq - peak) / peak).min()),
        ulcer=ulcer_index(eq), streak=worst_losing_streak(arr),
    )


def _row(label, m):
    return (f"  {label:<22}{m['n']:>6}{m['net']:>10.4f}{m['pf']:>7.2f}{m['sharpe']:>8.3f}"
            f"{m['max_dd']*100:>9.2f}%{m['ulcer']:>8.2f}{m['streak']:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-regime filter A/B (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=180, help="lookback in days (default 180)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; OOS)")
    parser.add_argument("--timeframe", default="5Min")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else CORE_UNIVERSE
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    warm = (end - timedelta(days=args.days + WARMUP_DAYS)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    print(f"Market-regime filter A/B (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe} | benchmark={BENCHMARK} (daily, shifted 1d)")

    regime_by_date = _daily_regime_by_date(start, end_s, warm)
    if not regime_by_date:
        print("\n  could not build SPY daily regime (no benchmark data).")
        return

    records = []
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                continue
            records += _simulate(bars, regime_by_date)
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    if not records:
        print("\n  no signals in window.")
        return

    base = np.array([r[0] for r in records])
    flags = {"trend_down": np.array([r[1] for r in records]),
             "adx_high": np.array([r[2] for r in records]),
             "vol_up": np.array([r[3] for r in records])}

    print(f"\n  signals={len(records)}")
    header = (f"  {'ARM':<22}{'N':>6}{'NET':>10}{'PF':>7}{'SHARPE':>8}{'MAXDD':>10}{'ULCER':>8}{'STREAK':>8}")
    print("\n" + header)
    print(_row("base (take all)", _metrics(base)))

    for name, legs in VARIANTS.items():
        hostile = np.logical_and.reduce([flags[l] for l in legs])
        frac = float(hostile.mean())
        keep = base[~hostile]                       # suppress: trade only benign regime
        half = np.where(hostile, base * 0.5, base)  # half-size hostile entries
        print(f"\n  [{name}]  hostile={frac*100:.0f}% of entries")
        print(_row(f"  suppress", _metrics(keep)))
        print(_row(f"  half-size", _metrics(half)))

    print("\n  Want: an arm with higher PF/Sharpe and shallower MAXDD/ulcer than base.")
    print("  suppress that only trims N without lifting PF == the cut trades weren't")
    print("  losers (no edge). Validate the winning arm OOS (--end) before wiring into")
    print("  the governor / market_regime suppressor (never get_decision).")


if __name__ == "__main__":
    main()
