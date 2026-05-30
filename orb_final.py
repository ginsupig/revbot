"""Final ORB study: longs-only, cut the cost tax. Then we call it.

Lever 1: higher timeframes (5m/15m/30m) -> fewer trades -> less cost drag.
Lever 2: trade the UNDERLYING (tighter spreads/lower cost) vs the leveraged ETF,
         signal always from the underlying's opening range.
All: longs only (shorts hurt in the up-tape), wider 3.0-ATR stop, BE@1R + 2%
trail, scale-outs 25%/25%/rest into the close, EOD-flat, costs, 80% deploy.
yfinance 60d cap on intraday; single regime -- directional read only.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.indicators import calculate_atr
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

PAIRS = {"QQQ": "TQQQ", "SOXX": "SOXL", "XLK": "TECL", "NVDA": "NVDL", "TSLA": "TSLL",
         "AAPL": "AAPU", "META": "METU", "GOOGL": "GGLL", "MSFT": "MSFU", "AMZN": "AMZU"}
TRAIL_PCT, STOP_MULT, DEPLOY = 0.02, 3.0, 0.80
# OR length and no-entry tail, in BARS, per timeframe (keep ~15min OR, ~10min tail).
TF_CFG = {"5m": dict(or_bars=3, tail=2), "15m": dict(or_bars=1, tail=1), "30m": dict(or_bars=1, tail=1)}
# Cost per side: ETFs wide (8/5 bps), underlyings tighter (3/2 bps).
COST = {"etf": (8 / 1e4, 5 / 1e4), "under": (3 / 1e4, 2 / 1e4)}
SCALE1, SCALE2, FINAL = 15 * 60, 15 * 60 + 30, 15 * 60 + 55


def load(t, interval):
    df = yf.download(t, period="60d", interval=interval, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime": "date", "Date": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def prep(under, etf, interval):
    u, e = load(under, interval), load(etf, interval)
    if u is None or e is None:
        return None
    m = pd.merge(u, e, on="date", suffixes=("_u", "_e")).reset_index(drop=True)
    if len(m) < 120:
        return None
    for sfx in ("u", "e"):
        m[f"atr_{sfx}"] = calculate_atr(
            m.rename(columns={f"high_{sfx}": "high", f"low_{sfx}": "low", f"close_{sfx}": "close"}), 14)
    m["day"] = m["date"].dt.date
    et = m["date"].dt.tz_convert("America/New_York")
    m["etmin"] = et.dt.hour * 60 + et.dt.minute
    return m.dropna().reset_index(drop=True)


def simulate(m, instr, or_bars, tail):
    """instr 'etf' or 'under': which series we trade (signal always from underlying)."""
    uh, ul, uc = m["high_u"].to_numpy(), m["low_u"].to_numpy(), m["close_u"].to_numpy()
    sfx = "e" if instr == "etf" else "u"
    po = m[f"close_{sfx}"].to_numpy(); ph = m[f"high_{sfx}"].to_numpy()
    pl = m[f"low_{sfx}"].to_numpy(); atr = m[f"atr_{sfx}"].to_numpy()
    day = m["day"].to_numpy(); etm = m["etmin"].to_numpy()
    ec, xc = COST[instr]
    trades = []
    i = 0
    while i < len(m):
        j = i
        while j < len(m) and day[j] == day[i]:
            j += 1
        idx = list(range(i, j)); i = j
        if len(idx) < or_bars + 2:
            continue
        orh = uh[idx[:or_bars]].max()
        for k in idx[or_bars: len(idx) - tail]:
            if uc[k] <= orh:                  # longs only: break of OR high
                continue
            entry = po[k]; a = atr[k]
            if a <= 0:
                break
            stop = entry - STOP_MULT * a; R = entry - stop
            rem, realized, cost, peak, be, s1, s2 = 1.0, 0.0, ec, entry, False, False, False

            def take(frac, px):
                nonlocal realized, rem, cost
                realized += frac * (px / entry - 1); cost += frac * xc; rem -= frac

            for n in range(k + 1, idx[-1] + 1):
                peak = max(peak, ph[n])
                if not be and ph[n] >= entry + R:
                    be = True; stop = max(stop, entry)
                if be:
                    stop = max(stop, peak * (1 - TRAIL_PCT))
                if pl[n] <= stop:
                    take(rem, stop); break
                if not s1 and etm[n] >= SCALE1 and rem > 0:
                    take(min(0.25, rem), po[n]); s1 = True
                if not s2 and etm[n] >= SCALE2 and rem > 0:
                    take(min(0.25, rem), po[n]); s2 = True
                if (etm[n] >= FINAL or n == idx[-1]) and rem > 0:
                    take(rem, po[n]); break
            trades.append(realized - cost)
            break
    return trades


def metrics(tr, deploy=False):
    if not tr:
        return dict(n=0, win=float("nan"), ret=0.0, pf=float("nan"), sharpe=float("nan"), dd=0.0)
    a = np.array(tr); pr = pd.DataFrame({"strategy_return": a, "pnl": a})
    ret = ((1 + a).cumprod()[-1] - 1) * 100 * (DEPLOY if deploy else 1)
    return dict(n=len(a), win=round((a > 0).mean() * 100, 1), ret=round(ret, 1),
                pf=round(profit_factor(pr), 2), sharpe=round(sharpe_ratio(pr), 2),
                dd=round(max_drawdown(pr) * 100, 1))


def split(data, which, instr, or_bars, tail):
    tr = []
    for m in data.values():
        cut = int(len(m) * 0.70)
        seg = (m.iloc[:cut] if which == "train" else m.iloc[cut:]).reset_index(drop=True)
        tr += simulate(seg, instr, or_bars, tail)
    return metrics(tr, deploy=True)


def main():
    print(f"{'tf':>4}{'instr':>7}{'split':>7}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    for tf, cfg in TF_CFG.items():
        data = {}
        for under, etf in PAIRS.items():
            m = prep(under, etf, tf)
            if m is not None:
                data[etf] = m
        if not data:
            print(f"{tf:>4}  (no data)"); continue
        for instr in ("etf", "under"):
            for which in ("train", "test"):
                mm = split(data, which, instr, cfg["or_bars"], cfg["tail"])
                print(f"{tf:>4}{instr:>7}{which:>7}{mm['pf']:6.2f}{mm['sharpe']:8.2f}"
                      f"{mm['ret']:7.1f}{mm['win']:7.1f}{mm['n']:8d}")
        print()


if __name__ == "__main__":
    main()
