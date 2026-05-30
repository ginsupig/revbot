"""Opening-Range Breakout (ORB) on leveraged ETFs, signals from the UNDERLYING.

- Opening range = first OR_BARS of the session, measured on the UNDERLYING
  equity/index. Breakout of OR-high => long the bull ETF; OR-low => short it.
- Trade/manage on the ETF: wider ATR stop -> defines R; at +1R move stop to
  breakeven, then trail TRAIL_PCT off the high-water mark.
- EOD-flat, one trade/day/symbol, costs, shorts enabled, 80% deployment.
- 5Min free data (yfinance, 60d). Train/test split. Same intraday-OHLC caveats.
"""
from __future__ import annotations
import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.indicators import calculate_atr
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

# underlying signal -> leveraged ETF traded
PAIRS = {
    "QQQ": "TQQQ", "SOXX": "SOXL", "XLK": "TECL", "NVDA": "NVDL", "TSLA": "TSLL",
    "AAPL": "AAPU", "META": "METU", "GOOGL": "GGLL", "MSFT": "MSFU", "AMZN": "AMZU",
}
OR_BARS = 3                 # 15-min opening range (3x5min)
EOD_NOENTRY = 2             # no new entries in last 10 min
TRAIL_PCT = 0.02
BE_AT_R = 1.0
ENTRY_COST, EXIT_COST = 8 / 10_000, 5 / 10_000
DEPLOY = 0.80              # 80% equity deployment (equal-weight across names)
# Time-based scale-outs (ET minutes-of-day; CT+1h). 2:00/2:30/2:55 PM CT.
SCALE1, SCALE2, FINAL = 15 * 60, 15 * 60 + 30, 15 * 60 + 55
SCALE1_FRAC, SCALE2_FRAC = 0.25, 0.25


def load(t):
    df = yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def prep(under, etf):
    u, e = load(under), load(etf)
    if u is None or e is None:
        return None
    m = pd.merge(u, e, on="date", suffixes=("_u", "_e")).reset_index(drop=True)
    if len(m) < 400:
        return None
    m["atr_e"] = calculate_atr(m.rename(columns={"high_e": "high", "low_e": "low", "close_e": "close"}), 14)
    m["day"] = m["date"].dt.date
    et = m["date"].dt.tz_convert("America/New_York")
    m["etmin"] = et.dt.hour * 60 + et.dt.minute
    return m.dropna().reset_index(drop=True)


def simulate(m, stop_mult, allow_short):
    uh, ul, uc = m["high_u"].to_numpy(), m["low_u"].to_numpy(), m["close_u"].to_numpy()
    eo = m["close_e"].to_numpy(); eh = m["high_e"].to_numpy(); el = m["low_e"].to_numpy()
    atr = m["atr_e"].to_numpy(); day = m["day"].to_numpy(); etm = m["etmin"].to_numpy()
    trades = []
    i = 0
    while i < len(m):
        j = i
        while j < len(m) and day[j] == day[i]:
            j += 1
        idx = list(range(i, j)); i = j
        if len(idx) < OR_BARS + 2:
            continue
        orh = uh[idx[:OR_BARS]].max(); orl = ul[idx[:OR_BARS]].min()
        for k in idx[OR_BARS: len(idx) - EOD_NOENTRY]:
            direction = 0
            if uc[k] > orh:
                direction = 1
            elif uc[k] < orl and allow_short:
                direction = -1
            if direction == 0:
                continue
            entry = eo[k]; a = atr[k]
            if a <= 0:
                break
            rem = 1.0; realized = 0.0; cost = ENTRY_COST; s1 = s2 = False
            long = direction == 1
            stop = entry - stop_mult * a if long else entry + stop_mult * a
            R = abs(entry - stop); peak = entry; be = False

            def take(frac, px):
                nonlocal realized, rem, cost
                pnl = (px / entry - 1) if long else (entry / px - 1)
                realized += frac * pnl; cost += frac * EXIT_COST; rem -= frac

            for n in range(k + 1, idx[-1] + 1):
                # update trail/breakeven on the runner
                if long:
                    peak = max(peak, eh[n])
                    if not be and eh[n] >= entry + R:
                        be = True; stop = max(stop, entry)
                    if be:
                        stop = max(stop, peak * (1 - TRAIL_PCT))
                    stop_hit = el[n] <= stop
                else:
                    peak = min(peak, el[n])
                    if not be and el[n] <= entry - R:
                        be = True; stop = min(stop, entry)
                    if be:
                        stop = min(stop, peak * (1 + TRAIL_PCT))
                    stop_hit = eh[n] >= stop
                if stop_hit:                       # flatten the remainder at stop
                    take(rem, stop); break
                # time-based scale-outs on the runner
                if not s1 and etm[n] >= SCALE1 and rem > 0:
                    take(min(SCALE1_FRAC, rem), eo[n]); s1 = True
                if not s2 and etm[n] >= SCALE2 and rem > 0:
                    take(min(SCALE2_FRAC, rem), eo[n]); s2 = True
                if (etm[n] >= FINAL or n == idx[-1]) and rem > 0:
                    take(rem, eo[n]); break
            trades.append(realized - cost)
            break   # one trade per day
    return trades


def metrics(trades):
    if not trades:
        return dict(n=0, win=float("nan"), ret=0.0, pf=float("nan"), sharpe=float("nan"), dd=0.0)
    a = np.array(trades)
    pr = pd.DataFrame({"strategy_return": a, "pnl": a})
    eq = (1 + a).cumprod()
    return dict(n=len(a), win=round((a > 0).mean() * 100, 1),
                ret=round((eq[-1] - 1) * 100, 1), pf=round(profit_factor(pr), 2),
                sharpe=round(sharpe_ratio(pr), 2), dd=round(max_drawdown(pr) * 100, 1))


def portfolio(data, which, stop_mult, allow_short):
    all_tr = []
    for m in data.values():
        cut = int(len(m) * 0.70)
        seg = (m.iloc[:cut] if which == "train" else m.iloc[cut:]).reset_index(drop=True)
        all_tr += simulate(seg, stop_mult, allow_short)
    mm = metrics(all_tr)
    mm["ret"] = round(mm["ret"] * DEPLOY, 1)   # scale headline return to 80% deployment
    return mm


def main():
    print("Loading underlying+ETF 5Min pairs...")
    data = {}
    for under, etf in PAIRS.items():
        m = prep(under, etf)
        if m is not None:
            data[etf] = m
    print(f"pairs ready: {list(data)}")

    print("\n=== Stop sweep (train=70%, shorts ON, BE@1R + 2% trail, 80% deploy) ===")
    print(f"{'stopATR':>8}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    scored = []
    for sm in [1.0, 1.5, 2.0, 3.0]:
        mm = portfolio(data, "train", sm, True)
        scored.append((mm["pf"], sm, mm))
        print(f"{sm:8.1f}{mm['pf']:6.2f}{mm['sharpe']:8.2f}{mm['ret']:7.1f}{mm['win']:7.1f}{mm['n']:8d}")
    scored.sort(reverse=True, key=lambda x: (x[0] if not np.isnan(x[0]) else -9))
    best = scored[0][1]

    print(f"\n=== HELD-OUT TEST (last 30%), best stop={best} ATR ===")
    print(f"{'config':<22}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    for label, short in [("longs only", False), ("longs+shorts", True)]:
        mm = portfolio(data, "test", best, short)
        print(f"{label:<22}{mm['pf']:6.2f}{mm['sharpe']:8.2f}{mm['ret']:7.1f}{mm['win']:7.1f}{mm['n']:8d}")

    print(f"\n=== Per-ticker (full 60d, stop={best} ATR, shorts ON) ===")
    print(f"{'etf':<6}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}{'maxDD%':>8}")
    for etf, m in data.items():
        mm = metrics(simulate(m, best, True))
        print(f"{etf:<6}{mm['pf']:6.2f}{mm['sharpe']:8.2f}{mm['ret']:7.1f}{mm['win']:7.1f}{mm['n']:8d}{mm['dd']:8.1f}")


if __name__ == "__main__":
    main()
