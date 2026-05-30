"""Bracket-aware backtest: simulates the live ATR stop/target exits.

Unlike backtest_leveraged.py (which uses the simplified mean-reclaim exit),
this reproduces the bot's risk mechanics: enter on LONG_REVERSION, then exit
at entry +/- ATR*multiple (target/stop), intrabar via high/low. Sweeps the
stop/target multiples on the first 70% of history and validates on the last
30%. Daily free-data proxy; the live engine is 5Min intraday with EOD flat,
so treat as directional. No costs modeled.
"""
from __future__ import annotations
import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

UNIVERSE = ["TQQQ", "SOXL", "TECL", "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU"]
ATR_FLOOR_PCT = 0.0035            # RiskConfig.atr_floor_pct
ADX_HARD, RSI_HARD = 50.0, 70.0   # ReversionConfig hard caps

# Sweep grid for (stop_mult, target_mult). Default is (1.20, 2.00).
STOPS = [1.0, 1.2, 1.5]
TARGETS = [1.5, 2.0, 2.5, 3.0]


def load(t):
    df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    return df[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


def enrich(df):
    """Compute indicators + vectorized LONG_REVERSION entry mask (default filters off)."""
    cfg = ReversionConfig(min_history=2, min_price=0.0, min_dollar_volume=0.0)
    eng = ReversionEngine(cfg)
    e = eng.calculate_indicators(df)
    ready = e[["sma", "lb1", "lb2", "ri", "rsi", "adx", "atr", "vwap"]].notna().all(axis=1)
    in_zone = e["close"] <= e["lb1"]
    oversold = (e["ri"] <= cfg.ri_threshold) | (e["rsi"] <= cfg.rsi_max)
    not_extended = ~((e["adx"] >= ADX_HARD) & (e["rsi"] >= RSI_HARD))
    e["entry"] = (ready & in_zone & oversold & not_extended & (e["atr"] > 0)).to_numpy()
    return e


def simulate(e, stop_mult, target_mult):
    """Return per-bar strategy return series respecting ATR bracket exits."""
    close = e["close"].to_numpy(float); high = e["high"].to_numpy(float)
    low = e["low"].to_numpy(float); atr = e["atr"].to_numpy(float)
    entry = e["entry"]; n = len(close)
    ret = np.zeros(n)
    holding = False; stop = target = 0.0; trades = []; cur = 0.0
    for i in range(1, n):
        if holding:
            base = close[i - 1]
            if low[i] <= stop:                       # stop priority (conservative)
                r = stop / base - 1; ret[i] = r; cur += r
                trades.append(cur); holding = False; cur = 0.0
            elif high[i] >= target:
                r = target / base - 1; ret[i] = r; cur += r
                trades.append(cur); holding = False; cur = 0.0
            else:
                r = close[i] / base - 1; ret[i] = r; cur += r
        if not holding and entry[i]:                 # enter at this bar's close
            atr_used = max(atr[i], close[i] * ATR_FLOOR_PCT)
            stop = close[i] - atr_used * stop_mult
            target = close[i] + atr_used * target_mult
            holding = True; cur = 0.0
    return pd.Series(ret, index=e["date"]), trades


def portfolio(data_e, stop_mult, target_mult, which):
    rets, ntrades, wins = [], 0, []
    for e in data_e.values():
        cut = int(len(e) * 0.70)
        seg = (e.iloc[:cut] if which == "train" else e.iloc[cut:]).reset_index(drop=True)
        r, trades = simulate(seg, stop_mult, target_mult)
        rets.append(r); ntrades += len(trades); wins += trades
    port = pd.concat(rets, axis=1).mean(axis=1).fillna(0)
    pr = pd.DataFrame({"strategy_return": port.to_numpy(), "pnl": port.to_numpy()})
    eq = (1 + port).cumprod()
    win = (np.mean([t > 0 for t in wins]) * 100) if wins else float("nan")
    return dict(pf=profit_factor(pr), sharpe=sharpe_ratio(pr),
                dd=max_drawdown(pr) * 100, ret=(eq.iloc[-1] - 1) * 100,
                trades=ntrades, win=win)


def main():
    data_e = {}
    for t in UNIVERSE:
        df = load(t)
        if df is not None and len(df) > 120:
            data_e[t] = enrich(df)
    print(f"Bracket sweep over {len(data_e)} tickers, {len(STOPS)*len(TARGETS)} combos (train=first 70%)\n")

    scored = []
    for s, tg in itertools.product(STOPS, TARGETS):
        if tg / s < 1.0:
            continue
        m = portfolio(data_e, s, tg, "train")
        scored.append((m["sharpe"], s, tg, m))
    scored.sort(reverse=True, key=lambda x: x[0])

    print("=== Top 6 by TRAIN Sharpe ===")
    print(f"{'stop':>5}{'tgt':>5}{'rr':>5}{'sharpe':>8}{'pf':>6}{'maxDD':>8}{'ret%':>8}{'win%':>7}{'trades':>8}")
    for sh, s, tg, m in scored[:6]:
        print(f"{s:5.1f}{tg:5.1f}{tg/s:5.2f}{m['sharpe']:8.2f}{m['pf']:6.2f}{m['dd']:8.1f}{m['ret']:8.1f}{m['win']:7.1f}{m['trades']:8d}")

    _, bs, bt, _ = scored[0]
    print("\n=== Held-out TEST (last 30%): default(1.2/2.0) vs tuned ===")
    print(f"{'config':<18}{'sharpe':>8}{'pf':>6}{'maxDD':>8}{'ret%':>8}{'win%':>7}{'trades':>8}")
    for label, (s, tg) in [("default 1.2/2.0", (1.2, 2.0)), (f"tuned {bs}/{bt}", (bs, bt))]:
        m = portfolio(data_e, s, tg, "test")
        print(f"{label:<18}{m['sharpe']:8.2f}{m['pf']:6.2f}{m['dd']:8.1f}{m['ret']:8.1f}{m['win']:7.1f}{m['trades']:8d}")


if __name__ == "__main__":
    main()
