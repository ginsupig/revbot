"""Bracket + EOD-flat 5Min backtest — closes the exit-logic gap.

Reproduces the LIVE mechanics on the LIVE timeframe:
  - 5Min bars (free yfinance, last 60d)
  - tuned entry params (the 60d autotune medians)
  - ATR stop/target BRACKET exits (RiskConfig multiples), intrabar via high/low
  - EOD flat: any open position closed at the last bar of the session
  - morning blackout (first 30min) + no entries in last 10min (mirrors main.py)
  - transaction costs: marketable-limit entry offset + market-exit slippage

This is the test that the daily proxy and the mean-reclaim autotune could not do.
Still has caveats: 60d single-ish regime, 5Min OHLC can't resolve stop-vs-target
order within a bar (stop assumed first = conservative).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

UNIVERSE = ["TQQQ", "SOXL", "TECL", "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU"]

# Tuned entry params (60d autotune medians).
TUNED = dict(band_length=20, band_std_1=2.0, band_std_2=2.5,
             ri_threshold=-1.0, rsi_max=30.0, adx_max=25.0, min_history=2,
             min_price=0.0, min_dollar_volume=0.0)
ADX_HARD, RSI_HARD = 50.0, 70.0

# Live risk mechanics.
ATR_FLOOR_PCT = 0.0035
STOP_MULT, TARGET_MULT = 1.20, 2.00          # RiskConfig defaults
ENTRY_COST = 8 / 10_000                       # marketable limit offset (LIMIT_ENTRY_OFFSET_BPS)
EXIT_COST = 5 / 10_000                        # market-exit slippage assumption
BARS_PER_DAY_BLACKOUT = 6                      # first 30min (6x5min) no entries
BARS_EOD_NOENTRY = 2                           # last 10min no new entries


def load(t):
    df = yf.download(t, period="60d", interval="5m", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


def enrich(df):
    cfg = ReversionConfig(**TUNED)
    e = ReversionEngine(cfg).calculate_indicators(df)
    ready = e[["sma", "lb1", "lb2", "ri", "rsi", "adx", "atr", "vwap"]].notna().all(axis=1)
    in_zone = e["close"] <= e["lb1"]
    oversold = (e["ri"] <= cfg.ri_threshold) | (e["rsi"] <= cfg.rsi_max)
    not_ext = ~((e["adx"] >= ADX_HARD) & (e["rsi"] >= RSI_HARD))
    e["entry"] = (ready & in_zone & oversold & not_ext & (e["atr"] > 0)).to_numpy()
    # Day-structure flags.
    day = pd.to_datetime(e["date"]).dt.date.to_numpy()
    is_last = np.append(day[1:] != day[:-1], True)
    pos_in_day = np.zeros(len(e), int); from_end = np.zeros(len(e), int)
    i = 0
    while i < len(e):
        j = i
        while j < len(e) and day[j] == day[i]:
            j += 1
        for k in range(i, j):
            pos_in_day[k] = k - i
            from_end[k] = j - 1 - k
        i = j
    e["is_day_last"] = is_last
    e["can_enter"] = (pos_in_day >= BARS_PER_DAY_BLACKOUT) & (from_end >= BARS_EOD_NOENTRY)
    return e


def simulate(e):
    close = e["close"].to_numpy(float); high = e["high"].to_numpy(float)
    low = e["low"].to_numpy(float); atr = e["atr"].to_numpy(float)
    entry = e["entry"].to_numpy(); can_enter = e["can_enter"].to_numpy()
    is_last = e["is_day_last"].to_numpy()
    n = len(close); ret = np.zeros(n)
    holding = False; stop = target = 0.0; cur = 0.0; trades = []

    def book(r):
        nonlocal cur
        ret[i] += r; cur += r

    for i in range(1, n):
        if holding:
            base = close[i - 1]
            if low[i] <= stop:                              # stop first (conservative)
                book(stop / base - 1); ret[i] -= EXIT_COST; cur -= EXIT_COST
                trades.append(cur); holding = False; cur = 0.0
            elif high[i] >= target:
                book(target / base - 1); ret[i] -= EXIT_COST; cur -= EXIT_COST
                trades.append(cur); holding = False; cur = 0.0
            elif is_last[i]:                                # EOD flat at close
                book(close[i] / base - 1); ret[i] -= EXIT_COST; cur -= EXIT_COST
                trades.append(cur); holding = False; cur = 0.0
            else:
                book(close[i] / base - 1)
        if not holding and entry[i] and can_enter[i] and not is_last[i]:
            au = max(atr[i], close[i] * ATR_FLOOR_PCT)
            stop = close[i] - au * STOP_MULT
            target = close[i] + au * TARGET_MULT
            holding = True; cur = -ENTRY_COST; ret[i] -= ENTRY_COST
    return pd.Series(ret, index=e["date"]), trades


def main():
    rows, port_rets = [], []
    for t in UNIVERSE:
        df = load(t)
        if df is None or len(df) < 200:
            rows.append(dict(ticker=t, bars=0 if df is None else len(df), note="insufficient")); continue
        e = enrich(df)
        r, trades = simulate(e)
        port_rets.append(r)
        eq = (1 + r).cumprod()
        wins = [x > 0 for x in trades]
        pr = pd.DataFrame({"strategy_return": r.to_numpy(), "pnl": r.to_numpy()})
        rows.append(dict(
            ticker=t, bars=len(e), trades=len(trades),
            win_rate=round(np.mean(wins) * 100, 1) if trades else float("nan"),
            ret=round((eq.iloc[-1] - 1) * 100, 1),
            pf=round(profit_factor(pr), 2), sharpe=round(sharpe_ratio(pr), 2),
            max_dd=round(max_drawdown(pr) * 100, 1),
        ))
    tbl = pd.DataFrame(rows)

    if port_rets:
        port = pd.concat(port_rets, axis=1).mean(axis=1).fillna(0)
        eq = (1 + port).cumprod()
        pr = pd.DataFrame({"strategy_return": port.to_numpy(), "pnl": port.to_numpy()})
        tbl = pd.concat([tbl, pd.DataFrame([dict(
            ticker="PORTFOLIO", bars=len(port), trades=int(tbl["trades"].sum()),
            win_rate=np.nan, ret=round((eq.iloc[-1] - 1) * 100, 1),
            pf=round(profit_factor(pr), 2), sharpe=round(sharpe_ratio(pr), 2),
            max_dd=round(max_drawdown(pr) * 100, 1))])], ignore_index=True)

    pd.set_option("display.width", 140)
    print("\n=== Bracket + EOD-flat 5Min backtest (tuned params, costs incl.) ===")
    print(f"entry_cost={ENTRY_COST*1e4:.0f}bps exit_cost={EXIT_COST*1e4:.0f}bps  stop/target={STOP_MULT}/{TARGET_MULT}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
