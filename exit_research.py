"""Exit research on the live 5Min path (costs + EOD-flat, tuned entries).

Lever 1: sweep ATR stop/target multiples for the bracket exit (the 1.2 stop
         looked too tight) on a train split, validate on held-out test.
Lever 2: a mean-reclaim exit (protective ATR stop + exit when close >= SMA,
         optional time stop) vs the best bracket, on the same test split.

Same caveats as bracket_eod_backtest: 60d 5Min single-ish regime, intraday
OHLC can't order stop-vs-target within a bar (stop assumed first).
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
TUNED = dict(band_length=20, band_std_1=2.0, band_std_2=2.5, ri_threshold=-1.0,
             rsi_max=30.0, adx_max=25.0, min_history=2, min_price=0.0, min_dollar_volume=0.0)
ADX_HARD, RSI_HARD = 50.0, 70.0
ATR_FLOOR_PCT = 0.0035
ENTRY_COST, EXIT_COST = 8 / 10_000, 5 / 10_000
BLACKOUT, EOD_NOENTRY = 6, 2


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
    oversold = (e["ri"] <= cfg.ri_threshold) | (e["rsi"] <= cfg.rsi_max)
    not_ext = ~((e["adx"] >= ADX_HARD) & (e["rsi"] >= RSI_HARD))
    e["entry"] = (ready & (e["close"] <= e["lb1"]) & oversold & not_ext & (e["atr"] > 0)).to_numpy()
    day = pd.to_datetime(e["date"]).dt.date.to_numpy()
    is_last = np.append(day[1:] != day[:-1], True)
    from_end = np.zeros(len(e), int); pos = np.zeros(len(e), int)
    i = 0
    while i < len(e):
        j = i
        while j < len(e) and day[j] == day[i]:
            j += 1
        for k in range(i, j):
            pos[k] = k - i; from_end[k] = j - 1 - k
        i = j
    e["is_day_last"] = is_last
    e["can_enter"] = (pos >= BLACKOUT) & (from_end >= EOD_NOENTRY)
    return e


def simulate(e, stop_mult, target_mult, mode="bracket", max_hold=None):
    close = e["close"].to_numpy(float); high = e["high"].to_numpy(float)
    low = e["low"].to_numpy(float); atr = e["atr"].to_numpy(float); sma = e["sma"].to_numpy(float)
    entry = e["entry"].to_numpy(); can = e["can_enter"].to_numpy(); is_last = e["is_day_last"].to_numpy()
    n = len(close); ret = np.zeros(n)
    holding = False; stop = target = 0.0; cur = 0.0; held = 0; trades = []
    for i in range(1, n):
        if holding:
            base = close[i - 1]; exit_now = False; px = close[i]
            if low[i] <= stop:
                px = stop; exit_now = True
            elif mode == "bracket" and high[i] >= target:
                px = target; exit_now = True
            elif mode == "reclaim" and close[i] >= sma[i]:
                px = close[i]; exit_now = True
            elif max_hold is not None and held >= max_hold:
                px = close[i]; exit_now = True
            elif is_last[i]:
                px = close[i]; exit_now = True
            r = px / base - 1; ret[i] += r; cur += r; held += 1
            if exit_now:
                ret[i] -= EXIT_COST; cur -= EXIT_COST
                trades.append(cur); holding = False; cur = 0.0; held = 0
        if not holding and entry[i] and can[i] and not is_last[i]:
            au = max(atr[i], close[i] * ATR_FLOOR_PCT)
            stop = close[i] - au * stop_mult
            target = close[i] + au * target_mult
            holding = True; cur = -ENTRY_COST; ret[i] -= ENTRY_COST; held = 0
    return pd.Series(ret, index=range(n)), trades


def portfolio(data, which, stop, target, mode="bracket", max_hold=None):
    rets, ntr, wins = [], 0, []
    for e in data.values():
        cut = int(len(e) * 0.70)
        seg = (e.iloc[:cut] if which == "train" else e.iloc[cut:]).reset_index(drop=True)
        r, tr = simulate(seg, stop, target, mode, max_hold)
        rets.append(r.reset_index(drop=True)); ntr += len(tr); wins += [t > 0 for t in tr]
    port = pd.concat(rets, axis=1).mean(axis=1).fillna(0)
    pr = pd.DataFrame({"strategy_return": port.to_numpy(), "pnl": port.to_numpy()})
    eq = (1 + port).cumprod()
    return dict(pf=round(profit_factor(pr), 2), sharpe=round(sharpe_ratio(pr), 2),
                dd=round(max_drawdown(pr) * 100, 1), ret=round((eq.iloc[-1] - 1) * 100, 1),
                trades=ntr, win=round(np.mean(wins) * 100, 1) if wins else float("nan"))


def main():
    print("Loading + enriching 5Min data...")
    data = {}
    for t in UNIVERSE:
        df = load(t)
        if df is not None and len(df) > 400:
            data[t] = enrich(df)

    # ---- Lever 1: bracket stop/target sweep ----
    stops, targets = [1.5, 2.0, 2.5, 3.0], [2.0, 2.5, 3.0, 4.0]
    print(f"\n=== LEVER 1: bracket stop/target sweep ({len(stops)*len(targets)} combos, train=70%) ===")
    scored = []
    for s, tg in itertools.product(stops, targets):
        if tg / s < 1.0:
            continue
        m = portfolio(data, "train", s, tg, "bracket")
        scored.append((m["pf"], s, tg, m))
    scored.sort(reverse=True, key=lambda x: x[0])
    print(f"{'stop':>5}{'tgt':>5}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    for pf, s, tg, m in scored[:6]:
        print(f"{s:5.1f}{tg:5.1f}{m['pf']:6.2f}{m['sharpe']:8.2f}{m['ret']:7.1f}{m['win']:7.1f}{m['trades']:8d}")
    bs, bt = scored[0][1], scored[0][2]

    # ---- Lever 2: mean-reclaim exit (sweep protective stop + optional time stop) ----
    print(f"\n=== LEVER 2: mean-reclaim exit (train=70%) ===")
    print(f"{'stop':>5}{'hold':>6}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    rec = []
    for s in [2.0, 2.5, 3.0]:
        for mh in [None, 78]:   # None or 1 trading day (~78 5min bars)
            m = portfolio(data, "train", s, 0, "reclaim", mh)
            rec.append((m["pf"], s, mh, m))
            print(f"{s:5.1f}{str(mh):>6}{m['pf']:6.2f}{m['sharpe']:8.2f}{m['ret']:7.1f}{m['win']:7.1f}{m['trades']:8d}")
    rec.sort(reverse=True, key=lambda x: x[0])
    rs, rmh = rec[0][1], rec[0][2]

    # ---- Held-out TEST: baseline vs tuned-bracket vs reclaim ----
    print(f"\n=== HELD-OUT TEST (last 30%) ===")
    print(f"{'config':<28}{'pf':>6}{'sharpe':>8}{'ret%':>7}{'win%':>7}{'trades':>8}")
    configs = [
        ("baseline bracket 1.2/2.0", dict(stop=1.2, target=2.0, mode="bracket")),
        (f"tuned bracket {bs}/{bt}", dict(stop=bs, target=bt, mode="bracket")),
        (f"reclaim stop={rs} hold={rmh}", dict(stop=rs, target=0, mode="reclaim", max_hold=rmh)),
    ]
    for label, kw in configs:
        m = portfolio(data, "test", kw["stop"], kw["target"], kw["mode"], kw.get("max_hold"))
        print(f"{label:<28}{m['pf']:6.2f}{m['sharpe']:8.2f}{m['ret']:7.1f}{m['win']:7.1f}{m['trades']:8d}")


if __name__ == "__main__":
    main()
