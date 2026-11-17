"""Where, if anywhere, does revbot's long entry have an edge? Uses the saved live-service replay."""
import os, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import research.regime_gate_ab as ab
for k, v in ab.FLEET_ENV.items(): os.environ.setdefault(k, v)
START, END, COST = "2024-01-01", "2026-11-14", 8.0
replays = pickle.loads((ab.DATA / f"replay_{START}_{END}.pkl").read_bytes())
spy_d = pd.read_pickle(ab.DATA / "SPY_1Day.pkl")
spy_d["day"] = pd.to_datetime(spy_d["date"], utc=True).dt.tz_convert(ab.NY).dt.normalize().dt.tz_localize(None)
spy_5 = ab._load("SPY"); spy_5 = spy_5.set_index("date")
rows = []
for s, sig in replays.items():
    b = ab._load(s); c = b["close"].astype(float); t = pd.DatetimeIndex(b["date"]).tz_convert(ab.NY)
    day = t.normalize(); o = b["open"].astype(float)
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    sma = c.rolling(20).mean(); sd = c.rolling(20).std()
    z = (c - sma) / sd
    dayopen = o.groupby(day).transform("first"); dayret = c / dayopen - 1
    last_i = pd.Series(np.arange(len(b))).groupby(day).transform("last").values
    first_i = pd.Series(np.arange(len(b))).groupby(day).transform("first").values
    prev_close = c.groupby(day).last().shift(1); gap = (dayopen.values / prev_close.reindex(day).values) - 1
    vol = b["volume"].astype(float); rvol = vol / vol.rolling(78).mean()
    trades = ab.simulate_arm(s, sig, lambda i: True, START, END, COST)
    tr_by = {x["signal_bar"]: x for x in trades}
    for i, (plan, atr, close) in sig.items():
        ts = b["date"].iloc[i]
        off = bool(ab.risk_off_at(spy_d, spy_5.reset_index(), ts))
        e = i + 1                                    # order goes in on the next bar
        if e >= len(b): continue
        ep = float(o.iloc[e]); li = int(last_i[e])
        def fwd(n):
            j = min(e + n, li); return (float(c.iloc[j]) / ep - 1) * 1e4
        nxt = li + 1
        spy_now = spy_5["close"].asof(ts); spy_open = spy_5["open"][spy_5.index.tz_convert(ab.NY).normalize() == day[i]]
        tr = tr_by.get(i)
        rows.append(dict(symbol=s, time=t[i], year=t[i].year, minute=t[i].hour * 60 + t[i].minute, risk_off=off,
            rsi=float(rsi.iloc[i]), z=float(z.iloc[i]), dayret=float(dayret.iloc[i]) * 100, gap=float(gap[i]) * 100,
            rvol=float(rvol.iloc[i]), stop_pct=plan["risk_per_share"] / plan["entry_price"] * 100, rr=plan["rr_ratio"],
            atr_pct=atr / close * 100 if close else np.nan,
            spy_dayret=(float(spy_now) / float(spy_open.iloc[0]) - 1) * 100 if len(spy_open) else np.nan,
            f30=fwd(6), f60=fwd(12), f120=fwd(24), fclose=(float(c.iloc[li]) / ep - 1) * 1e4,
            fnext_open=(float(o.iloc[nxt]) / ep - 1) * 1e4 if nxt < len(b) else np.nan,
            traded=tr is not None, ret_bps=tr["ret_bps"] if tr else np.nan, pnl=tr["pnl"] if tr else np.nan,
            exit=tr.get("exit_reason", tr.get("reason")) if tr else None, bars_held=(tr.get("exit_bar", np.nan) - tr.get("entry_bar", np.nan)) if tr else np.nan))
    print(s, len(sig), "signals", len(trades), "trades; trade keys:", sorted(trades[0]) if trades else None, flush=True)
df = pd.DataFrame(rows); df.to_pickle(Path(__file__).with_name("signals.pkl")); print(len(df))
