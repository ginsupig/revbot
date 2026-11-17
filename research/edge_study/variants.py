import os, sys, pickle, copy
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import research.regime_gate_ab as ab
for k, v in ab.FLEET_ENV.items(): os.environ.setdefault(k, v)
START, END = "2024-01-01", "2026-11-14"
df = pd.read_pickle(Path(__file__).with_name("signals.pkl"))
replays = pickle.loads((ab.DATA / f"replay_{START}_{END}.pkl").read_bytes())
tr = df[df.traded & ~df.risk_off]
def stat(x, pc="pnl", rc="ret_bps"):
    p = x[pc]; w, l = p[p > 0].sum(), -p[p < 0].sum()
    yrs = " | ".join(f"{y}: n={len(z)} {z[rc].mean():+.1f}bps pnl {z[pc].sum():+.0f}" for y, z in x.groupby("year"))
    return f"n={len(x):4d} pf={w/l if l else np.nan:5.2f} mean={x[rc].mean():+6.1f}bps pnl={p.sum():+7.0f}  [{yrs}]"
print("-- filters on the existing trades (not re-simulated; cooldown interactions ignored)")
print("all                      ", stat(tr))
print("before 13:00 ET          ", stat(tr[tr.minute <= 780]))
print("13:00 ET and later       ", stat(tr[tr.minute > 780]))
print("no CLF                   ", stat(tr[tr.symbol != "CLF"]))
print("before 13:00, no CLF     ", stat(tr[(tr.minute <= 780) & (tr.symbol != "CLF")]))
gate = {(r.symbol, None): None for r in []}
off = {(s, t): o for s, t, o in zip(df.symbol, df.time, df.risk_off)}
print("-- re-simulated with wider stops (gate on, 8 bps), target unchanged")
for mult in (1.0, 2.0, 3.0):
    allt = []
    for s, sig in replays.items():
        b = ab._load(s); tloc = pd.DatetimeIndex(b["date"]).tz_convert(ab.NY)
        sig2 = {}
        for i, (plan, atr, close) in sig.items():
            if off.get((s, tloc[i]), False): continue
            p = dict(plan); p["risk_per_share"] = plan["risk_per_share"] * mult
            p["stop_price"] = plan["entry_price"] - p["risk_per_share"]; p["rr_ratio"] = plan["reward_per_share"] / p["risk_per_share"]
            sig2[i] = (p, atr, close)
        for t in ab.simulate_arm(s, sig2, lambda i: True, START, END, 8.0):
            t["year"] = t["time"].year; allt.append(t)
    x = pd.DataFrame(allt)
    print(f"stop x{mult:g}                  ", stat(x))
