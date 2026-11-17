import os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import research.regime_gate_ab as ab
pd.set_option("display.width", 220)
df = pd.read_pickle(Path(__file__).with_name("signals.pkl"))
# baseline: every bar in the entry window, same forward-return definition
base = []
for s in df.symbol.unique():
    b = ab._load(s); c = b["close"].astype(float).values; o = b["open"].astype(float).values
    t = pd.DatetimeIndex(b["date"]).tz_convert(ab.NY); day = t.normalize()
    li = pd.Series(np.arange(len(b))).groupby(day).transform("last").values
    m = t.hour * 60 + t.minute
    ok = np.where((m >= 595) & (m < 935) & (t >= "2024-01-01") & (t < "2026-11-14"))[0]
    ok = ok[ok + 1 < len(b)]; e = ok + 1; l = li[e]
    base.append(pd.DataFrame(dict(symbol=s, year=t[ok].year, f30=(c[np.minimum(e+6,l)]/o[e]-1)*1e4, f60=(c[np.minimum(e+12,l)]/o[e]-1)*1e4,
                                  f120=(c[np.minimum(e+24,l)]/o[e]-1)*1e4, fclose=(c[l]/o[e]-1)*1e4)))
base = pd.concat(base)
# de-cluster signals: first signal per symbol per 60 minutes
df = df.sort_values(["symbol", "time"]); gap_ok = df.groupby("symbol").time.diff().dt.total_seconds().fillna(1e9) >= 3600
first = df[gap_ok]
H = ["f30", "f60", "f120", "fclose"]
def se(x): return x.std() / np.sqrt(len(x))
print("=== gross forward return from the next bar's open, bps (no costs) ===")
print("all signal bars     n=%d  " % len(df) + "  ".join(f"{h} {df[h].mean():+.1f}±{se(df[h]):.1f}" for h in H))
print("de-clustered        n=%d  " % len(first) + "  ".join(f"{h} {first[h].mean():+.1f}±{se(first[h]):.1f}" for h in H))
g = first[~first.risk_off]; print("  gate lets through n=%d  " % len(g) + "  ".join(f"{h} {g[h].mean():+.1f}±{se(g[h]):.1f}" for h in H))
g = first[first.risk_off]; print("  gate blocks       n=%d  " % len(g) + "  ".join(f"{h} {g[h].mean():+.1f}±{se(g[h]):.1f}" for h in H))
print("random entry        n=%d  " % len(base) + "  ".join(f"{h} {base[h].mean():+.1f}" for h in H))
print("\nby year (de-clustered, gate on): signal vs random, fclose")
for y in sorted(first.year.unique()):
    a = first[(first.year == y) & ~first.risk_off]; bb = base[base.year == y]
    print(f"  {y}: signal n={len(a)} f60 {a.f60.mean():+.1f}±{se(a.f60):.1f} fclose {a.fclose.mean():+.1f}±{se(a.fclose):.1f} | random f60 {bb.f60.mean():+.1f} fclose {bb.fclose.mean():+.1f}")
print("\nby symbol (de-clustered, gate on): fclose signal vs random")
for s in first.symbol.unique():
    a = first[(first.symbol == s) & ~first.risk_off]; bb = base[base.symbol == s]
    print(f"  {s:5s} n={len(a):3d} f60 {a.f60.mean():+6.1f} fclose {a.fclose.mean():+6.1f}±{se(a.fclose):.1f} | random fclose {bb.fclose.mean():+.1f}")

tr = df[df.traded & ~df.risk_off].copy()
def stat(x):
    p = x.pnl; w, l = p[p > 0].sum(), -p[p < 0].sum()
    return f"n={len(x):4d} pf={w/l if l else np.nan:5.2f} mean={x.ret_bps.mean():+6.1f}bps win={(p>0).mean():.0%} pnl={p.sum():+7.0f}"
print("\n=== simulated trades, gate on, 8 bps cost ===\nALL   ", stat(tr))
def cut(col, bins, labels=None):
    print(f"\n-- by {col}")
    k = pd.cut(tr[col], bins) if not isinstance(bins, str) else tr[col]
    for name, x in tr.groupby(k, observed=True):
        yrs = " | ".join(f"{y}: n={len(z)} {z.ret_bps.mean():+.0f}" for y, z in x.groupby("year"))
        print(f"  {str(name):18s} {stat(x)}   [{yrs}]")
cut("symbol", "x"); cut("year", "x")
cut("minute", [599, 660, 720, 780, 840, 945]); cut("rsi", [0, 15, 20, 25, 30, 40, 100]); cut("z", [-10, -3, -2.5, -2, -1.5, 10])
cut("dayret", [-50, -5, -3, -1.5, 0, 50]); cut("gap", [-50, -2, -0.5, 0.5, 2, 50]); cut("spy_dayret", [-10, -1, -0.3, 0.3, 1, 10])
cut("rvol", [0, 0.7, 1.2, 2, 4, 1e9]); cut("stop_pct", [0, 0.3, 0.5, 0.8, 1.2, 100]); cut("rr", [0, 1.5, 2.5, 4, 100]); cut("atr_pct", [0, 0.2, 0.35, 0.5, 100])
tr["held"] = tr.bars_held; cut("held", [-1, 1, 3, 6, 12, 24, 1000])
print("\ngross of costs: mean %.1f bps; cost 8 → net %.1f" % (tr.ret_bps.mean() + 8, tr.ret_bps.mean()))
