import json
from datetime import datetime, timezone, timedelta
from collections import Counter

cutoff = datetime.now(timezone.utc) - timedelta(days=7)
rows = []
bad = 0
for l in open("state/performance/trades.jsonl", encoding="utf-8"):
    l = l.strip()
    if not l: continue
    try:
        r = json.loads(l)
        if datetime.fromisoformat(r["timestamp"]) >= cutoff:
            rows.append(r)
    except Exception:
        bad += 1

by_symbol = Counter(r["symbol"] for r in rows)
by_style = Counter(r["entry_style"] for r in rows)
by_regime = Counter(r["regime"] for r in rows)
by_day = Counter(r["timestamp"][:10] for r in rows)
by_bucket = Counter(r["time_bucket"] for r in rows)
total_notional = sum(float(r["position_value"]) for r in rows)
avg_score = sum(float(r["trade_score"]) for r in rows) / max(len(rows), 1)
avg_rr = sum(float(r["rr_ratio"]) for r in rows) / max(len(rows), 1)

print("Trades (last 7d):", len(rows), "(skipped", bad, "malformed)")
print("Total notional:   $%s" % format(total_notional, ",.0f"))
print("Avg trade_score:  %.3f" % avg_score)
print("Avg rr_ratio:     %.3f" % avg_rr)
print()
print("By day:")
for d, n in sorted(by_day.items()): print(" ", d, n)
print("By symbol:")
for s, n in by_symbol.most_common(): print(" ", s, n)
print("By entry_style:")
for s, n in by_style.most_common(): print(" ", s, n)
print("By regime:")
for s, n in by_regime.most_common(): print(" ", s, n)
print("By time_bucket:")
for s, n in by_bucket.most_common(): print(" ", s, n)
