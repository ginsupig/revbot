import os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass
from alpaca_trade_api.rest import REST

KEY = os.environ["APCA_API_KEY_ID"]; SECRET = os.environ["APCA_API_SECRET_KEY"]
BASE = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 21
client = REST(KEY, SECRET, BASE)

counts = Counter()
totals = defaultdict(float)
examples = {}
for i in range(DAYS + 1):
    d = (datetime.now(timezone.utc) - timedelta(days=i)).date()
    try:
        acts = client.get_activities(date=d.isoformat())
    except Exception as e:
        print(f"skip {d}: {e}", file=sys.stderr); continue
    for a in acts:
        t = getattr(a, "activity_type", "?")
        counts[t] += 1
        amt = float(getattr(a, "net_amount", 0) or 0)
        totals[t] += amt
        if t not in examples:
            examples[t] = {k: getattr(a, k, None) for k in ("activity_type","symbol","qty","price","side","net_amount","description")}

print(f"{'type':<10} {'count':>6} {'net_amount_total':>20}")
for t, n in counts.most_common():
    print(f"{t:<10} {n:>6} {totals[t]:>20,.4f}")
print("\nExamples:")
for t, ex in examples.items():
    print(" ", t, ex)
