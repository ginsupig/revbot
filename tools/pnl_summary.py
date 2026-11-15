"""Pull realized PnL + fees from Alpaca for the last N days.

Reads APCA_API_KEY_ID / APCA_API_SECRET_KEY / APCA_API_BASE_URL from env (or .env if python-dotenv is installed).
Usage:  python tools/pnl_summary.py [days]   # default 7
"""
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alpaca_trade_api.rest import REST

KEY = os.environ.get("APCA_API_KEY_ID")
SECRET = os.environ.get("APCA_API_SECRET_KEY")
BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
if not KEY or not SECRET:
    sys.exit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in env (.env)")

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
client = REST(KEY, SECRET, BASE_URL)

cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).date()
print(f"Account: {BASE_URL}")
print(f"Window:  last {DAYS} days (since {cutoff})\n")

# --- Daily PnL from portfolio history ---
period = f"{DAYS}D"
ph = client.get_portfolio_history(period=period, timeframe="1D")
print("=== Daily PnL (from portfolio_history) ===")
total_pnl = 0.0
for ts, pl, pl_pct in zip(ph.timestamp, ph.profit_loss, ph.profit_loss_pct):
    d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    pl = float(pl or 0.0)
    pct = float(pl_pct or 0.0) * 100
    total_pnl += pl
    print(f"  {d}  PnL: ${pl:>10,.2f}   ({pct:+.2f}%)")
print(f"  {'TOTAL':<11}  ${total_pnl:>10,.2f}\n")

# --- Activities: fees + fills ---
fee_types = ("FEE", "REG", "TAF", "CSD", "CSW", "NTRF", "FINRA", "PTC", "OPC", "PSO", "USAF")
fees_by_day = defaultdict(float)
fee_total = 0.0
fills_by_symbol = defaultdict(lambda: {"buy_qty": 0.0, "buy_$": 0.0, "sell_qty": 0.0, "sell_$": 0.0, "n": 0})
fills_by_day = defaultdict(lambda: {"buys": 0, "sells": 0})

# Paginate activities; date param filters per-day, so loop over days
for i in range(DAYS + 1):
    d = (datetime.now(timezone.utc) - timedelta(days=i)).date()
    try:
        acts = client.get_activities(date=d.isoformat())
    except Exception as exc:
        print(f"  (skipped {d}: {exc})", file=sys.stderr)
        continue
    for a in acts:
        atype = getattr(a, "activity_type", "")
        if atype == "FILL":
            sym = a.symbol
            qty = float(a.qty)
            price = float(a.price)
            side = a.side  # "buy" or "sell"
            row = fills_by_symbol[sym]
            if side.startswith("buy"):
                row["buy_qty"] += qty
                row["buy_$"]   += qty * price
                fills_by_day[d]["buys"] += 1
            else:
                row["sell_qty"] += qty
                row["sell_$"]   += qty * price
                fills_by_day[d]["sells"] += 1
            row["n"] += 1
        elif atype in fee_types or "FEE" in atype.upper():
            amt = float(getattr(a, "net_amount", 0.0) or 0.0)
            fees_by_day[d] += amt
            fee_total += amt

print("=== Fees / regulatory charges ===")
if fees_by_day:
    for d, amt in sorted(fees_by_day.items()):
        print(f"  {d}  ${amt:>8,.4f}")
    print(f"  {'TOTAL':<11}  ${fee_total:>8,.4f}")
else:
    print("  (no fee activity in window)")
print()

print("=== Fills by day ===")
for d, c in sorted(fills_by_day.items()):
    print(f"  {d}  buys: {c['buys']:>3}  sells: {c['sells']:>3}")
print()

print("=== Per-symbol fills (net cash flow = sells$ - buys$) ===")
print(f"  {'sym':<6} {'fills':>6} {'buy_qty':>10} {'buy_$':>14} {'sell_qty':>10} {'sell_$':>14} {'net_$':>12}")
totals = {"n": 0, "buy_$": 0.0, "sell_$": 0.0, "net": 0.0}
for sym, r in sorted(fills_by_symbol.items(), key=lambda kv: -abs(kv[1]["sell_$"] - kv[1]["buy_$"])):
    net = r["sell_$"] - r["buy_$"]
    print(f"  {sym:<6} {r['n']:>6} {r['buy_qty']:>10,.0f} {r['buy_$']:>14,.2f} {r['sell_qty']:>10,.0f} {r['sell_$']:>14,.2f} {net:>12,.2f}")
    totals["n"] += r["n"]
    totals["buy_$"]  += r["buy_$"]
    totals["sell_$"] += r["sell_$"]
    totals["net"]    += net
print(f"  {'TOTAL':<6} {totals['n']:>6} {'':>10} {totals['buy_$']:>14,.2f} {'':>10} {totals['sell_$']:>14,.2f} {totals['net']:>12,.2f}")
print()
print("Note: portfolio_history PnL is authoritative. Per-symbol net cash flow approximates realized PnL")
print("only when positions opened in-window also closed in-window; open positions inflate buys (negative net).")
