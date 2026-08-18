import os
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass
from alpaca_trade_api.rest import REST

c = REST(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"],
         os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"))

positions = c.list_positions()
print(f"{'sym':<6} {'qty':>8} {'avg_cost':>10} {'last':>10} {'cost_basis':>13} {'market_val':>13} {'unrealized':>13} {'%':>7}")
tot_cost = tot_mv = tot_unr = 0.0
for p in sorted(positions, key=lambda x: -abs(float(x.unrealized_pl))):
    q = float(p.qty); avg = float(p.avg_entry_price); last = float(p.current_price)
    cb = float(p.cost_basis); mv = float(p.market_value); unr = float(p.unrealized_pl)
    pct = float(p.unrealized_plpc) * 100
    tot_cost += cb; tot_mv += mv; tot_unr += unr
    print(f"{p.symbol:<6} {q:>8,.0f} {avg:>10,.2f} {last:>10,.2f} {cb:>13,.2f} {mv:>13,.2f} {unr:>13,.2f} {pct:>6.2f}%")
print(f"{'TOTAL':<6} {'':>8} {'':>10} {'':>10} {tot_cost:>13,.2f} {tot_mv:>13,.2f} {tot_unr:>13,.2f}")
