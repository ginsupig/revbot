"""A/B experiment: does cutting churn raise *net* edge?

Runs the realistic backtester on one symbol under several cost/churn settings —
a re-entry cooldown (live parity: 30 min == 6 bars on 5-min data) and a
reward-to-cost entry gate — and prints trades, profit factor, total return, and
average PnL per trade side by side. The point is to see whether trading *less*
keeps more money, which is the only edge lever in costs.

Strategy params come from your current .env (the tuned values), so this tests
the churn controls on top of what the bot actually trades.

Usage:
    python churn_backtest.py TQQQ                  # 60 days, default scenarios
    python churn_backtest.py SOXL --days 90
    python churn_backtest.py QQQ --cooldowns 0 6 12 --edge-gates 0 3

Reads Alpaca creds from the environment / .env.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

from reversion_bot.walkforward import mean_reversion_strategy
from run_real_backtest import fetch_alpaca_bars


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def tuned_params_from_env() -> dict:
    """Read the strategy params the bot currently uses from .env."""
    return dict(
        band_length=int(os.getenv("BAND_LENGTH", 20)),
        ri_threshold=float(os.getenv("RI_THRESHOLD", -0.5)),
        rsi_max=float(os.getenv("RSI_MAX", 48.0)),
        use_vwap_filter=_parse_bool(os.getenv("USE_VWAP_FILTER", "False")),
        max_vwap_extension_pct=float(os.getenv("MAX_VWAP_EXTENSION_PCT", 0.012)),
        min_history=int(os.getenv("TRADE_LOOKBACK", 160)),
    )


def summarize(res) -> dict:
    """Trade count, profit factor, total return and avg PnL/trade from a result df."""
    trades = int(res["signal"].sum())
    # Per-trade returns live on the bars where a position closes (return != 0 and
    # the trade ended). Use the realized round-trip returns: bars whose return is
    # the bracket exit. Approximate trade PnL by summing strategy_return over the
    # run and dividing by trades; report compounded total too.
    rets = res["strategy_return"].to_numpy()
    wins = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    total_return = float(np.prod(1.0 + rets) - 1.0)
    avg_per_trade = (float(rets.sum()) / trades) if trades else 0.0
    return {
        "trades": trades,
        "profit_factor": pf,
        "total_return": total_return,
        "avg_return_per_trade": avg_per_trade,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Churn A/B backtest for one symbol")
    parser.add_argument("symbol", help="symbol to test (e.g. TQQQ)")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days (default 60)")
    parser.add_argument("--timeframe", default="5Min", help="bar timeframe (default 5Min)")
    parser.add_argument("--cooldowns", type=int, nargs="+", default=[0, 6, 12],
                        help="re-entry cooldown bars to test (default 0 6 12)")
    parser.add_argument("--edge-gates", type=float, nargs="+", default=[0, 3],
                        help="reward-to-cost gate multiples to test (default 0 3)")
    args = parser.parse_args()

    load_dotenv()
    base = tuned_params_from_env()
    symbol = args.symbol.strip().upper()

    end_dt = datetime.now()
    start = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    print(f"Churn A/B | {symbol} | {args.days}d ({start} -> {end}) | tf={args.timeframe}")
    print(f"Tuned params: {base}")
    bars = fetch_alpaca_bars(symbol, start, end, args.timeframe)
    if bars is None or len(bars) < base["min_history"]:
        print(f"Insufficient data ({0 if bars is None else len(bars)} bars).")
        return
    print(f"Fetched {len(bars)} bars.\n")

    print(f"  {'COOLDOWN':>9}{'EDGEGATE':>10}{'TRADES':>8}{'PF':>8}"
          f"{'TOTAL RET':>12}{'RET/TRADE bps':>15}")
    print(f"  {'-'*8:>9}{'-'*8:>10}{'-'*7:>8}{'-'*6:>8}{'-'*10:>12}{'-'*13:>15}")
    for cd in args.cooldowns:
        for gate in args.edge_gates:
            res = mean_reversion_strategy(
                bars, reentry_cooldown_bars=cd, min_reward_cost_ratio=gate, **base
            )
            s = summarize(res)
            pf_str = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
            print(f"  {cd:>9}{gate:>10.0f}{s['trades']:>8}{pf_str:>8}"
                  f"{s['total_return']*100:>11.2f}%{s['avg_return_per_trade']*1e4:>15.1f}")


if __name__ == "__main__":
    main()
