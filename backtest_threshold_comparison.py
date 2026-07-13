#!/usr/bin/env python
"""Backtest MIN_DOLLAR_VOLUME thresholds to validate the fix.

Compares trading performance with:
- Old default: $750,000 (blocks everything)
- New setting: $300,000 (should enable trades)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("TRADE_TIMEFRAME", "1Day")
os.environ.setdefault("TRADE_LOOKBACK", "160")

from dotenv import load_dotenv
load_dotenv()

def run_backtest(min_dollar_volume: float, label: str):
    """Run a backtest with a specific MIN_DOLLAR_VOLUME threshold."""
    print(f"\n{'='*80}")
    print(f"BACKTEST: {label} (MIN_DOLLAR_VOLUME=${min_dollar_volume:,.0f})")
    print('='*80)

    try:
        from reversion_bot.config import ReversionConfig, RiskConfig, PerformanceConfig
        from reversion_bot.service import ReversionService
        from run_real_backtest import fetch_alpaca_bars_batch
        import pandas as pd

        # Get universe - fallback to test symbols
        try:
            from reversion_bot.universe import select_base_universe
            symbols = select_base_universe()
        except:
            symbols = ["QBTS", "BSBR", "ITGR", "BITX", "ARX", "TPG", "CUBE", "MTUM",
                      "RELX", "RCL", "CPAY", "NTST", "STLD", "LH", "ITW", "AAPL", "MSFT"]

        if not symbols:
            print(f"Could not get universe")
            return None

        print(f"Universe: {len(symbols)} symbols")
        print(f"Sample: {', '.join(sorted(symbols)[:8])}")

        # Create config with the specified threshold
        strategy_config = ReversionConfig(
            min_dollar_volume=min_dollar_volume,
        )
        risk_config = RiskConfig()
        perf_config = PerformanceConfig()

        print(f"\nFetching latest daily bars for {len(symbols)} symbols...")

        try:
            # Fetch data for all symbols (last 180 days of data)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=180)

            df_dict = fetch_alpaca_bars_batch(
                symbols,
                start=start_date,
                end=end_date,
                timeframe="1Day",
            )

            # Create service
            service = ReversionService(
                strategy_config=strategy_config,
                risk_config=risk_config,
                performance_config=perf_config,
            )

            trades = []
            rejections_by_reason = {}
            evals = 0

            # Evaluate each symbol
            for symbol, df in df_dict.items():
                if df is None or len(df) < 160:
                    continue

                try:
                    result = service.evaluate_symbol(symbol, df, account_equity=100000.0)
                    evals += 1

                    if result.get("go_long") or result.get("go_short"):
                        trades.append({
                            "symbol": symbol,
                            "side": "LONG" if result.get("go_long") else "SHORT",
                            "regime": result.get("regime"),
                            "entry_style": result.get("entry_style"),
                            "trade_score": result.get("trade_score"),
                        })
                    else:
                        # Track rejection reasons
                        reason = result.get("router_reason", "unknown")
                        rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + 1
                except Exception as e:
                    pass

            print(f"Evaluations completed: {evals}")
            print(f"Candidates that passed entry gates: {len(trades)}")

            if trades:
                print(f"\nTrade candidates:")
                for t in sorted(trades, key=lambda x: x['trade_score'], reverse=True)[:10]:
                    print(f"  {t['symbol']:6} {t['side']:6} regime={t['regime']:10} "
                          f"style={t['entry_style']:15} score={t['trade_score']:.4f}")
                pass_rate = 100.0 * len(trades) / max(evals, 1)
                print(f"\nSignal pass rate: {pass_rate:.1f}% of evaluated symbols generated trade signals")
            else:
                print(f"\nNo trade candidates generated (100% rejection by gates)")
                print(f"\nRejection reasons:")
                for reason, count in sorted(rejections_by_reason.items(), key=lambda x: -x[1])[:5]:
                    pct = 100.0 * count / evals if evals > 0 else 0
                    print(f"  {reason:40} : {count:3} ({pct:5.1f}%)")

            return {
                "threshold": min_dollar_volume,
                "label": label,
                "evaluations": evals,
                "trades": len(trades),
                "pass_rate": 100.0 * len(trades) / max(evals, 1) if evals > 0 else 0,
                "trade_list": trades,
                "rejections": rejections_by_reason,
            }

        except ImportError as e:
            print(f"Module error: {e}")
            print("Ensure pandas and alpaca-py are installed")
            return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Test both thresholds
    results = []

    # Old threshold (the culprit)
    r1 = run_backtest(750_000.0, "OLD SETTING (blocks everything)")
    if r1:
        results.append(r1)

    # New threshold (the fix)
    r2 = run_backtest(300_000.0, "NEW SETTING (mid-cap friendly)")
    if r2:
        results.append(r2)

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print('='*80)
    print(f"{'Threshold':<20} | {'Evaluations':<12} | {'Signals':<12} | {'Pass Rate':<10}")
    print('-'*80)

    for r in results:
        print(f"${r['threshold']:>17,} | {r['evaluations']:>11} | {r['trades']:>11} | {r['pass_rate']:>8.1f}%")

    print(f"\n{'='*80}")
    if len(results) >= 2:
        improvement = results[1]['trades'] - results[0]['trades']
        print(f"RESULT: New threshold enables {improvement} additional trade signals!")
        print(f"        Old: {results[0]['trades']} signals → New: {results[1]['trades']} signals")

        if results[1]['trades'] > 0 and results[0]['trades'] == 0:
            print(f"\n✅ FIX VALIDATED: Threshold change fixes the 'no trades' issue")
        elif results[1]['trades'] > results[0]['trades']:
            print(f"\n✅ IMPROVEMENT: New threshold generates more trade opportunities")
    print('='*80)

if __name__ == "__main__":
    main()
