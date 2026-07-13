#!/usr/bin/env python
"""Test different MIN_DOLLAR_VOLUME thresholds against the current universe."""

import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

def main():
    try:
        from run_real_backtest import fetch_alpaca_bars
        from reversion_bot.universe import select_base_universe
        import pandas as pd
    except ImportError as e:
        print(f"Error: {e}")
        print("Required modules: pandas, alpaca-trade-api")
        sys.exit(1)

    # Get the universe
    print("Fetching universe...")
    try:
        symbols = select_base_universe()
        print(f"Found {len(symbols)} symbols in universe")
    except Exception as e:
        print(f"Error getting universe: {e}")
        print("Using fallback test symbols...")
        symbols = [
            "QBTS", "BSBR", "ITGR", "BITX", "ARX", "TPG", "CUBE", "MTUM",
            "RELX", "RCL", "CPAY", "NTST", "STLD", "LH", "ITW", "AAPL", "MSFT"
        ]

    # Fetch data
    print(f"\nFetching daily bars for {len(symbols)} symbols...")
    try:
        df_dict = fetch_alpaca_bars(symbols, "1Day", 20)
    except Exception as e:
        print(f"Error fetching bars: {e}")
        sys.exit(1)

    # Calculate dollar volumes
    volumes = {}
    for symbol, df in df_dict.items():
        if df is not None and len(df) > 0:
            try:
                # Use average over last 20 bars (matches config.volume_lookback)
                df['dollar_vol'] = df['close'].astype(float) * df['volume'].astype(float)
                avg_dollar_vol = df['dollar_vol'].tail(20).mean()
                latest_dollar_vol = df['dollar_vol'].iloc[-1]
                volumes[symbol] = {
                    'avg_20day': avg_dollar_vol,
                    'latest': latest_dollar_vol,
                    'close': float(df['close'].iloc[-1]),
                    'volume': int(df['volume'].iloc[-1]),
                }
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    if not volumes:
        print("No volume data collected")
        sys.exit(1)

    # Test thresholds
    thresholds = [
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        750_000,
    ]

    print("\n" + "="*80)
    print("DOLLAR VOLUME THRESHOLD ANALYSIS")
    print("="*80)
    print(f"\nTotal symbols: {len(volumes)}")
    print(f"Min avg daily dollar vol: ${min(v['avg_20day'] for v in volumes.values()):,.0f}")
    print(f"Max avg daily dollar vol: ${max(v['avg_20day'] for v in volumes.values()):,.0f}")
    print(f"Median avg daily dollar vol: ${sorted(v['avg_20day'] for v in volumes.values())[len(volumes)//2]:,.0f}")

    print("\n" + "-"*80)
    print("THRESHOLD ANALYSIS (20-day average dollar volume)")
    print("-"*80)
    print(f"{'Threshold':>15} | {'Pass':>4} | {'Fail':>4} | {'Pass %':>7} | {'Min Pass':>12} | {'Max Fail':>12}")
    print("-"*80)

    results = {}
    for threshold in thresholds:
        passing = [s for s, v in volumes.items() if v['avg_20day'] >= threshold]
        failing = [s for s, v in volumes.items() if v['avg_20day'] < threshold]
        pass_pct = 100 * len(passing) / len(volumes)

        min_pass_vol = min((v['avg_20day'] for s, v in volumes.items() if s in passing), default=0)
        max_fail_vol = max((v['avg_20day'] for s, v in volumes.items() if s in failing), default=0)

        results[threshold] = {
            'passing': passing,
            'failing': failing,
            'pass_pct': pass_pct,
            'min_pass': min_pass_vol,
            'max_fail': max_fail_vol,
        }

        print(f"${threshold:>13,} | {len(passing):>4} | {len(failing):>4} | {pass_pct:>6.1f}% | ${min_pass_vol:>11,.0f} | ${max_fail_vol:>11,.0f}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find thresholds that allow 10-30 candidates
    good_thresholds = [t for t, r in results.items() if 10 <= len(r['passing']) <= 30]

    if good_thresholds:
        print(f"\nFor 10-30 passing candidates (good coverage):")
        for t in good_thresholds:
            r = results[t]
            print(f"  MIN_DOLLAR_VOLUME=${t:,}: {len(r['passing'])} symbols pass")

    # Show what the current default filters
    current_default = 750_000
    r = results[current_default]
    print(f"\nCurrent default (${current_default:,}): {len(r['passing'])} symbols pass")
    if r['passing']:
        print(f"  Passing: {', '.join(sorted(r['passing'])[:10])}")
    if r['failing']:
        print(f"  Failing: {', '.join(sorted(r['failing'])[:10])}")

    print("\nSuggestion:")
    # Find the threshold that gives ~20 candidates
    best = min(good_thresholds, key=lambda t: abs(len(results[t]['passing']) - 20), default=None)
    if best:
        print(f"  Use MIN_DOLLAR_VOLUME=${best:,} for ~{len(results[best]['passing'])} active candidates")
    else:
        print(f"  Use MIN_DOLLAR_VOLUME=300000 (mid-cap friendly, typically 20-30 candidates)")

    print("\n" + "="*80)
    print("\nSample failing symbols (below $300k avg daily dollar volume):")
    failing_300k = sorted(results[300_000]['failing'], key=lambda s: volumes[s]['avg_20day'], reverse=True)[:10]
    for symbol in failing_300k:
        v = volumes[symbol]
        print(f"  {symbol:6} : ${v['avg_20day']:>12,.0f} avg  (latest: ${v['latest']:>12,.0f})")

if __name__ == "__main__":
    main()
