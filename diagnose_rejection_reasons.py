#!/usr/bin/env python
"""Diagnose why the bot is rejecting all candidates by checking the actual logs."""

import os
import re
from pathlib import Path
from collections import defaultdict

def parse_logs(log_file: str = "reversion_service.log"):
    """Parse reversion_service.log and extract rejection reasons."""

    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print("   Run the bot first to generate logs")
        return {}

    print(f"📋 Analyzing {log_file}...")

    rejections = defaultdict(int)
    decisions = defaultdict(int)
    evals = 0
    trades = 0

    # Pattern to match evaluation log lines
    pattern = r"symbol=(\w+).*decision=(\w+).*reason=(\w+).*router_reason=([^\s]+)"

    with open(log_file, 'r') as f:
        for line in f:
            if "Evaluating symbol=" in line:
                evals += 1

            match = re.search(pattern, line)
            if match:
                symbol, decision, reason, router_reason = match.groups()
                decisions[decision] += 1

                if decision == "WAIT":
                    rejections[reason] += 1
                    # Also track the router reason for secondary gates
                    if router_reason != reason:
                        rejections[f"{router_reason} (router)"] += 1
                elif decision in ("LONG_REVERSION", "SHORT_REVERSION"):
                    trades += 1

    return {
        'total_evals': evals,
        'trades': trades,
        'decisions': decisions,
        'rejections': rejections,
    }

def main():
    results = parse_logs()

    if not results or results['total_evals'] == 0:
        print("❌ No evaluation logs found")
        return

    print("\n" + "="*80)
    print("BOT REJECTION ANALYSIS")
    print("="*80)

    evals = results['total_evals']
    trades = results['trades']
    rejections = results['rejections']

    print(f"\nTotal evaluations: {evals}")
    print(f"Trade signals generated: {trades}")
    print(f"Pass rate: {100*trades/max(evals,1):.1f}%")

    if trades == 0:
        print("\n⚠️  NO TRADES EXECUTING - All candidates are being rejected")
        print("\nRejection breakdown (top reasons):")
        print("-" * 80)

        sorted_rejections = sorted(rejections.items(), key=lambda x: -x[1])
        for reason, count in sorted_rejections[:10]:
            pct = 100.0 * count / evals
            print(f"  {reason:50} : {count:3} ({pct:5.1f}%)")

        print("\n📋 LIKELY CAUSES:")

        # Analyze the pattern
        reason_counts = {r.replace(" (router)", ""): c for r, c in rejections.items()}

        if reason_counts.get("Dollar_Volume_Too_Low", 0) > evals * 0.8:
            print("  ❌ MIN_DOLLAR_VOLUME too high → Update .env: MIN_DOLLAR_VOLUME=300000")

        if reason_counts.get("Not_In_Reversion_Zone", 0) > evals * 0.5:
            print("  ⚠️  Price not below lower Bollinger Band → Market is not oversold")

        if reason_counts.get("RI_Not_Oversold", 0) > evals * 0.5:
            print("  ⚠️  RI/RSI not oversold → Market is trending, not mean-reverting")

        if reason_counts.get("ADX_Trend_Too_Strong", 0) > evals * 0.5:
            print("  ⚠️  ADX too high → Strong trend, not mean-reversion setup")
            print("      Consider: USE_TREND_FILTER=false in .env")

        if reason_counts.get("Higher_Timeframe_Trend_Too_Weak", 0) > evals * 0.5:
            print("  ⚠️  Price too far below trend EMA → Trend filter too strict")
            print("      Consider: USE_TREND_FILTER=false in .env")

        if reason_counts.get("hard_gate:Dollar_Volume_Too_Low", 0) > evals * 0.8:
            print("  ❌ Dollar volume gate blocking → Update .env: MIN_DOLLAR_VOLUME=300000")

        if reason_counts.get("hard_gate:Price_Too_Low", 0) > evals * 0.5:
            print("  ⚠️  Price below minimum → Update .env: MIN_PRICE=2.0")

        print("\n📊 NEXT STEPS:")
        print("  1. Check the top rejection reason above")
        print("  2. Update .env with the suggested fix")
        print("  3. Restart the bot")
        print("  4. Re-run this diagnostic to verify")

    else:
        print(f"\n✅ Trades are executing ({trades} signals generated)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
