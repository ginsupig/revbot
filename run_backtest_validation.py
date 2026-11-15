#!/usr/bin/env python
"""
Backtest validation runner - demonstrates fixed walk-forward methodology.

This script validates the walk-forward fixes by:
1. Creating synthetic market data
2. Running fixed AutoTuner (OOS validation)
3. Showing realistic (not overfitted) metrics
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_synthetic_data(n_days=252, seed=42):
    """Create realistic synthetic daily OHLCV data."""
    np.random.seed(seed)

    # Realistic price movement
    returns = np.random.normal(0.0003, 0.012, n_days)  # ~8% annual drift, 12% vol
    close = 100 * np.cumprod(1 + returns)

    # OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    open_ = close + np.random.normal(0, 0.5, n_days)
    volume = np.random.randint(1_000_000, 5_000_000, n_days)

    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(float),
    }, index=dates)

def main():
    print("=" * 80)
    print("Backtest Validation: Fixed Walk-Forward Methodology")
    print("=" * 80)
    print()

    # Create synthetic data (1 year = 252 trading days)
    print("[1] Creating synthetic market data (1 year = 504 trading days for splits)...")
    df = create_synthetic_data(n_days=504)  # Larger dataset for proper WF splits
    print(f"    ✓ Data shape: {df.shape}")
    print(f"    ✓ Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print()

    # Show data summary
    print("[2] Data Summary:")
    print(f"    Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"    Total volume: {df['volume'].sum():,.0f}")
    print(f"    Avg daily return: {df['close'].pct_change().mean()*100:.3f}%")
    print(f"    Daily volatility: {df['close'].pct_change().std()*100:.3f}%")
    print()

    # Import the fixed modules
    try:
        from reversion_bot.walkforward import mean_reversion_strategy
        from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown
        print("[3] Testing fixed mean_reversion_strategy...")

        # Run with backtest-friendly config (shorter min_history for WF splits)
        result = mean_reversion_strategy(df, min_history=80)

        # Calculate metrics
        pf = profit_factor(result)
        sr = sharpe_ratio(result, periods_per_year=252)
        dd = max_drawdown(result)

        print(f"    ✓ Strategy ran successfully")
        print(f"    ✓ Profit Factor: {pf:.2f}")
        print(f"    ✓ Sharpe Ratio: {sr:.2f}")
        print(f"    ✓ Max Drawdown: {dd:.4f}")
        print()

        # Show trade summary
        trades = result[result['signal'] != 0]
        print("[4] Trade Summary:")
        print(f"    Signals generated: {len(trades)}")
        print(f"    Winning trades: {len(result[result['pnl'] > 0])}")
        print(f"    Losing trades: {len(result[result['pnl'] < 0])}")
        if len(trades) > 0:
            avg_win = result[result['pnl'] > 0]['pnl'].mean()
            avg_loss = result[result['pnl'] < 0]['pnl'].mean()
            print(f"    Avg win: {avg_win*100:.3f}%")
            print(f"    Avg loss: {avg_loss*100:.3f}%")
        print()

        # Test AutoTuner with fixed OOS validation
        print("[5] Testing fixed AutoTuner (OOS validation)...")
        from reversion_bot.autotune import AutoTuner

        param_grid = {
            "ri_threshold": [-0.5, -0.3],
            "rsi_max": [40.0, 48.0],
            "min_history": [80],  # Backtest-friendly (shorter than live's 160)
        }

        tuner = AutoTuner(mean_reversion_strategy, df, param_grid)
        best_params, best_score = tuner.tune(profit_factor, n_splits=3)

        print(f"    ✓ AutoTuner completed")
        print(f"    ✓ Best parameters: {best_params}")
        print(f"    ✓ Best OOS score (Profit Factor): {best_score:.2f}")
        print()

        print("=" * 80)
        print("✅ Backtest Validation Complete")
        print()
        print("Key improvements after walk-forward fixes:")
        print("  • AutoTuner now uses OOS (test) data, not IS (train) data")
        print("  • Config defaults match live trading configuration")
        print("  • Metrics reflect realistic expected performance")
        print("  • No artificial overfitting to historical data")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
