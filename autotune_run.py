import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from reversion_bot.walkforward import run_walkforward_backtest
from run_real_backtest import fetch_alpaca_bars
from datetime import datetime, timedelta

load_dotenv()

# Maps walk-forward param names to .env keys
PARAM_TO_ENV = {
    'band_length':  'BAND_LENGTH',
    'band_std_1':   'BAND_STD_1',
    'band_std_2':   'BAND_STD_2',
    'ri_threshold': 'RI_THRESHOLD',
    'rsi_max':      'RSI_MAX',
    'adx_max':      'ADX_MAX',
    'min_history':  'TRADE_LOOKBACK',
}

INTEGER_PARAMS = {'band_length', 'min_history'}


def update_env_file(env_path, updates):
    """Update key=value lines in a .env file in-place."""
    with open(env_path, 'r') as f:
        lines = f.readlines()

    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}\n")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append any keys that didn't already exist in the file
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)


def main():
    # Symbols: from TRADE_SYMBOL env var, else fallback watchlist
    env_symbols = os.getenv('TRADE_SYMBOL', '')
    if env_symbols:
        symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
    else:
        symbols = ["TQQQ", "SOXL", "TECL", "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU"]

    # Lookback window: CLI arg (days) > AUTOTUNE_DAYS env var > default 30
    days = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv('AUTOTUNE_DAYS', 30))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start = start_dt.strftime('%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    timeframe = '5Min'

    print(f"Autotune | symbols={len(symbols)} | window={days}d ({start} → {end})")

    param_grid = {
        'band_length':  [20, 50],
        'band_std_1':   [1.5, 2.0],
        'band_std_2':   [2.5, 3.0],
        'ri_threshold': [-1.0, -0.5, 0.0],
        'rsi_max':      [30, 35],
        'adx_max':      [25, 30],
        'min_history':  [160],
    }

    all_best_params = []

    for symbol in symbols:
        try:
            print(f"\n--- Autotune: {symbol} ---")
            bars = fetch_alpaca_bars(symbol, start, end, timeframe)

            if bars is None or len(bars) < 160:
                print(f"Skipping {symbol}: insufficient data ({len(bars) if bars is not None else 0} bars).")
                continue

            print(f"Fetched {len(bars)} bars.")

            results, oos_metrics, best_params = run_walkforward_backtest(
                bars, param_grid=param_grid, n_splits=3
            )

            print(f"Best params: {best_params}")
            if oos_metrics:
                print(f"Best OOS metrics: {oos_metrics[-1]}")

            all_best_params.append(best_params)

        except Exception as e:
            print(f"Error tuning {symbol}: {e}")

    if not all_best_params:
        print("\nNo valid results — .env not updated.")
        return

    # Aggregate: median across all symbols for each tunable param
    aggregated = {}
    for param in PARAM_TO_ENV:
        values = [p[param] for p in all_best_params if param in p]
        if values:
            aggregated[param] = np.median(values)

    # Format for .env
    env_updates = {}
    for param, env_key in PARAM_TO_ENV.items():
        if param not in aggregated:
            continue
        val = aggregated[param]
        env_updates[env_key] = str(int(round(val))) if param in INTEGER_PARAMS else f"{val:.1f}"

    print("\n--- Aggregated best params (median across symbols) ---")
    for env_key, val in env_updates.items():
        print(f"  {env_key}={val}")

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    update_env_file(env_path, env_updates)
    print(f"\n.env updated: {env_path}")


if __name__ == "__main__":
    main()