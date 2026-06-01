import argparse
import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from reversion_bot.walkforward import run_walkforward_backtest, evaluate_fixed_params
from reversion_bot.allowlist import select_allowlist
from run_real_backtest import fetch_alpaca_bars
from datetime import datetime, timedelta

load_dotenv()

# Maps walk-forward param names to .env keys
PARAM_TO_ENV = {
    'band_length':            'BAND_LENGTH',
    'ri_threshold':           'RI_THRESHOLD',
    'use_vwap_filter':        'USE_VWAP_FILTER',
    'max_vwap_extension_pct': 'MAX_VWAP_EXTENSION_PCT',
    'rsi_max':                'RSI_MAX',
    'min_history':            'TRADE_LOOKBACK',
}

INTEGER_PARAMS = {'band_length', 'min_history'}
BOOL_PARAMS = {'use_vwap_filter'}


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
    parser = argparse.ArgumentParser(
        description="Walk-forward autotuner + per-symbol allowlist gate."
    )
    parser.add_argument(
        'days', nargs='?', type=int, default=None,
        help='lookback window in days (default: AUTOTUNE_DAYS env or 30)',
    )
    parser.add_argument(
        '--symbols', type=str, default=None,
        help='comma-separated symbols to tune (overrides TRADE_SYMBOL / fallback list)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='score symbols and print the scorecard WITHOUT writing .env',
    )
    args = parser.parse_args()

    # Symbols: --symbols flag > TRADE_SYMBOL env var > fallback watchlist
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        env_symbols = os.getenv('TRADE_SYMBOL', '')
        if env_symbols:
            symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
        else:
            symbols = ["MU", "WDC", "ASTS", "NVDA", "AMD", "SMCI", "CRWD", "AAPL", "TSLA", "APP", "META", "INOD"]

    # Lookback window: positional days arg > AUTOTUNE_DAYS env var > default 30
    days = args.days if args.days is not None else int(os.getenv('AUTOTUNE_DAYS', 30))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start = start_dt.strftime('%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    timeframe = '5Min'

    print(f"Autotune | symbols={len(symbols)} | window={days}d ({start} → {end})")

    # Task 4: tune deep-oversold RI x VWAP extension (the dimensions that matter
    # for fast 3x leveraged assets), not the lagging Bollinger width. Centered so
    # the optimum sits inside the grid and folds actually contain trades.
    param_grid = {
        'band_length':            [20],
        'ri_threshold':           [-1.0, -0.5, 0.0, 0.5],
        'use_vwap_filter':        [True, False],
        'max_vwap_extension_pct': [0.012, 0.020, 0.030],
        'rsi_max':                [40, 48, 55],
        'min_history':            [160],
    }

    all_best_params = []
    symbol_scores = {}      # symbol -> tune-best OOS PF/Sharpe (informational)
    bars_by_symbol = {}     # cache bars for the as-traded validation pass

    for symbol in symbols:
        try:
            print(f"\n--- Autotune: {symbol} ---")
            bars = fetch_alpaca_bars(symbol, start, end, timeframe)

            if bars is None or len(bars) < 160:
                print(f"Skipping {symbol}: insufficient data ({len(bars) if bars is not None else 0} bars).")
                continue

            print(f"Fetched {len(bars)} bars.")
            bars_by_symbol[symbol] = bars

            results, oos_metrics, best_params = run_walkforward_backtest(
                bars, param_grid=param_grid, n_splits=3
            )

            print(f"Best params: {best_params}")
            if oos_metrics:
                print(f"Best OOS metrics: {oos_metrics[-1]}")
                # Average OOS metrics across folds -> the symbol's scorecard.
                symbol_scores[symbol] = {
                    "profit_factor": float(np.mean([m["profit_factor"] for m in oos_metrics])),
                    "sharpe": float(np.mean([m["sharpe"] for m in oos_metrics])),
                }

            all_best_params.append(best_params)

        except Exception as e:
            print(f"Error tuning {symbol}: {e}")

    if not all_best_params:
        print("\nNo valid results — .env not updated.")
        return

    # Aggregate across symbols: median for numerics, majority vote for bools.
    aggregated = {}
    for param in PARAM_TO_ENV:
        values = [p[param] for p in all_best_params if param in p]
        if not values:
            continue
        if param in BOOL_PARAMS:
            aggregated[param] = sum(bool(v) for v in values) >= (len(values) / 2)
        else:
            aggregated[param] = float(np.median([float(v) for v in values]))

    # Format for .env
    env_updates = {}
    for param, env_key in PARAM_TO_ENV.items():
        if param not in aggregated:
            continue
        val = aggregated[param]
        if param in BOOL_PARAMS:
            env_updates[env_key] = "True" if val else "False"
        elif param in INTEGER_PARAMS:
            env_updates[env_key] = str(int(round(val)))
        else:
            # General float formatting (e.g. RI -1.0, VWAP 0.012) without
            # truncating small fractions to a single decimal.
            env_updates[env_key] = f"{val:g}"

    # As-traded validation pass: re-score every symbol with the AGGREGATE params
    # the bot will actually trade with (not each symbol's bespoke best). A name
    # that only wins under its own optimum but loses under the shared median
    # params must NOT make the allowlist — otherwise the bot trades a live loser
    # (e.g. QQQ passing on its own params but losing on the median ones).
    as_traded_params = {}
    for param in PARAM_TO_ENV:
        if param not in aggregated:
            continue
        val = aggregated[param]
        if param in BOOL_PARAMS:
            as_traded_params[param] = bool(val)
        elif param in INTEGER_PARAMS:
            as_traded_params[param] = int(round(val))
        else:
            as_traded_params[param] = float(val)

    as_traded_scores = {}
    for symbol, bars in bars_by_symbol.items():
        try:
            m = evaluate_fixed_params(bars, as_traded_params, n_splits=3)
            as_traded_scores[symbol] = {
                "profit_factor": float(np.mean([x["profit_factor"] for x in m])),
                "sharpe": float(np.mean([x["sharpe"] for x in m])),
            }
        except Exception as e:
            print(f"As-traded scoring failed for {symbol}: {e}")

    # Per-symbol allowlist gate, scored AS TRADED. Thresholds are configurable;
    # defaults reject break-even/losing names.
    min_pf = float(os.getenv('ALLOWLIST_MIN_PF', 1.10))
    min_sharpe = float(os.getenv('ALLOWLIST_MIN_SHARPE', 0.0))
    allowlist = select_allowlist(as_traded_scores, min_pf, min_sharpe)

    print(f"\n--- As-traded scorecard (shared median params; gate: PF>={min_pf:g}, Sharpe>={min_sharpe:g}) ---")
    print(f"    params: {as_traded_params}")
    for sym, m in sorted(as_traded_scores.items(), key=lambda kv: kv[1]['profit_factor'], reverse=True):
        mark = "PASS" if sym in allowlist else "drop"
        tune_pf = symbol_scores.get(sym, {}).get('profit_factor', float('nan'))
        print(f"  [{mark}] {sym:<6} as-traded PF={m['profit_factor']:.2f}  Sharpe={m['sharpe']:.2f}"
              f"   (tune-best PF={tune_pf:.2f})")
    if allowlist:
        print(f"Allowlist: {', '.join(allowlist)}")
    else:
        print("Allowlist: (empty) — no symbol cleared the gate; the bot will trade nothing.")
    env_updates['TRADE_ALLOWLIST'] = ','.join(allowlist)

    print("\n--- Aggregated best params (median across symbols) ---")
    for env_key, val in env_updates.items():
        print(f"  {env_key}={val}")

    if args.dry_run:
        print("\n[dry-run] .env NOT modified.")
        print(f"[dry-run] Would set TRADE_ALLOWLIST={env_updates['TRADE_ALLOWLIST']}")
        return

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    update_env_file(env_path, env_updates)
    print(f"\n.env updated: {env_path}")


if __name__ == "__main__":
    main()