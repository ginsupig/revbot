import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from reversion_bot.autotune import AutoTuner

def dummy_strategy(df, fast=5, slow=10):
    df = df.copy()
    df['fast'] = df['close'].rolling(fast).mean()
    df['slow'] = df['close'].rolling(slow).mean()
    df['signal'] = (df['fast'] > df['slow']).astype(int)
    df['pnl'] = df['signal'] * df['close'].pct_change().fillna(0)
    return df

def profit_factor(trades):
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -trades[trades['pnl'] < 0]['pnl'].sum()
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss

def test_autotuner_tune():
    n = 100
    np.random.seed(1)
    close = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({'close': close})
    param_grid = {'fast': [3, 5], 'slow': [8, 10]}
    tuner = AutoTuner(dummy_strategy, df, param_grid)
    best_params, best_score = tuner.tune(profit_factor, n_splits=3)
    assert isinstance(best_params, dict)
    assert best_score > 0


# --- aggregate_best_params: VWAP-extension dilution fix ---------------------

def test_vwap_extension_aggregates_over_vwap_on_symbols_only():
    from autotune_run import aggregate_best_params
    # Three names turn VWAP ON at a sane 0.02; two leave it OFF but carry a tiny
    # placeholder extension (0.012). The naive median across all five would land
    # at the harmful 0.012; the fix must median over the vwap-ON names -> 0.02.
    best = [
        {"use_vwap_filter": True,  "max_vwap_extension_pct": 0.02, "rsi_max": 40},
        {"use_vwap_filter": True,  "max_vwap_extension_pct": 0.02, "rsi_max": 40},
        {"use_vwap_filter": True,  "max_vwap_extension_pct": 0.02, "rsi_max": 48},
        {"use_vwap_filter": False, "max_vwap_extension_pct": 0.012, "rsi_max": 44},
        {"use_vwap_filter": False, "max_vwap_extension_pct": 0.012, "rsi_max": 44},
    ]
    agg = aggregate_best_params(best)
    assert agg["use_vwap_filter"] is True            # majority vote on
    assert agg["max_vwap_extension_pct"] == 0.02      # not diluted to 0.012
    assert agg["rsi_max"] == 44.0                     # plain median of all 5, unaffected


def test_vwap_extension_empty_when_no_symbol_uses_filter():
    from autotune_run import aggregate_best_params
    # If no symbol turns VWAP on, the extension simply isn't emitted (nothing to
    # base it on) and the filter aggregates off.
    best = [
        {"use_vwap_filter": False, "max_vwap_extension_pct": 0.012},
        {"use_vwap_filter": False, "max_vwap_extension_pct": 0.020},
    ]
    agg = aggregate_best_params(best)
    assert agg["use_vwap_filter"] is False
    assert "max_vwap_extension_pct" not in agg
