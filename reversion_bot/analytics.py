from __future__ import annotations

from typing import Optional

import numpy as np

def profit_factor(trades):
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -trades[trades['pnl'] < 0]['pnl'].sum()
    if gross_loss == 0:
        return float('inf')
    return gross_profit / gross_loss

def sharpe_ratio(trades, risk_free_rate=0.0):
    returns = trades['pnl'].dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252
    return excess.mean() / excess.std(ddof=0) * np.sqrt(252)

def max_drawdown(trades):
    equity = trades['pnl'].cumsum()
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    return drawdown.min()


def run_monte_carlo(
    initial_invest: float,
    daily_mean: float,
    daily_std: float,
    days: int = 252,
    sims: int = 10_000,
    seed: Optional[int] = 42,
) -> dict:
    if initial_invest <= 0:
        raise ValueError('initial_invest must be > 0')
    if daily_std < 0:
        raise ValueError('daily_std must be >= 0')
    if days <= 0 or sims <= 0:
        raise ValueError('days and sims must be > 0')

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(sims):
        returns = rng.normal(daily_mean, daily_std, days)
        final_value = initial_invest * np.prod(1.0 + returns)
        results.append(final_value)

    arr = np.asarray(results, dtype=float)
    p5 = np.percentile(arr, 5)
    tail = arr[arr <= p5]
    return {
        'initial_invest': float(initial_invest),
        'expected_final_value': float(arr.mean()),
        'median_final_value': float(np.median(arr)),
        'best_case': float(arr.max()),
        'worst_case': float(arr.min()),
        'percentile_5_value': float(p5),
        'var_95': float(initial_invest - p5),
        'cvar_95': float(initial_invest - tail.mean()),
        'days': int(days),
        'sims': int(sims),
    }
