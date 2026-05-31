from __future__ import annotations

from typing import Optional

import numpy as np

# Configs that barely trade produce a zero-loss fold -> PF=inf, which lets the
# optimizer "win" by not trading. Require at least this many round-trip trades
# for a profit_factor to count; below it, the score is neutralized.
MIN_TRADES_FOR_SCORE = 5


def _trade_count(trades) -> int:
    """Number of completed round-trip trades (nonzero strategy_return bars)."""
    col = "strategy_return" if "strategy_return" in trades else "pnl"
    series = trades[col].dropna()
    return int((series != 0).sum())


def profit_factor(trades):
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -trades[trades['pnl'] < 0]['pnl'].sum()
    # Too few trades to be meaningful -> neutral score, never inf.
    if _trade_count(trades) < MIN_TRADES_FOR_SCORE:
        return 0.0
    if gross_loss == 0:
        # Profitable with no losers but enough trades: cap instead of inf so the
        # optimizer can still rank it without an unbounded value dominating.
        return 10.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def sharpe_ratio(trades, risk_free_rate=0.0):
    returns = trades['pnl'].dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252
    std = excess.std(ddof=0)
    # No dispersion (e.g. an all-zero / no-trade fold) -> undefined Sharpe.
    # Return 0.0 rather than nan so it doesn't poison out-of-sample averages.
    if std == 0 or np.isnan(std):
        return 0.0
    return excess.mean() / std * np.sqrt(252)

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
