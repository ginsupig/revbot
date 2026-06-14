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

def sharpe_ratio(trades, risk_free_rate=0.0, periods_per_year: float = 252.0):
    """Annualized Sharpe of the per-bar return stream.

    ``periods_per_year`` must match the bar frequency or the number is on the
    wrong scale entirely: daily bars -> 252 (the old hardcoded behavior, kept
    as the default for backward compatibility); 5-minute RTH bars -> 78 * 252
    = 19_656. The walkforward passes the right factor for its timeframe.
    Note the magnitude only matters for absolute interpretation — autotune
    rankings are unaffected because the factor is constant within a run.
    """
    returns = trades['pnl'].dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=0)
    # No dispersion (e.g. an all-zero / no-trade fold) -> undefined Sharpe.
    # Return 0.0 rather than nan so it doesn't poison out-of-sample averages.
    if std == 0 or np.isnan(std):
        return 0.0
    return excess.mean() / std * np.sqrt(periods_per_year)

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


def extract_trade_returns(trades) -> np.ndarray:
    """Per-trade realized returns from a backtest frame: the nonzero entries of
    ``strategy_return`` (or ``pnl``). This is the empirical population the
    bootstrap resamples — the same trades ``_trade_count`` counts.
    """
    col = "strategy_return" if "strategy_return" in trades else "pnl"
    series = trades[col].dropna()
    return series[series != 0].to_numpy(dtype=float)


def run_bootstrap_monte_carlo(
    initial_invest: float,
    trade_returns,
    n_trades: Optional[int] = None,
    sims: int = 10_000,
    seed: Optional[int] = 42,
) -> dict:
    """Monte Carlo over the EMPIRICAL trade-return distribution.

    Unlike ``run_monte_carlo`` (which draws from a *fitted normal*), this
    resamples the strategy's actual per-trade returns with replacement and
    compounds each draw into an equity curve. It preserves the real skew and fat
    tails — exactly what drives VaR/CVaR and what a Gaussian understates for a
    stop-and-target reversion strategy (clipped upside, occasional full-stop
    losers). Returns the same keys as ``run_monte_carlo`` (``days`` -> ``n_trades``).

    i.i.d. trade bootstrap: trade ORDER is shuffled, so it assumes trades are
    independent events. If losing/winning streaks (autocorrelation) matter,
    a block bootstrap would be the next refinement.

    ``trade_returns`` are fractional per-trade returns (e.g. +0.012, -0.008);
    pass ``extract_trade_returns(backtest_df)`` for a walkforward frame.
    ``n_trades`` is the path length (default: the observed sample size).
    """
    arr = np.asarray(list(trade_returns), dtype=float)
    if initial_invest <= 0:
        raise ValueError('initial_invest must be > 0')
    if arr.size == 0:
        raise ValueError('trade_returns must be non-empty')
    if sims <= 0:
        raise ValueError('sims must be > 0')
    n = int(n_trades) if n_trades is not None else arr.size
    if n <= 0:
        raise ValueError('n_trades must be > 0')

    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(sims, n), replace=True)
    finals = initial_invest * np.prod(1.0 + draws, axis=1)

    p5 = float(np.percentile(finals, 5))
    tail = finals[finals <= p5]
    return {
        'initial_invest': float(initial_invest),
        'expected_final_value': float(finals.mean()),
        'median_final_value': float(np.median(finals)),
        'best_case': float(finals.max()),
        'worst_case': float(finals.min()),
        'percentile_5_value': p5,
        'var_95': float(initial_invest - p5),
        'cvar_95': float(initial_invest - tail.mean()),
        'n_trades': n,
        'sims': int(sims),
        'sample_size': int(arr.size),
    }
