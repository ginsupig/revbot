# Reversion Bot Production Package

This package hardens the original mean-reversion sketch into a production-oriented strategy module.

## What is included

- `ReversionEngine`: indicator calculation, validation, and entry decisioning
- `RiskManager`: account-risk sizing and bracket price planning
- `AlpacaExecutor`: Alpaca bracket execution wrapper
- `ReversionService`: high-level evaluation service for one symbol
- `run_monte_carlo`: portfolio risk simulation helper
- unit tests for engine, risk, and analytics

## Entry model

A long signal requires all of the following unless disabled in config:

- price inside lower reversion zone: `close <= lb1 and close > lb2`
- normalized reversion index below threshold
- ADX below threshold
- RSI below threshold
- optional reclaim of lower band
- optional bullish close confirmation
- optional volume expansion
- optional VWAP distance guard
- optional higher-timeframe EMA veto
- liquidity, spread, and price floors

## Required dataframe columns

- `open`
- `high`
- `low`
- `close`
- `volume`
- optional `spread_bps`

## Example

```python
import pandas as pd
from reversion_bot.service import ReversionService

svc = ReversionService()
df = pd.read_csv('bars.csv')
result = svc.evaluate_symbol('SPY', df, account_equity=100000)
print(result)
```

## Install

```bash
pip install numpy pandas pytest alpaca-py
pytest -q
```

## Notes

- This package does **not** include broker-state reconciliation, duplicate-order suppression, live market-session checks, or child-leg tracking for bracket replacement.
- For live deployment, add a session gate, open-position gate, order-idempotency layer, and market data freshness checks.
