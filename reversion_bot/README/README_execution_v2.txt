Execution Layer v2 Pack

Included files:
- config.py
- risk.py
- execution.py
- service.py
- main.py

What changed:
- risk.py now builds strategy-specific ATR plans for mean reversion, trend following, and trendfail
- invalid plans are rejected cleanly in service.py as WAIT states instead of raising execution-breaking errors
- execution.py adds has_open_long_position() so main.py can skip symbols that already have an open long
- main.py pre-checks open positions before evaluation to stop duplicate entry attempts
- config.py adds separate ATR multiples for trend and trendfail plans

Suggested env values:
- TRADE_LOOKBACK=160
- MIN_TRADE_SCORE=0.38
- MIN_RR=1.05
- TREND_STOP_ATR_MULTIPLE=1.0
- TREND_TARGET_ATR_MULTIPLE=2.5
- TRENDFAIL_STOP_ATR_MULTIPLE=1.1
- TRENDFAIL_TARGET_ATR_MULTIPLE=2.2

Drop-in order:
1. Replace reversion_bot/config.py
2. Replace reversion_bot/risk.py
3. Replace reversion_bot/execution.py
4. Replace reversion_bot/service.py
5. Replace main.py
