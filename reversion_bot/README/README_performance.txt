RevBot Performance Layer v1

Purpose
- add persistent trade journaling
- track setup quality and outcomes by strategy/regime/symbol/time bucket
- compute lightweight adaptive threshold suggestions
- prepare the bot for data-driven tuning instead of intuition-only tuning

Files
- performance.py
- service.py
- main.py
- config.py

Drop-in order
1. Replace reversion_bot/config.py
2. Replace reversion_bot/service.py
3. Add reversion_bot/performance.py
4. Replace main.py

New behavior
- every evaluation can emit a journal row to state/performance/evaluations.jsonl
- every planned trade can emit a journal row to state/performance/trades.jsonl
- adaptive threshold suggestions are derived from rolling win-rate / expectancy by entry_style + regime
- main.py prints a compact performance summary every cycle

Suggested env
MIN_TRADE_SCORE=0.36
MIN_RR=1.00
PERF_STATE_DIR=state/performance
PERF_ENABLE_ADAPTIVE_THRESHOLD=true
PERF_MIN_SAMPLES=20
PERF_MAX_THRESHOLD_ADJ=0.05
