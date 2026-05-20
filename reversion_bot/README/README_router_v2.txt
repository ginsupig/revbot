Adaptive Router v2

Included files:
- config.py
- service.py
- main.py
- engine.py (unchanged from prior rewrite, included for convenience)
- ml.py (unchanged from prior rewrite, included for convenience)

What changed:
- Added a regime router so strong trend setups are no longer blocked just because RI is not oversold.
- Added router_reason and regime logging to explain WHY a trade was or was not taken.
- Added mixed-mode routing so the best sleeve can win if the composite is strong enough.
- Relaxed min_rr default from 1.20 to 1.05 for intraday viability.
- Kept long-only bracket execution.

Recommended first-pass .env values:
TRADE_LOOKBACK=160
MIN_TRADE_SCORE=0.38
TREND_FOLLOWING_MIN_SCORE=0.52
MEAN_REVERSION_MIN_SCORE=0.52
TRENDFAIL_MIN_SCORE=0.62
MIN_RR=1.05

Expected behavior:
- More trades in strong uptrends where trend_following is leading.
- Fewer cases where score > threshold but the system still prints WAIT only because RI_Not_Oversold.
- Better logs for tuning because regime and router_reason are now explicit.
