Rewrite pack summary

Files included:
- config.py
- engine.py
- service.py
- ml.py
- main.py

Primary changes:
- Removes RSI/ADX as routine hard rejects and reserves hard blocking for true safety cases.
- Lowers the score floor to a realistic live intraday range.
- Allows trend-following and trend-fail sleeves to contribute meaningfully.
- Increases default lookback so ML is no longer permanently neutral.
- Adds clearer live logging with component scores and entry style.

Suggested env defaults for first live paper pass:
- TRADE_LOOKBACK=160
- MIN_TRADE_SCORE=0.38
- RSI_MAX=45
- ADX_MAX=40
- REQUIRE_RECLAIM_LB1=False
- REQUIRE_BULLISH_CLOSE=False
- REQUIRE_VOLUME_EXPANSION=False
