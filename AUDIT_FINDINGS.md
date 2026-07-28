# Full Code Audit — Findings

Date: 2026-07-28. Scope: entire live path (`main.py`, `reversion_bot/`), data fetch,
persistence, walkforward/autotune, and the recently added diagnostic scripts.
Method: four independent subsystem reviews, top findings re-verified by hand against
the code (the #1 finding was reproduced end-to-end on synthetic data). Baseline:
all 484 unit tests pass.

Severity: **CRITICAL** = can lose money, leave a position unprotected, or halt/corrupt
trading silently. **HIGH** = wrong trading behavior. **MEDIUM** = latent bug.
**LOW** = smell/nit.

---

## CRITICAL

### C1. The router can BUY names the engine explicitly vetoed (falling knife / blow-off top)
`reversion_bot/service.py:142-198`. `go_long = passes_score and not is_short_signal`
never requires `decision.signal == "LONG_REVERSION"`. Only three liquidity reasons are
hard gates; every other engine veto — `Downtrend_Too_Extended`, `Momentum_Too_Extended`,
`ADX_Trend_Too_Strong`, `RI_Not_Oversold`, `ATR_Invalid` — can be bypassed because
`_score_mean_reversion` still awards depth (+0.20), RI-stretch (+0.15) and RSI-softness
(+0.10) bonuses to a WAIT decision (0.15 base → up to 0.60 total ≥ the 0.45 gate), and
`_route_style`'s `best_style_edge_mean_reversion` branch routes it straight to the gate.
**Reproduced**: a synthetic accelerating downtrend (ADX 96.9, RSI 0.3) returned
`decision=WAIT reason=Downtrend_Too_Extended` yet `go_long=True` with a real ~$20k
PositionPlan; the governor only checks `go_long`, so the order would be placed. This is
verbatim the failure mode the `config.py:36-38` hard guard was built to prevent.
**Fix direction**: hard-gate ALL non-`LONG_REVERSION`/`SHORT_REVERSION` decisions (or at
minimum add the extended/too-strong reasons to `hard_wait_reasons`), and stop awarding
validation-dependent bonuses to unvalidated signals.

### C2. EOD/carryover flatten cancels ALL protective stops, then gives up on failed closes — unprotected positions overnight
`main.py:431-480` (`liquidate_all_positions`) calls `cancel_all_orders()` first, then
retries `close_position` 4×; on final failure it only prints and moves on. A position
whose close is rejected (the live 7/3 failure mode) is left with NO stop and NO target.
Worse, `reconcile_carryover` (`main.py:420-428`) stamps the day "reconciled" even when
every close failed, so an overnight orphan is never retried until tomorrow. Same
cancel-then-close pattern in `close_long`/`close_short` (`execution.py:307-349`) can
strand a channel-exited position unprotected intraday (the exit never re-fires once
price slips back under the threshold, and the trailing stop requires an existing stop
leg to update). **Fix direction**: only stamp reconciled on success; on close failure,
re-arm a protective stop; cancel per-symbol orders instead of account-wide cancel-all.

### C3. Daily/swing timeframe is silently dead: fetch window math is wrong for every non-5-minute timeframe
`main.py:163-167`: `bars_per_day = 78 if "5" in tf else 390`. For `TRADE_TIMEFRAME=1Day`
(the documented `SWING_OVERNIGHT_MODE` pairing) this fetches **3 calendar days** (~2 daily
bars) against a 160-bar lookback — every symbol returns `None`, every cycle, forever,
while the heartbeat stays green. Also wrong for 15Min ("5" in "15min" → 78 assumed vs 26
actual), 30Min and 1Hour. The channel exit's daily fetch (`main.py:499-523`) reuses the
same window → `USE_CHANNEL_EXIT=true` is silently non-functional too. Verified by hand:
`get_fetch_days("1Day", 160) == 3`. Zero tests cover `get_fetch_days`/`build_fetch_window`.

### C4. Liquidity is measured on the IEX feed (~2-3% of consolidated tape) — the dollar-volume gate rejects genuinely liquid names
`run_real_backtest.py:89,133` pin `feed='iex'`. The engine's
`avg_dollar_volume` (`engine.py:71-74`) therefore understates true dollar volume by
~30-50×: the $750k gate effectively demanded ~$25-40M/day of real liquidity, which is
why the bot rejected everything on `Dollar_Volume_Too_Low`. **The recent change to
MIN_DOLLAR_VOLUME=300000 recalibrated the gate to a feed artifact rather than fixing
the measurement** — it silently breaks if the feed changes or IEX share drifts. Volume
expansion/RVOL confirmations share the distortion. **Fix direction**: use SIP feed if
subscribed, or explicitly document/scale the IEX-basis threshold as such.

### C5. Stale single-instance lock after a reboot can halt trading permanently, with exit code 0
`single_instance.py:81-82` + `main.py:686-689`: after a crash/reboot the lock file
survives with the old PID; if the OS reassigned that PID to any process, `_pid_alive()`
returns True and main exits 0 ("clean, stop" to the supervisor). The bot never trades
again until someone deletes the lock. Also a TOCTOU in stale-reclaim can let two racing
starts both run (double-placing orders).

### C6. (Now removed) `backtest_threshold_comparison.py` wrote phantom records into LIVE adaptive-threshold state
The script instantiated `ReversionService` with the default
`state_dir="state/performance"` — every run appended fake EvalRecords/TradeRecords to
the same files `reconcile_outcomes` uses to attribute REAL broker fills, feeding the
live adaptive threshold fabricated evidence (confirmed on disk: phantom AAPL records at
2026-07-28T00:35). It also wasn't a backtest: it evaluated only the latest bar, its
180-calendar-day fetch produced ~124 bars vs the 160 required (so both runs evaluated
**zero** symbols and compared 0-vs-0), and its universe fallback always used 17
hardcoded names. `test_dollar_volume_thresholds.py` called `fetch_alpaca_bars` with the
wrong signature and could never have produced output. `diagnose_rejection_reasons.py`
counted raw engine signals over the ENTIRE historical log — its "trades are executing /
13.2% pass rate" conclusion did not measure executed, approved, or even router-passed
trades. **All three scripts are deleted in this commit.** If any were run on the live
machine, delete the phantom entries from `state/performance/evaluations.jsonl` and
`trades.jsonl` (or reset the directory) before the next live session.

---

## HIGH

- **H1. Position caps can be overshot by in-flight orders.** `governor.py:118` counts
  only filled positions; resting entry brackets (USE_LIMIT_ENTRY defaults true) are
  invisible to `max_open_positions`, exposure, and heat — 3 positions + 2 approvals in
  one cycle → 5 positions. (`governor.py:113-116` only blocks same-symbol dups.)
- **H2. `parse_bool("")` returns False, not the default** (`main.py:158-161`). Blanking
  a line like `USE_MARKET_REGIME_FILTER=` or `USE_TRAILING_STOP=` (the way
  `.env.example` blanks `TRADE_WATCHLIST=`) silently disables validated safety
  defaults. Unrecognized tokens ("Ture") also become False.
- **H3. `.env.example` contradicts validated live defaults on ≥8 settings** while
  claiming "defaults shown": `ENABLE_SHORTS=true` (code default False; shorts documented
  as OOS-bleeding), `USE_TREND_FILTER=False` (vs True), `MIN_TRADE_SCORE=0.36` (vs 0.45,
  widening C1), `TARGET_ATR_MULTIPLE=2.00` (vs validated 3.50), `RSI_MAX=48` (vs 40),
  `RSI_MIN=52` (vs 65), `MIN_RR=1.5` (vs 1.00), `MIN_PRICE=5.0` (vs 2.0). Copying the
  example flips the bot into configurations the repo's own backtests condemned.
- **H4. The default universe is arbitrary, not top-market-cap.** `execution.py:95` sorts
  by `getattr(x, 'market_cap', 0)` — Alpaca assets have no `market_cap`, all keys are 0,
  so the bot trades the first ~40 assets in API order (roughly alphabetical), and the
  vetted fallback watchlist never engages.
- **H5. Short side is missing the ADX trend-strength veto.** `engine.py:410-464`: longs
  reject `adx > adx_max` (40); `_evaluate_short` has no equivalent — it will short an
  ADX-45 uptrend rip that the long side would call `ADX_Trend_Too_Strong`.
- **H6. No HTTP timeout on the trading REST client** (`execution.py:24`).
  `ALPACA_HTTP_TIMEOUT` covers only market-data calls; a hung `get_clock`/
  `list_positions`/`submit_order` freezes the loop indefinitely — no EOD flatten, no
  trailing-stop management.
- **H7. Trailing stop is dead under `USE_ALPACA_PY=true`.**
  `execution.py:230-233` calls `replace_order(order_id=..., stop_price=...)` but the
  facade's signature is `replace_order(order_id, limit_price)` (`alpaca_py_client.py:
  177-180`) — TypeError on every trail update, swallowed by a blanket except.
- **H8. Walkforward PnL is double-counted for multi-bar trades**
  (`walkforward.py:147,152` + short mirror): held bars book mark-to-market returns AND
  the exit bar books the full entry→exit return again. Every walkforward PF/Sharpe/DD
  in the repo is inflated in both directions.
- **H9. The autotuner's "out-of-sample" is selected on the test folds themselves**
  (`autotune.py:19-25` + `walkforward.py:404-419`): params are argmaxed directly on the
  evaluation folds, then those same folds are reported as OOS — lookahead by
  construction, violating the repo's own research discipline.

## MEDIUM (selected)

- **M1.** Adaptive-threshold *loosening* is dead code: every route carries its own floor
  ≥ baseline, so `baseline − adj` can never admit a previously blocked candidate
  (`service.py:129-137`, `349-356`). Raising works; and it learns from all outcomes
  forever (no recency window), some of which may be misattributed (see M3/M4).
- **M2.** The ML component (25% of the blend, lock-serialized retrains that stall other
  symbol threads) affects **no trading decision** — the gate and sizing use only the
  routed component score; ML reaches only the logged blend (`service.py:171,221-227`).
- **M3.** TradeRecords are logged at decision time, before governor approval/fill
  (`service.py:242-260`), so `reconcile_outcomes` can attribute real closes to vetoed
  phantom entries — the adaptive threshold learns from the wrong (style, regime) cell.
- **M4.** Outcome reconcile: cursor advances past skipped events (fills lost forever,
  `performance.py:231-257`); lexical ISO-timestamp comparison breaks across `Z` vs
  `+00:00` vs space-separated formats (`performance.py:166,235`); a tz-naive outcome
  timestamp raises outside the try and silently kills that symbol's evaluation.
- **M5.** `PerformanceTracker` has zero locking under ~90 concurrent eval threads and
  two processes can interleave; `ml_eval_count += 1` races outside `_ml_lock`
  (`service.py:436`); works today mostly by CPython/Linux accident.
- **M6.** `PortfolioState` JSON writes are non-atomic and cwd-relative
  (`portfolio.py:28,64-66`): crash mid-write or launching from a different directory
  silently resets cooldowns, trail state, and the drawdown peak. Corruption falls back
  to defaults with no warning.
- **M7.** EOD env values parsed inside the loop each cycle — a malformed
  `EOD_LIQUIDATION_MINUTES` becomes a crash loop that starts at market open
  (`main.py:354,367`); half-day close fallback assumes 15:00 CT if the clock API fails
  (`main.py:334-345`); poll interval > liquidation window can hop over the flatten.
- **M8.** Fractional positions (`int(float(pos.qty)) == 0`) are never flattened
  (`main.py:455-457`). `cancel_all_orders()` nukes manual/other-strategy orders on the
  shared account (`main.py:437`).
- **M9.** Live VWAP never session-resets (RangeIndex after `reset_index()` defeats the
  session grouping, `indicators.py:97-104`) — the VWAP filter, if enabled, measures
  against a multi-day anchor. Signals are computed on the still-forming bar
  (`end=now`), making close/bullish-close/volume confirmations unstable if enabled.
- **M10.** Regime gates fail open: any SPY fetch error = risk-on, and the sector cache
  stores fail-open results for a full hour (`main.py:212-233,264-269`) — the gate is
  most likely to be off exactly on a stressed, flaky-feed day. In persistent mode the
  universe/allowlist/params are resolved once at startup and never refreshed.
- **M11.** Unbounded growth: `evaluations.jsonl` (one record per symbol per cycle,
  full-file read to slice `[-500:]`), `trades.jsonl` re-read fully on every reconcile,
  `reversion_service.log` never rotates.

## LOW (selected)

- Conviction boost inflates risk budget up to +35% over `RISK_PER_TRADE_PCT`
  (`risk.py:114-115`), contradicting the adjacent invariant comment.
- `.env.example` documents 4 vars the live bot never reads (`TRADE_SYMBOL`,
  `ALLOWLIST_MIN_PF/SHARPE`, `ALPACA_HTTP_TIMEOUT` for trading calls) while ~30 vars
  main.py DOES read are undocumented (all portfolio caps, `USE_TRAILING_STOP`,
  `SWING_OVERNIGHT_MODE`, `RUN_PERSISTENT`, ...).
- RSI reads 50 (neutral) instead of 100 on all-gain windows (`indicators.py:43-45`);
  ADX NaN→0 passes "no trend" checks on degenerate tapes (`indicators.py:75`).
- `RiskConfig.round_lot` unused; `within_minutes_before_close` (tested) imported but
  bypassed in favor of the untested env/clock path; cooldown timestamp parse failures
  fail open.
- Config-default drift between dataclasses and main.py env defaults: `MIN_RR` 1.00 vs
  1.5, `MAX_POSITION_VALUE_PCT` 0.20 vs 0.15, `use_limit_entry` False vs True.

## Verified correct (checked, not findings)

Short-side bracket geometry (stop above / target below) everywhere incl. trailing
mirror; trailing ratchet only tightens and replace is broker-atomic on the legacy path;
cooldowns/high-water marks persisted tz-aware; `risk_per_share=0` and `min_qty`
guards; FIFO PnL engines in `trade_report.py`/`pnl_report.py` mutually consistent;
`bar_batch.split_bars_by_symbol` edge cases; the diagnose script's regex (tested — the
mis-capture concern was unfounded).

## Top recommended fix order

1. **C1** — hard-gate all engine WAIT reasons in the router (small, testable change;
   biggest money-loss risk).
2. **C2** — stop stamping reconcile on failure; re-protect positions whose close failed.
3. **C3** — fix `get_fetch_days` for 15Min/30Min/1Hour/1Day (+ tests).
4. **C4** — decide the liquidity basis (SIP vs documented IEX-scaled threshold) and set
   `MIN_DOLLAR_VOLUME` accordingly; the current 300k was calibrated to a feed artifact.
5. **H2/H3** — fix `parse_bool` empty-string handling and bring `.env.example` in line
   with validated defaults (config-drift test).
6. **H1** — count working entry orders toward all caps.
7. **C5, H4-H9, M-series** as follow-ups, each behind the repo's reversible-flag
   discipline where behavior changes.

## Missing-test priorities

1. Service-level "WAIT never trades": for every engine veto reason assert
   `evaluate_symbol` → `go_long=False` (would have caught C1).
2. `get_fetch_days`/`build_fetch_window` parameterized over all timeframes (C3, H-channel).
3. Flatten-failure paths: close fails after cancel → position re-protected, reconcile
   not stamped (C2); fractional qty flattened (M8).
4. Cap enforcement with resting unfilled entries (H1); `parse_bool("") == default` (H2);
   `.env.example` values equal code defaults (H3).
