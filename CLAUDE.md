# CLAUDE.md

Guidance for Claude working in this repo (`revbot` — an Alpaca mean-reversion
trading bot plus a battery of research backtests).

## Working principle: exhaust the fix before admitting defeat

**Do not dismiss an alternate strategy or a backtest result without first
understanding *why* it failed and trying a better solution to that failure.**
Give every idea our best effort to succeed before calling it dead.

When a backtest disappoints:
1. **Diagnose the cause, don't just report the number.** Is it the signal, the
   exit, the entry timing, the universe, position/occupancy accounting, cost
   assumptions, sample size, data quality, or the regime? Find the *mechanism*.
2. **Try the obvious better version first.** A weak result often means the idea
   was implemented naively, not that it's wrong — e.g. "buy at the close" is a
   wash, but *conditioning* the entry on the day's action is a different test;
   a fixed target caps winners where a trailing stop lets them run; an
   occupancy-free sim oversamples clustered losers vs the live one-position bot.
3. **Only then call it.** Conclude "no edge" after the better versions have been
   tried and also fail — and say specifically what was tried and why it failed.

This cuts both ways: be just as rigorous about *not* shipping a false positive
(see the discipline below) as about not abandoning a real idea too early.

## Research discipline (how we judge a backtest)

These are the standards that separate a real edge from a mirage. Hold to them:

- **Out-of-sample is mandatory.** A result must hold on a second, non-overlapping
  window, not just in-sample. Sign-flips across windows = regime-dependent, not an
  edge. (Killed exhaustion, swing reversion, and the n=14 volume mirage this way.)
- **Sample size.** Discount thin-N cells hard — a PF of 3 at N=2 is noise. Want
  N in the hundreds before believing a per-name or per-feature result.
- **Survivorship / selection control.** Test on a *neutral* universe (large caps
  incl. laggards), not just today's curated winners — momentum especially flatters
  a hindsight-selected universe.
- **Cost realism.** Sweep round-trip cost (bps). A thin gross edge eaten by cost
  isn't tradeable; daily-turnover strategies are cost-fragile.
- **Occupancy realism.** The live bot holds one position per symbol with cooldowns;
  an occupancy-free per-signal sim oversamples clustered (often losing) re-entries.
- **Regime over signal.** Recurring finding: regime dominates. Momentum pays when
  trending, reversion when ranging; neither persists across both. Don't fight the
  tape with a static signal.

## Orientation

- **Bot**: `main.py` (live/paper loop) drives `reversion_bot/` — `engine.py`
  (LONG_REVERSION signal), `risk.py`/`config.py` (ATR brackets), `execution.py`
  (Alpaca orders), `governor.py`/`portfolio.py` (sizing, cooldowns, caps),
  `market_regime.py` (SPY risk-off gate), `trailing.py` (trailing stop).
- **Validated, live defaults**: long-only, per-symbol trend filter, loss-aware
  re-entry brake, market-regime gate, trailing stop (TP 3.5 + 1.5 ATR), carryover
  guard (entry cutoff + EOD flatten). Toggle via env (see `main.py`).
- **Reports**: `pnl_report.py` (`--daily`, `--scoreboard`) pulls realized PnL from
  Alpaca; account is selected by the `.env` API keys (live vs paper URL must match).
- **Research backtests** (root, `*_backtest.py` / `*_report.py`): each fetches
  daily/intraday bars via `run_real_backtest.fetch_alpaca_bars` (needs creds),
  runs the real engine, and reports per the discipline above. Follow their pattern
  (two windows, cost sweep, OOS) for new hypotheses.

## Conventions

- **Tests**: `python -m pytest tests/ -q`. Pure logic is unit-tested; broker glue
  uses a fake client / `importorskip`. Some modules need optional deps (sklearn,
  alpaca) absent in sandboxes — CI has them.
- **Research scripts are root-level and not wired into the live path.** Keep them
  that way until a result earns deployment under the discipline above.
- Commit live-trading/risk changes behind reversible env flags, defaulted to the
  validated behavior.
