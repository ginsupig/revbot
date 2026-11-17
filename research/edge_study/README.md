# Entry edge study (2026-11-16)

Question: does Revbot's long entry beat a random entry in the same names?

Run from the repo root, in order:

    python research/edge_study/edge.py      # builds signals.pkl from state/regime_ab/replay_2024-01-01_2026-11-14.pkl
    python research/edge_study/analyze.py   # signals vs random entry; trades sliced by symbol, year, hour, RSI, ...
    python research/edge_study/variants.py  # afternoon / CLF filters; stops x1, x2, x3

The replay comes from `research/regime_gate_ab.py` (the live service's own
signals), 10 names, 2024-01-01 to 2026-11-14, 8 bps round trip.

Findings:

- Gross forward return after a signal the market filter lets through
  (n=1,122): -0.8 bps at +60 min, +2.1 bps to the close. Random entry in the
  same names and hours: +0.6 and +1.6. No measurable edge before costs.
- Net: 1,034 trades, PF 0.81, -$2,216. By year: PF 1.02, 0.77, 0.68.
- Entries from 13:00 ET on: 493 trades, PF 0.63, -$2,108, negative every year.
- CLF: 117 trades, PF 0.41, -$1,027, negative every year.
- Before 13:00 ET and without CLF: 487 trades, PF 1.06, +$288
  (2024 +$649, 2025 -$98, 2026 -$263).
- No RSI, z-score, day-return, gap, SPY-return, volume, stop-size or
  reward/risk bucket is positive in all three years.
- Widening the stop x2 or x3 leaves PF at 0.81.

Applied in the fleet (C:\MegaEmpire\config\paper_runtimes.json, revbot env):
`EOD_ENTRY_CUTOFF_MINUTES=180` and CLF removed from `TRADE_WATCHLIST` and
`TRADE_ALLOWLIST`. This reduces the loss; it does not create an edge.

Limits: the slice numbers filter existing trades and are not re-simulated, so
they ignore the slots the removed trades would have freed. The replay has no
spread filter.
