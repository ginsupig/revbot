"""Sweep/execution engine adapters. Each produces per-trade returns the
orchestrator (research.pipeline.evaluate_sweep) gates.

  reference     : pure-numpy flush-bounce sweep (testable, the canonical adapter)
  vectorbt_sweep: fast vectorized sweep TEMPLATE (run where vectorbt is installed)
  backtrader_exec: realistic-fill TEMPLATE (run where backtrader is installed)
"""
from .reference import flush_bounce_signals, sweep, EXIT

__all__ = ["flush_bounce_signals", "sweep", "EXIT"]
