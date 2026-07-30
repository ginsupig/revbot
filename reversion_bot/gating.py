"""Entry-gate decision — extracted so it can be A/B'd (blended vs routed).

The live service gates every entry on the *blended* weighted score (mean_reversion
0.30 + trend_following 0.30 + ml 0.25 + trendfail 0.15). The concern (review #4b):
for a routed mean-reversion setup, 55% of the gate is anti-correlated
trend/ml components, so opposed signals net toward the middle and the threshold
goes mushy.

This helper expresses the gate once, parameterized by ``mode``:
  - "blended" : the current behavior (gate on the weighted blend).
  - "routed"  : gate on the routed style's OWN component only.

Pure and dependency-free so it unit-tests in isolation and can be driven by the
A/B backtest without touching the live path.
"""
from __future__ import annotations

from typing import Mapping, Tuple

# entry_style values are exactly the component-score keys, so the routed
# component is component_scores[entry_style].


def gate_decision(
    *,
    component_scores: Mapping[str, float],
    weighted_score: float,
    entry_style: str,
    decision_signal: str,
    router_reason: str,
    threshold: float,
    mode: str = "routed",
    short_bias: bool = False,
) -> Tuple[bool, bool]:
    """Return (go_long, go_short) for the given gate ``mode``.

    Mirrors ReversionService.evaluate_symbol's gating exactly, except the scalar
    compared to ``threshold`` is either the blended score ("blended") or the
    routed style's own component ("routed").

    ``router_reason`` is accepted for call-site parity with the live service but
    is deliberately NOT part of the gate: the (possibly adaptive) threshold is
    the single authority (audit M1). Gating on it here would hard-block at the
    baseline even when the adaptive threshold had loosened below it, so no
    backtest could ever reproduce a loosened-threshold entry.
    """
    is_short = decision_signal == "SHORT_REVERSION"
    is_long = decision_signal == "LONG_REVERSION"

    # Default is "routed": gate on the primary component, not the blend.
    if mode == "blended":
        gate_score = float(weighted_score)
    else:
        gate_score = float(component_scores.get(entry_style, weighted_score))

    passes = gate_score >= threshold

    go_long = passes and not is_short

    # A mean-reversion LONG must be engine-validated: the mr component alone can
    # clear the threshold on a WAIT decision, which live refuses
    # (router_reason="mr_requires_engine_signal"). Trend/trendfail styles
    # legitimately trade on WAIT — the engine only validates reversion setups.
    if go_long and entry_style == "mean_reversion" and not is_long:
        go_long = False
    mr_floor = 0.40 if short_bias else 0.45
    go_short = (
        passes
        and is_short
        and entry_style == "mean_reversion"
        and float(component_scores.get("mean_reversion", 0.0)) >= mr_floor
    )

    # Style-specific minimum-conviction downgrades (independent of the gate mode).
    if entry_style == "trend_following" and float(component_scores.get("trend_following", 0.0)) < 0.55:
        go_long = False
    if entry_style == "trendfail" and float(component_scores.get("trendfail", 0.0)) < 0.60:
        go_long = False

    return go_long, go_short
