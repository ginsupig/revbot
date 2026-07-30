import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.gating import gate_decision


def _call(mode, comps, weighted, entry_style="mean_reversion",
          signal="LONG_REVERSION", router="engine_validated_reversion", threshold=0.36):
    return gate_decision(
        component_scores=comps, weighted_score=weighted, entry_style=entry_style,
        decision_signal=signal, router_reason=router, threshold=threshold, mode=mode,
    )


def test_blended_mode_gates_on_the_blend():
    comps = {"mean_reversion": 0.30, "ml": 0.9, "trendfail": 0.45, "trend_following": 0.3}
    # Blend clears 0.36 even though mean_reversion alone (0.30) would not.
    assert _call("blended", comps, weighted=0.50) == (True, False)


def test_routed_mode_rejects_when_the_routed_component_is_weak():
    # The decontamination case: a strong blend (carried by ml/trend) but a weak
    # mean-reversion component. Blended fires; routed correctly does NOT.
    comps = {"mean_reversion": 0.30, "ml": 0.9, "trendfail": 0.45, "trend_following": 0.3}
    assert _call("blended", comps, weighted=0.50)[0] is True
    assert _call("routed", comps, weighted=0.50)[0] is False   # mr 0.30 < 0.36


def test_routed_mode_fires_on_a_strong_routed_component():
    comps = {"mean_reversion": 0.70, "ml": 0.2, "trendfail": 0.45, "trend_following": 0.3}
    # Blend might be middling, but the routed mean_reversion component clears.
    assert _call("routed", comps, weighted=0.40)[0] is True


def test_short_requires_short_signal_and_mr_floor():
    comps = {"mean_reversion": 0.50, "ml": 0.5, "trendfail": 0.45, "trend_following": 0.3}
    go_long, go_short = gate_decision(
        component_scores=comps, weighted_score=0.50, entry_style="mean_reversion",
        decision_signal="SHORT_REVERSION", router_reason="engine_validated_short_reversion",
        threshold=0.36, mode="blended",
    )
    assert (go_long, go_short) == (False, True)


def test_threshold_is_the_sole_score_authority_not_router_reason():
    """A loosened (adaptive) threshold must be able to admit an entry the
    BASELINE router called score_below_threshold.

    The gate used to AND in `router_reason != "score_below_threshold"`, which
    hard-blocked at the baseline even when the adaptive threshold had dropped
    below it — mirroring a live behavior that audit M1 removed from the service,
    so no A/B could reproduce a loosened-threshold entry.
    """
    comps = {"mean_reversion": 0.9, "ml": 0.9, "trendfail": 0.45, "trend_following": 0.3}
    assert _call("blended", comps, weighted=0.9, router="score_below_threshold") == (True, False)


def test_mean_reversion_long_requires_engine_validation():
    """Live refuses an mr-routed LONG on a WAIT decision
    (router_reason=mr_requires_engine_signal); the gate must too, or backtests
    count entries the bot would never take."""
    comps = {"mean_reversion": 0.70, "ml": 0.2, "trendfail": 0.45, "trend_following": 0.3}
    go_long, _ = _call("routed", comps, weighted=0.40, signal="WAIT",
                       router="best_style_edge_mean_reversion")
    assert go_long is False
    # Trend styles legitimately trade on WAIT — the engine only validates reversion.
    comps_tf = {"mean_reversion": 0.3, "ml": 0.5, "trendfail": 0.45, "trend_following": 0.80}
    assert _call("routed", comps_tf, weighted=0.6, entry_style="trend_following",
                 signal="WAIT", router="trend_regime_alignment")[0] is True


def test_trend_following_downgrade_applies_in_both_modes():
    comps = {"mean_reversion": 0.3, "ml": 0.5, "trendfail": 0.45, "trend_following": 0.50}
    for mode in ("blended", "routed"):
        go_long, _ = _call(mode, comps, weighted=0.6, entry_style="trend_following",
                           router="trend_regime_alignment")
        assert go_long is False   # trend_following component 0.50 < 0.55
