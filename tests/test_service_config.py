import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.service import ReversionService
from reversion_bot.config import PerformanceConfig


def _svc(tmp_path, **kwargs):
    # Isolate performance state per-test so log writes don't collide.
    return ReversionService(
        performance_config=PerformanceConfig(state_dir=str(tmp_path)),
        **kwargs,
    )


def test_min_trade_score_defaults_to_036(tmp_path):
    assert _svc(tmp_path).min_trade_score == 0.36


def test_min_trade_score_override_is_respected(tmp_path):
    # The value main.py reads from MIN_TRADE_SCORE must flow through and become
    # both the routing floor and the (pre-adaptive) trade threshold.
    assert _svc(tmp_path, min_trade_score=0.40).min_trade_score == 0.40
