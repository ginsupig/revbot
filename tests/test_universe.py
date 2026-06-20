import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.universe import (
    NEUTRAL_LARGECAP, broad_reversion_universe, select_base_universe,
)


def test_broad_universe_is_broad_and_includes_laggards():
    u = broad_reversion_universe()
    assert len(u) >= 80
    # survivorship control: laggards present, not just momentum winners
    for laggard in ("INTC", "T", "VZ", "PFE", "BA", "CLF", "F"):
        assert laggard in u
    # returns a fresh copy (mutating it must not corrupt the module constant)
    u.append("ZZZZ")
    assert "ZZZZ" not in NEUTRAL_LARGECAP


def test_no_duplicate_symbols():
    assert len(NEUTRAL_LARGECAP) == len(set(NEUTRAL_LARGECAP))


def test_select_base_universe_broad_overrides_scanner():
    out = select_base_universe(["FOO", "BAR"], ["FB1"], use_broad=True)
    assert out == broad_reversion_universe()


def test_select_base_universe_uses_scanner_then_fallback():
    assert select_base_universe(["FOO"], ["FB1"], use_broad=False) == ["FOO"]
    assert select_base_universe([], ["FB1", "FB2"], use_broad=False) == ["FB1", "FB2"]


def test_live_universe_matches_research_neutral_set():
    # the live basket and the research sector map must not drift apart.
    try:
        from research.realtime_scorer import NEUTRAL_SECTOR_OF
    except Exception:
        import pytest
        pytest.skip("research package import unavailable in this env")
    assert set(NEUTRAL_LARGECAP) == set(NEUTRAL_SECTOR_OF)
