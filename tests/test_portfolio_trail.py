import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.portfolio import PortfolioState


def _ps(d):
    return PortfolioState(state_dir=str(Path(d) / "pf"))


def test_trail_state_stored_at_entry_and_seeds_high_water():
    with tempfile.TemporaryDirectory() as d:
        ps = _ps(d)
        ps.note_new_position("WDC", "mean_reversion", "reversion", "2026-06-14T14:30:00Z",
                             side="long", entry_price=100.0, stop_distance=1.2)
        state = ps.get_trail_state("WDC")
        assert state == (100.0, 1.2, 100.0)   # high_water seeds at entry


def test_update_high_water_only_ratchets_up():
    with tempfile.TemporaryDirectory() as d:
        ps = _ps(d)
        ps.note_new_position("WDC", "mean_reversion", "reversion", "2026-06-14T14:30:00Z",
                             side="long", entry_price=100.0, stop_distance=1.2)
        assert ps.update_high_water("WDC", 103.0) == 103.0
        assert ps.update_high_water("WDC", 101.0) == 103.0   # never lowers
        assert ps.update_high_water("WDC", 105.0) == 105.0
        assert ps.get_trail_state("WDC")[2] == 105.0


def test_no_trail_state_for_short_or_untracked():
    with tempfile.TemporaryDirectory() as d:
        ps = _ps(d)
        # Short: no entry_price/stop_distance passed -> no trail state.
        ps.note_new_position("TSLA", "mean_reversion", "reversion", "2026-06-14T14:30:00Z",
                             side="short")
        assert ps.get_trail_state("TSLA") is None
        assert ps.update_high_water("TSLA", 200.0) is None
        # Unknown symbol.
        assert ps.get_trail_state("NVDA") is None


def test_update_via_candidate_pulls_plan_for_long_only():
    with tempfile.TemporaryDirectory() as d:
        ps = _ps(d)
        ps.update({"symbol": "ARM", "entry_style": "mean_reversion", "regime": "reversion",
                   "position_plan": {"entry_price": 50.0, "risk_per_share": 0.6}})
        assert ps.get_trail_state("ARM") == (50.0, 0.6, 50.0)
        # go_short candidate -> long-only trail state is skipped.
        ps.update({"symbol": "QQQ", "entry_style": "mean_reversion", "regime": "reversion",
                   "go_short": True, "position_plan": {"entry_price": 400.0, "risk_per_share": 2.0}})
        assert ps.get_trail_state("QQQ") is None
