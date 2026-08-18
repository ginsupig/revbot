from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from main import attach_quote_spread, completed_bars_only
from reversion_bot.portfolio import PortfolioState


def test_incomplete_intraday_bar_is_dropped():
    bars = pd.DataFrame({
        "date": ["2026-08-13T15:55:00Z", "2026-08-13T15:59:00Z"],
        "close": [100, 101],
    })
    out = completed_bars_only(
        bars, "5Min", datetime(2026, 8, 13, 16, 2, tzinfo=timezone.utc)
    )
    assert list(out["close"]) == [100]


def test_quote_spread_is_attached_in_bps():
    bars = pd.DataFrame({"close": [100.0]})
    out = attach_quote_spread(bars, SimpleNamespace(bid_price=99.95, ask_price=100.05))
    assert round(out.iloc[-1]["spread_bps"], 6) == 10.0


def test_flip_side_survives_open_metadata_prune(tmp_path):
    ps = PortfolioState(str(tmp_path))
    now = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)
    ps.note_new_position("XYZ", "mean_reversion", "reversion", now.isoformat(), side="short")
    ps.prune_closed_positions([])
    assert ps.in_direction_flip_cooldown("XYZ", "long", now, 60)


def test_pending_symbol_metadata_is_preserved_when_passed_to_prune(tmp_path):
    ps = PortfolioState(str(tmp_path))
    now = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)
    ps.note_new_position("XYZ", "mean_reversion", "reversion", now.isoformat(), side="long")
    ps.prune_closed_positions({"XYZ"})
    assert "XYZ" in ps.get_all_position_metadata()
