import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.portfolio import PortfolioState


def test_intraday_drawdown_from_session_peak(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    ps.update_equity(100000, today="2026-06-03")
    ps.update_equity(105000, today="2026-06-03")   # session peak -> 105k
    ps.update_equity(102000, today="2026-06-03")
    # Measured from the day's high-water mark, so giving back gains counts.
    dd = ps.get_drawdown_pct(102000, today="2026-06-03")
    assert abs(dd - (105000 - 102000) / 105000) < 1e-9


def test_session_peak_ratchets_up_no_drawdown(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    ps.update_equity(100000, today="2026-06-03")
    ps.update_equity(110000, today="2026-06-03")
    assert ps.get_drawdown_pct(110000, today="2026-06-03") == 0.0


def test_new_day_resets_drawdown(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    ps.update_equity(100000, today="2026-06-02")
    ps.update_equity(96000, today="2026-06-02")           # -4% intraday -> breaker would fire
    assert ps.get_drawdown_pct(96000, today="2026-06-02") > 0.025

    # Next morning starts flat even though equity is unchanged.
    assert ps.get_drawdown_pct(96000, today="2026-06-03") == 0.0
    ps.update_equity(96000, today="2026-06-03")
    assert ps.get_drawdown_pct(96000, today="2026-06-03") == 0.0


def test_all_time_peak_does_not_brick_later_days(tmp_path):
    # The old bug: a high all-time peak permanently paused trading. The session
    # anchor must ignore it — a flat day below the old peak shows zero drawdown.
    ps = PortfolioState(state_dir=str(tmp_path))
    ps.update_equity(130000, today="2026-06-01")   # all-time high
    assert ps.get_drawdown_pct(124000, today="2026-06-03") == 0.0
    ps.update_equity(124000, today="2026-06-03")
    assert ps.get_drawdown_pct(124000, today="2026-06-03") == 0.0
    # All-time peak is still recorded for reporting, just not used by the breaker.
    assert ps._load()["peak_equity"] == 130000


def test_breaker_fires_within_a_single_day(tmp_path):
    ps = PortfolioState(state_dir=str(tmp_path))
    ps.update_equity(125000, today="2026-06-03")           # session peak
    ps.update_equity(121000, today="2026-06-03")           # -3.2% from peak
    assert ps.get_drawdown_pct(121000, today="2026-06-03") > 0.025
