"""Guard: a stale high-water mark carried across a paper->live account switch must not
fake a catastrophic drawdown and pause the bot."""
import json, tempfile, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversion_bot.portfolio import PortfolioState

TODAY = datetime.now(ZoneInfo("America/Chicago")).date().isoformat()


def _seeded(state, peak, session_peak):
    d = os.path.join(state, "portfolio")
    os.makedirs(d, exist_ok=True)
    json.dump({"peak_equity": peak, "session_date": TODAY, "session_peak_equity": session_peak,
               "last_equity": session_peak, "daily_new_positions": [],
               "last_trade_ts_by_symbol": {}, "position_meta": {}},
              open(os.path.join(d, "portfolio_state.json"), "w"))
    return PortfolioState(state_dir=d)


def test_stale_paper_peak_is_reset():
    # paper peak $98,595 carried into a live $13,340 account, same day
    with tempfile.TemporaryDirectory() as t:
        ps = _seeded(t, 115325.93, 98595.49)
        ps.update_equity(13340.27)
        dd = ps.get_drawdown_pct(13340.27)
        data = ps._load()
        assert data["session_peak_equity"] == 13340.27, f"session_peak not reset: {data['session_peak_equity']}"
        assert data["peak_equity"] == 13340.27, f"stale all-time peak not reset: {data['peak_equity']}"
        assert dd == 0.0, f"false drawdown survived: {dd:.2%}"


def test_real_small_drawdown_not_touched():
    # a legit 2% intraday slide must be preserved (guard must not fire)
    with tempfile.TemporaryDirectory() as t:
        ps = _seeded(t, 13340.0, 13340.0)
        ps.update_equity(13073.0)          # -2.0% vs session peak
        dd = ps.get_drawdown_pct(13073.0)
        data = ps._load()
        assert data["session_peak_equity"] == 13340.0, "guard wrongly reset a real drawdown peak"
        assert abs(dd - 0.02) < 0.001, f"real drawdown mismeasured: {dd:.3%}"


if __name__ == "__main__":
    import traceback
    p = 0
    for n in [x for x in dir() if x.startswith("test_")]:
        try:
            globals()[n](); print(f"PASS {n}"); p += 1
        except AssertionError as e:
            print(f"FAIL {n}: {e}"); traceback.print_exc()
    print(f"\n{p} passed")
