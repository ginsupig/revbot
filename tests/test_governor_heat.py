import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.governor import ExecutionGovernor
from reversion_bot.portfolio import PortfolioState


@dataclass
class FakePosition:
    symbol: str
    unrealized_pl: float
    qty: float = 10.0


def _governor(tmp_path, meta=None):
    ps = PortfolioState(state_dir=str(tmp_path / "portfolio"))
    if meta:
        data = ps._load()
        data["position_meta"] = meta
        ps._save(data)
    return ExecutionGovernor(portfolio_state=ps)


def test_risk_heat_uses_stop_distance_from_meta(tmp_path):
    meta = {
        "ASTS": {"stop_distance": 2.50},
        "MU": {"stop_distance": 1.00},
    }
    gov = _governor(tmp_path, meta)
    positions = [
        FakePosition("ASTS", -2405.76, qty=100),
        FakePosition("MU", 1453.40, qty=50),   # winner still contributes risk
    ]
    heat = gov.total_risk_heat(positions)
    # ASTS: 100 * 2.50 = 250, MU: 50 * 1.00 = 50 => total 300
    assert abs(heat - 300.0) < 1e-6


def test_fallback_to_drawdown_when_no_meta(tmp_path):
    gov = _governor(tmp_path)
    positions = [
        FakePosition("ASTS", -2405.76),
        FakePosition("MU", 1453.40),    # winner -> 0 heat (fallback)
    ]
    heat = gov.total_risk_heat(positions)
    assert abs(heat - 2405.76) < 1e-6   # only the loser counts in fallback


def test_all_winners_zero_heat_fallback(tmp_path):
    gov = _governor(tmp_path)
    positions = [FakePosition("A", 100.0), FakePosition("B", 5.0)]
    assert gov.total_risk_heat(positions) == 0.0


def test_empty_book(tmp_path):
    gov = _governor(tmp_path)
    assert gov.total_risk_heat([]) == 0.0


def test_winners_with_meta_still_contribute_risk(tmp_path):
    # $126,716 equity, 2% cap = $2,534. With risk-at-stop, even winning
    # positions contribute heat, making the gate more conservative.
    meta = {
        "ASTS": {"stop_distance": 3.00},
        "MU": {"stop_distance": 1.50},
        "WDC": {"stop_distance": 1.00},
    }
    gov = _governor(tmp_path, meta)
    positions = [
        FakePosition("ASTS", -2405.76, qty=100),  # 100 * 3.00 = 300
        FakePosition("MU", 1453.40, qty=50),       # 50 * 1.50 = 75
        FakePosition("WDC", 651.50, qty=80),        # 80 * 1.00 = 80
    ]
    heat = gov.total_risk_heat(positions)
    assert abs(heat - 455.0) < 1e-6
