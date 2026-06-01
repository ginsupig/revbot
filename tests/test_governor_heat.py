import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.governor import ExecutionGovernor


@dataclass
class FakePosition:
    symbol: str
    unrealized_pl: float


def test_only_losses_count_as_heat():
    # Mirror the live account that triggered this fix.
    positions = [
        FakePosition("ASTS", -2405.76),
        FakePosition("MU", 1453.40),    # winner -> 0 heat
        FakePosition("WDC", 651.50),    # winner -> 0 heat
    ]
    heat = ExecutionGovernor.total_drawdown_heat(positions)
    assert abs(heat - 2405.76) < 1e-6   # only the loser counts


def test_all_winners_zero_heat():
    positions = [FakePosition("A", 100.0), FakePosition("B", 5.0)]
    assert ExecutionGovernor.total_drawdown_heat(positions) == 0.0


def test_multiple_losses_sum():
    positions = [FakePosition("A", -100.0), FakePosition("B", -50.0), FakePosition("C", 200.0)]
    assert ExecutionGovernor.total_drawdown_heat(positions) == 150.0


def test_empty_book():
    assert ExecutionGovernor.total_drawdown_heat([]) == 0.0


def test_winners_no_longer_block_entry():
    # $126,716 equity, 2% cap = $2,534. Old abs() heat was $4,510 (blocked).
    # Loss-only heat is just ASTS's $2,405.76, leaving headroom.
    positions = [
        FakePosition("ASTS", -2405.76),
        FakePosition("MU", 1453.40),
        FakePosition("WDC", 651.50),
    ]
    equity = 126716.38
    cap = equity * 0.02
    heat = ExecutionGovernor.total_drawdown_heat(positions)
    assert heat < cap        # would have been blocked under the old abs() logic
