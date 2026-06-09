import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.sectors import sector_for, etf_for_sector, etf_for_symbol


def test_known_symbols_map_to_sector_and_etf():
    assert sector_for("NVDA") == "semis_ai"
    assert sector_for("amd") == "semis_ai"          # case-insensitive
    assert etf_for_sector("semis_ai") == "SMH"
    assert etf_for_symbol("MU") == "SMH"
    assert etf_for_symbol("JPM") == "XLF"
    assert etf_for_symbol("OXY") == "XLE"
    assert etf_for_symbol("CLF") == "XLB"
    assert etf_for_symbol("LLY") == "XLV"
    assert etf_for_symbol("MSFT") == "XLK"


def test_unclassified_symbol_returns_none():
    assert sector_for("ASTS") is None               # space — intentionally unmapped
    assert sector_for("") is None
    assert etf_for_symbol("ZZZZ") is None


def test_etf_for_missing_sector_is_none():
    assert etf_for_sector(None) is None
    assert etf_for_sector("not_a_sector") is None
