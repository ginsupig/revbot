"""ALPACA_DATA_FEED selection for all bar fetches (audit C4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

alpaca = pytest.importorskip("alpaca")  # run_real_backtest imports alpaca-py

from run_real_backtest import _data_feed


def test_default_is_iex(monkeypatch):
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    assert _data_feed() == "iex"


def test_sip_respected(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "SIP")
    assert _data_feed() == "sip"


def test_garbage_falls_back_to_iex(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "bloomberg")
    assert _data_feed() == "iex"
