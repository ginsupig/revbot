import os

from reversion_bot.heartbeat import (
    write_heartbeat,
    read_heartbeat,
    heartbeat_age_seconds,
    is_alive,
)


# --- write / read round trip ------------------------------------------------

def test_write_then_read_round_trip(tmp_path):
    p = str(tmp_path / "sub" / "heartbeat.json")   # parent dir created on write
    payload = write_heartbeat(p, {"cycle": 7, "market_open": True})
    assert os.path.exists(p)
    hb = read_heartbeat(p)
    assert hb["cycle"] == 7
    assert hb["market_open"] is True
    assert "ts_epoch" in hb and "pid" in hb
    assert payload["cycle"] == 7


def test_write_is_atomic_no_leftover_tmp(tmp_path):
    p = str(tmp_path / "heartbeat.json")
    write_heartbeat(p, {})
    assert not os.path.exists(p + ".tmp")


def test_read_missing_returns_none(tmp_path):
    assert read_heartbeat(str(tmp_path / "nope.json")) is None


# --- age / liveness (pure, clock-injected) ----------------------------------

def test_age_uses_injected_now():
    hb = {"ts_epoch": 1000.0}
    assert heartbeat_age_seconds(hb, now=1075.0) == 75.0


def test_age_none_for_missing_or_unparseable():
    assert heartbeat_age_seconds(None) is None
    assert heartbeat_age_seconds({}) is None
    assert heartbeat_age_seconds({"ts_epoch": "abc"}) is None


def test_age_never_negative():
    # clock skew shouldn't produce a negative age
    assert heartbeat_age_seconds({"ts_epoch": 2000.0}, now=1990.0) == 0.0


def test_is_alive_fresh_vs_stale():
    hb = {"ts_epoch": 1000.0}
    assert is_alive(hb, max_age_seconds=180, now=1100.0) is True     # 100s old
    assert is_alive(hb, max_age_seconds=180, now=1200.0) is False    # 200s old


def test_is_alive_false_when_missing():
    assert is_alive(None, max_age_seconds=180) is False
