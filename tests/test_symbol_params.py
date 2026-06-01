import json

from reversion_bot.config import ReversionConfig
from reversion_bot.symbol_params import (
    load_symbol_params,
    config_for_symbol,
    build_symbol_configs,
)


# --- load_symbol_params -----------------------------------------------------

def test_load_missing_file_returns_empty(tmp_path):
    assert load_symbol_params(str(tmp_path / "nope.json")) == {}


def test_load_bad_json_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert load_symbol_params(str(p)) == {}


def test_load_uppercases_symbols_and_drops_nondict(tmp_path):
    p = tmp_path / "sp.json"
    p.write_text(json.dumps({"tqqq": {"ri_threshold": -1.0}, "bad": 5}))
    out = load_symbol_params(str(p))
    assert out == {"TQQQ": {"ri_threshold": -1.0}}


# --- config_for_symbol ------------------------------------------------------

def test_override_applied_and_coerced():
    base = ReversionConfig()
    cfg = config_for_symbol(base, {
        "ri_threshold": "0.0",          # str -> float
        "rsi_max": 48,                  # int -> float
        "band_length": 20.0,            # float -> int
        "use_vwap_filter": "true",      # str -> bool
        "max_vwap_extension_pct": 0.02,
    })
    assert cfg.ri_threshold == 0.0
    assert cfg.rsi_max == 48.0
    assert cfg.band_length == 20 and isinstance(cfg.band_length, int)
    assert cfg.use_vwap_filter is True
    assert cfg.max_vwap_extension_pct == 0.02


def test_unknown_and_uncoercible_keys_ignored():
    base = ReversionConfig()
    cfg = config_for_symbol(base, {"not_a_field": 1, "rsi_max": "abc"})
    # base is unchanged; nothing crashes
    assert cfg.rsi_max == base.rsi_max
    assert not hasattr(cfg, "not_a_field")


def test_base_untouched_for_unspecified_fields():
    base = ReversionConfig()
    cfg = config_for_symbol(base, {"ri_threshold": -1.0})
    assert cfg.ri_threshold == -1.0
    assert cfg.adx_max == base.adx_max          # untouched
    assert cfg.min_history == base.min_history


# --- build_symbol_configs ---------------------------------------------------

def test_build_maps_each_symbol_to_its_config():
    base = ReversionConfig()
    cfgs = build_symbol_configs(
        {"TQQQ": {"ri_threshold": -1.0}, "QQQ": {"ri_threshold": 0.0}}, base
    )
    assert cfgs["TQQQ"].ri_threshold == -1.0
    assert cfgs["QQQ"].ri_threshold == 0.0


def test_build_empty_map_is_empty():
    assert build_symbol_configs({}, ReversionConfig()) == {}
