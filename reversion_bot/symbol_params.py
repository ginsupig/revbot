"""Per-symbol strategy parameters.

``autotune_run.py`` writes a JSON map ``{SYMBOL: {param: value, ...}}`` holding
each allowlisted symbol's OWN tuned params. The live bot then trades every name
with its own config instead of one global median — so a winner whose optimum
differs from the crowd (e.g. EEM) keeps its edge, and the gate reflects exactly
what each symbol will trade with.

Pure (no network): load the map and turn it into ``ReversionConfig`` overrides.
Missing/garbage file -> empty map -> the bot falls back to the global .env
params, i.e. prior behavior. Fully backward compatible.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Dict

from .config import ReversionConfig

# Fields a per-symbol file may override on ReversionConfig, with type coercion.
_INT_FIELDS = {"band_length", "min_history"}
_BOOL_FIELDS = {"use_vwap_filter", "use_trend_filter", "require_reclaim_lb1"}
_FLOAT_FIELDS = {
    "ri_threshold", "rsi_max", "max_vwap_extension_pct",
    "band_std_1", "band_std_2", "adx_max",
}
_ALLOWED = _INT_FIELDS | _BOOL_FIELDS | _FLOAT_FIELDS


def load_symbol_params(path: str) -> Dict[str, dict]:
    """Load the per-symbol param map; return {} if missing/empty/unparseable."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}


def _coerce(field: str, value):
    if field in _BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if field in _INT_FIELDS:
        return int(round(float(value)))
    return float(value)


def config_for_symbol(base: ReversionConfig, overrides: dict) -> ReversionConfig:
    """Return a copy of ``base`` with the allowed, coerced overrides applied.

    Unknown/uncoercible keys are ignored so a stray field can't crash the bot.
    """
    clean = {}
    for key, value in (overrides or {}).items():
        if key in _ALLOWED:
            try:
                clean[key] = _coerce(key, value)
            except (TypeError, ValueError):
                continue
    return replace(base, **clean)


def build_symbol_configs(
    params_map: Dict[str, dict], base: ReversionConfig
) -> Dict[str, ReversionConfig]:
    """Map each symbol -> ReversionConfig (base overridden by its stored params)."""
    return {sym: config_for_symbol(base, ov) for sym, ov in params_map.items()}
