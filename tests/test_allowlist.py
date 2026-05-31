from reversion_bot.allowlist import select_allowlist, parse_allowlist, filter_symbols


# --- select_allowlist -------------------------------------------------------

def test_select_keeps_only_names_clearing_both_thresholds():
    scores = {
        "TQQQ": {"profit_factor": 2.53, "sharpe": 0.91},   # clears
        "AAPU": {"profit_factor": 1.10, "sharpe": 0.08},   # clears (borderline)
        "TSLL": {"profit_factor": 1.04, "sharpe": -0.06},  # PF ok, Sharpe fails
        "AMZU": {"profit_factor": 0.59, "sharpe": -0.69},  # both fail
    }
    assert select_allowlist(scores, min_pf=1.10, min_sharpe=0.0) == ["TQQQ", "AAPU"]


def test_select_sharpe_floor_rejects_marginal_pf():
    scores = {"X": {"profit_factor": 1.5, "sharpe": -0.2}}
    assert select_allowlist(scores, min_pf=1.10, min_sharpe=0.0) == []


def test_select_missing_metrics_default_to_zero_and_drop():
    assert select_allowlist({"X": {}}, min_pf=1.10) == []


def test_select_empty_scores_returns_empty():
    assert select_allowlist({}, min_pf=1.10) == []


# --- parse_allowlist --------------------------------------------------------

def test_parse_none_disables_gating():
    assert parse_allowlist(None) is None


def test_parse_empty_string_is_empty_set_not_none():
    # Explicitly empty => gate is on but nothing qualifies (trade nothing).
    assert parse_allowlist("") == set()


def test_parse_uppercases_and_strips():
    assert parse_allowlist(" tqqq , tecl ,, ") == {"TQQQ", "TECL"}


# --- filter_symbols ---------------------------------------------------------

def test_filter_none_allowlist_keeps_all():
    kept, dropped = filter_symbols(["TQQQ", "AMZU"], None)
    assert kept == ["TQQQ", "AMZU"]
    assert dropped == []


def test_filter_empty_allowlist_drops_all():
    kept, dropped = filter_symbols(["TQQQ", "AMZU"], set())
    assert kept == []
    assert dropped == ["TQQQ", "AMZU"]


def test_filter_partitions_and_preserves_original_case():
    kept, dropped = filter_symbols(["tqqq", "AMZU", "Tecl"], {"TQQQ", "TECL"})
    assert kept == ["tqqq", "Tecl"]
    assert dropped == ["AMZU"]
