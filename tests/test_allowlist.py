from reversion_bot.allowlist import (
    select_allowlist,
    parse_allowlist,
    filter_symbols,
    parse_symbol_csv,
    apply_watchlist,
)


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


# --- parse_symbol_csv -------------------------------------------------------

def test_parse_symbol_csv_blank_is_empty_set():
    assert parse_symbol_csv(None) == set()
    assert parse_symbol_csv("") == set()
    assert parse_symbol_csv("  ,  ") == set()


def test_parse_symbol_csv_uppercases_and_strips():
    assert parse_symbol_csv(" poet, avgo ,Pltr") == {"POET", "AVGO", "PLTR"}


# --- apply_watchlist --------------------------------------------------------

def test_apply_watchlist_force_includes_missing_names():
    out = apply_watchlist(["MU", "NVDA"], include={"POET", "AVGO"})
    assert out[:2] == ["MU", "NVDA"]
    assert set(out) == {"MU", "NVDA", "POET", "AVGO"}


def test_apply_watchlist_excludes_chronic_losers():
    out = apply_watchlist(["MU", "TSLA", "AMZN"], exclude={"TSLA", "AMZN"})
    assert out == ["MU"]


def test_apply_watchlist_exclude_wins_over_include():
    out = apply_watchlist(["MU"], include={"AMZN"}, exclude={"AMZN"})
    assert out == ["MU"]


def test_apply_watchlist_dedups_case_insensitively():
    out = apply_watchlist(["MU", "mu"], include={"MU"})
    assert out == ["MU"]


def test_apply_watchlist_roster_change_keeps_pl():
    # The real change: add 5, drop the 5 biggest losers but keep PL.
    universe = ["TSLA", "MU", "WDC", "AMZN", "SOFI", "MP", "PL", "ASTS", "NVDA"]
    out = apply_watchlist(
        universe,
        include={"POET", "AVGO", "RXT", "PLTR", "QBTS"},
        exclude={"AMZN", "TSLA", "WDC", "MP", "SOFI"},
    )
    assert "PL" in out                    # kept on purpose
    assert not ({"AMZN", "TSLA", "WDC", "MP", "SOFI"} & set(out))
    assert {"POET", "AVGO", "RXT", "PLTR", "QBTS"} <= set(out)
