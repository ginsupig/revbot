import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fees_report import summarize_fees


def test_sums_debits_as_fees_and_groups_by_type():
    activities = [
        {"activity_type": "REG", "net_amount": "-12.34"},
        {"activity_type": "TAF", "net_amount": "-3.10"},
        {"activity_type": "TAF", "net_amount": "-1.90"},
        {"activity_type": "DIV", "net_amount": "5.00"},   # a credit, not a fee
    ]
    report = summarize_fees(activities)
    assert report["total_fees"] == -17.34          # only the three debits
    assert report["by_type"]["TAF"] == {"count": 2, "total": -5.0}
    assert report["by_type"]["DIV"] == {"count": 1, "total": 5.0}


def test_empty_and_malformed_amounts_are_safe():
    report = summarize_fees([
        {"activity_type": "FEE", "net_amount": None},
        {"activity_type": "FEE", "net_amount": "not-a-number"},
    ])
    assert report["total_fees"] == 0.0
    assert report["by_type"]["FEE"]["count"] == 2


def test_no_activities():
    report = summarize_fees([])
    assert report == {"by_type": {}, "total_fees": 0.0}
