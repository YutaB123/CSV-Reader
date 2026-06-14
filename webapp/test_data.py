import pandas as pd
from data import summarize


def _df():
    return pd.DataFrame(
        {
            "category": ["a", "a", "b", "b", "b"],
            "region": ["x", "y", "x", "y", "y"],
            "sales": [10, 20, 30, 40, 50],
        }
    )


def test_summarize_reports_row_count():
    s = summarize(_df())
    assert s["meta"]["total_rows"] == 5


def test_summarize_classifies_columns():
    s = summarize(_df())
    assert "sales" in s["meta"]["numeric_columns"]
    assert "category" in s["meta"]["categorical_columns"]
    assert "region" in s["meta"]["categorical_columns"]


def test_summarize_includes_overall_means():
    s = summarize(_df())
    assert s["overall_means"]["sales"] == 30.0


def test_summarize_includes_group_counts():
    s = summarize(_df())
    assert s["group_counts"]["category"]["b"] == 3


def test_summarize_excludes_raw_rows():
    s = summarize(_df())
    assert set(s.keys()) == {"meta", "overall_means", "group_counts"}
    assert "raw" not in s
    assert "records" not in s
