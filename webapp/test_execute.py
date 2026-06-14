import pandas as pd
from execute import execute_plan


def _df():
    return pd.DataFrame(
        {
            "category": ["a", "a", "b", "b", "b"],
            "sales": [10, 20, 30, 40, 50],
        }
    )


def test_group_mean():
    plan = {"aggregation": "mean", "metric": "sales", "group_by": "category"}
    result, err = execute_plan(_df(), plan)
    assert err is None
    assert result["a"] == 15.0
    assert result["b"] == 40.0


def test_group_sum_with_ranking():
    plan = {
        "aggregation": "sum",
        "metric": "sales",
        "group_by": "category",
        "ranking": {"order": "desc", "top_n": 1},
    }
    result, err = execute_plan(_df(), plan)
    assert err is None
    assert list(result.index) == ["b"]
    assert result["b"] == 120


def test_filter_then_count():
    plan = {
        "filter": {"column": "category", "value": "b"},
        "aggregation": "count",
    }
    result, err = execute_plan(_df(), plan)
    assert err is None
    assert result == 3


def test_missing_filter_column_errors():
    plan = {"filter": {"column": "nope", "value": "x"}}
    result, err = execute_plan(_df(), plan)
    assert result is None
    assert "not found" in err.lower()


def test_numeric_filter_value_is_coerced():
    # The AI sends filter values as strings; a numeric column must still match.
    df = pd.DataFrame({"flag": [0, 1, 1, 1], "amount": [5, 10, 15, 20]})
    plan = {
        "filter": {"column": "flag", "value": "1"},
        "aggregation": "sum",
        "metric": "amount",
    }
    result, err = execute_plan(df, plan)
    assert err is None
    assert result == 45  # 10 + 15 + 20, not 0
