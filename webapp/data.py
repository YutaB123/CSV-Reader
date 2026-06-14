"""Data layer: parse CSVs and produce a schema+stats summary for the AI.

The summary returned by `summarize` is the ONLY thing ever sent to Claude.
It contains column names, dtypes, numeric means, and categorical top-value
counts — never raw rows.
"""

import pandas as pd

TOP_K = 30


def load_csv(file) -> pd.DataFrame:
    """Parse an uploaded CSV file object into a DataFrame."""
    return pd.read_csv(file)


def summarize(df: pd.DataFrame) -> dict:
    """Return schema + summary statistics. No raw rows are included."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card_numeric = [
        c for c in numeric_cols if df[c].nunique(dropna=True) <= 20
    ]
    categorical_cols = list(dict.fromkeys(cat_cols + low_card_numeric))

    overall_means = {}
    if numeric_cols:
        means = df[numeric_cols].mean(numeric_only=True)
        overall_means = {
            c: (float(means[c]) if pd.notna(means[c]) else None)
            for c in numeric_cols
        }

    group_counts = {}
    for cat in categorical_cols:
        vc = df[cat].astype("string").value_counts(dropna=True).head(TOP_K)
        group_counts[cat] = {str(idx): int(cnt) for idx, cnt in vc.to_dict().items()}

    return {
        "meta": {
            "total_rows": int(len(df)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        "overall_means": overall_means,
        "group_counts": group_counts,
    }
