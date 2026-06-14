"""Execution layer: run a plan on the full DataFrame and render charts.

Provider-independent — uses only pandas and matplotlib. The AI produces plans;
this module executes them locally on the full data.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def execute_plan(df, plan: dict):
    """Apply a query plan to df. Returns (result, error_message)."""
    try:
        working = df.copy()

        flt = plan.get("filter")
        if flt:
            col = flt.get("column")
            val = flt.get("value")
            if col not in working.columns:
                return None, "Filter column not found"
            working = working[working[col] == val]

        group_by = plan.get("group_by")
        if group_by and isinstance(group_by, str):
            group_by = [group_by]

        aggregation = plan.get("aggregation")
        metric = plan.get("metric")
        result = None

        if aggregation:
            if group_by:
                grouped = working.groupby(group_by)
                if aggregation == "count":
                    result = (
                        grouped[metric].count()
                        if metric and metric in working.columns
                        else grouped.size()
                    )
                else:
                    if not metric or metric not in working.columns:
                        return None, "Metric missing for aggregation"
                    result = grouped[metric].agg(aggregation)
            else:
                if aggregation == "count":
                    result = len(working) if not metric else working[metric].count()
                else:
                    if not metric or metric not in working.columns:
                        return None, "Metric missing for aggregation"
                    result = getattr(working[metric], aggregation)()
        else:
            result = len(working)

        ranking = plan.get("ranking")
        if ranking and hasattr(result, "sort_values"):
            order = ranking.get("order", "desc")
            top_n = ranking.get("top_n", 10)
            result = result.sort_values(ascending=(order == "asc")).head(top_n)

        return result, None
    except Exception as e:  # noqa: BLE001 - surface any execution error to the UI
        return None, str(e)


def generate_graph(df, chart_type, x_col, y_col):
    """Render a matplotlib Figure for the given chart spec, or None on failure."""
    if df is None or not x_col or not y_col:
        return None
    if x_col not in df.columns or y_col not in df.columns:
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_y = df[y_col].dtype.kind in "if"

        if chart_type == "Bar Chart":
            if numeric_y:
                series = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
            else:
                series = df[x_col].value_counts()
            series.plot(kind="bar", ax=ax, color=list(plt.cm.tab20.colors)[: len(series)])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col}")
        elif chart_type == "Pie Chart":
            if numeric_y:
                series = df.groupby(x_col)[y_col].sum()
            else:
                series = df[x_col].value_counts()
            series.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"{y_col} distribution by {x_col}")
        elif chart_type == "Line Chart":
            if numeric_y:
                series = df.groupby(x_col)[y_col].mean()
            else:
                series = df[x_col].value_counts().sort_index()
            series.plot(kind="line", marker="o", ax=ax)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col}")
        elif chart_type == "Scatter Plot":
            if df[x_col].dtype.kind in "if" and numeric_y:
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
            else:
                plt.close(fig)
                return None
        else:
            plt.close(fig)
            return None

        fig.tight_layout()
        return fig
    except Exception:  # noqa: BLE001
        plt.close("all")
        return None
