# Streamlit + Claude CSV Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Gradio CSV analyzer to a Streamlit app deployed on Streamlit Community Cloud, backed by Claude instead of OpenAI, sending only schema + summary stats to the model.

**Architecture:** Three modules under `webapp/` — `data.py` (pandas parsing + summary that is the only thing the AI sees), `ai.py` (Anthropic SDK calls returning structured plans), and `app.py` (Streamlit UI + local pandas/matplotlib execution). The AI returns *plans*; Python executes them on the full DataFrame locally.

**Tech Stack:** Python, Streamlit, pandas, matplotlib, anthropic SDK, pydantic, pytest.

> **IMPORTANT — folder name:** the app folder is `webapp/`, NOT `streamlit/`. A local package named `streamlit` would shadow the installed `streamlit` library and break `import streamlit as st`. Streamlit Cloud runs `webapp/app.py`, which puts `webapp/` on `sys.path`, so all intra-app imports are **flat** (`from data import ...`, `from execute import ...`, `import ai`). Tests use the same flat imports, enabled by an empty `webapp/conftest.py`.

---

## File Structure

All new files live under `webapp/` in the existing `CSV-Reader` repo. The original `csv_reader/csv_reader.py` is untouched.

- `webapp/data.py` — `load_csv()`, `summarize()`. No AI, no Streamlit. Pure pandas.
- `webapp/execute.py` — `execute_plan()`, `generate_graph()`. Pure pandas/matplotlib, provider-independent.
- `webapp/ai.py` — Anthropic SDK wrapper: `QueryPlan`, `ChartPlan` schemas + `plan_query()`, `explain()`, `plan_chart()`. Reads secrets.
- `webapp/app.py` — Streamlit UI wiring the above together.
- `webapp/conftest.py` — empty; makes pytest add `webapp/` to `sys.path` so tests use flat imports.
- `webapp/test_data.py` — unit tests for `summarize()`.
- `webapp/test_execute.py` — unit tests for `execute_plan()`.
- `requirements.txt` — at repo root, for Streamlit Cloud.
- `.streamlit/config.toml` — minimal theme/config (this dot-folder is Streamlit's real config dir and keeps its name).

---

## Task 1: Project scaffolding (requirements + config)

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `webapp/conftest.py` (empty, so pytest puts `webapp/` on sys.path for flat imports)

- [ ] **Step 1: Create `requirements.txt`**

```
streamlit>=1.40
pandas>=2.0
matplotlib>=3.8
anthropic>=0.40
pydantic>=2.0
pytest>=8.0
```

- [ ] **Step 2: Create `.streamlit/config.toml`**

```toml
[theme]
base = "dark"

[server]
maxUploadSize = 50
```

- [ ] **Step 3: Create empty `webapp/conftest.py`**

Create an empty file at `webapp/conftest.py`.

- [ ] **Step 4: Install dependencies locally**

Run: `python -m pip install -r requirements.txt`
Expected: installs without error (anthropic, streamlit, pandas, matplotlib, pydantic, pytest present).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .streamlit/config.toml webapp/conftest.py
git commit -m "chore: scaffold Streamlit app dependencies and config"
```

---

## Task 2: Data layer — `summarize()` (TDD)

**Files:**
- Create: `webapp/data.py`
- Test: `webapp/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `webapp/test_data.py`:

```python
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
    # low-cardinality + object columns are categorical
    assert "category" in s["meta"]["categorical_columns"]
    assert "region" in s["meta"]["categorical_columns"]


def test_summarize_includes_overall_means():
    s = summarize(_df())
    assert s["overall_means"]["sales"] == 30.0


def test_summarize_includes_group_counts():
    s = summarize(_df())
    assert s["group_counts"]["category"]["b"] == 3


def test_summarize_excludes_raw_rows():
    # The whole point: no raw row data leaves the machine.
    s = summarize(_df())
    assert set(s.keys()) == {"meta", "overall_means", "group_counts"}
    assert "raw" not in s
    assert "records" not in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'` (or ImportError for `summarize`).

- [ ] **Step 3: Write minimal implementation**

Create `webapp/data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/test_data.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add webapp/data.py webapp/test_data.py
git commit -m "feat: add data layer with schema+stats summary"
```

---

## Task 3: Execution layer — `execute_plan()` (TDD)

**Files:**
- Create: `webapp/execute.py`
- Test: `webapp/test_execute.py`

- [ ] **Step 1: Write the failing test**

Create `webapp/test_execute.py`:

```python
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
    assert list(result.index) == ["b"]  # b sums to 120, a to 30
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/test_execute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'execute'`.

- [ ] **Step 3: Write minimal implementation**

Create `webapp/execute.py`. This ports the original `execute_plan` and `generate_graph` logic (provider-independent), with `generate_graph` returning a matplotlib `Figure` instead of a PIL image (Streamlit renders figures directly).

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/test_execute.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add webapp/execute.py webapp/test_execute.py
git commit -m "feat: add execution layer for plans and charts"
```

---

## Task 4: AI layer — Anthropic SDK wrapper

**Files:**
- Create: `webapp/ai.py`

No unit test: this module calls the live API and reads `st.secrets`. It is smoke-tested manually after deploy. Keep all logic that CAN be tested (parsing, execution) out of this module — it stays a thin wrapper.

- [ ] **Step 1: Create `webapp/ai.py`**

```python
"""AI layer: Claude calls that return structured plans.

Reads the API key and model from Streamlit secrets. The only data passed to
Claude is the schema+stats summary from data.summarize() and already-computed
results — never raw rows.
"""

import json
from typing import List, Optional

import anthropic
import streamlit as st
from pydantic import BaseModel


# ---- Structured-output schemas ----

class Ranking(BaseModel):
    order: str  # "asc" or "desc"
    top_n: int


class Filter(BaseModel):
    column: str
    value: str


class QueryPlan(BaseModel):
    filter: Optional[Filter] = None
    aggregation: Optional[str] = None  # mean|sum|count|max|min
    metric: Optional[str] = None
    group_by: Optional[List[str]] = None
    ranking: Optional[Ranking] = None


class ChartPlan(BaseModel):
    chart_type: str  # Bar Chart | Pie Chart | Line Chart | Scatter Plot
    x_column: str
    y_column: str


DEFAULT_MODEL = "claude-haiku-4-5"


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


def _model() -> str:
    return st.secrets.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def plan_query(question: str, summary: dict) -> QueryPlan:
    """Ask Claude for a pandas query plan, given only the data summary."""
    system = (
        "You are a data query planner. Given a dataset summary (columns, dtypes, "
        "means, category counts) and a question, output a plan describing the pandas "
        "operations needed. Use only columns present in the summary. Never compute "
        "the answer yourself — only describe the operations. aggregation is one of "
        "mean, sum, count, max, min. Leave a field null if not needed."
    )
    user = (
        f"Dataset summary:\n{json.dumps(summary, default=str)}\n\n"
        f"Question: {question}"
    )
    response = _client().messages.parse(
        model=_model(),
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=QueryPlan,
    )
    return response.parsed_output


def plan_chart(request: str, summary: dict) -> ChartPlan:
    """Ask Claude which chart type + columns best answer the request."""
    system = (
        "You are a data visualization planner. Given a dataset summary and a chart "
        "request, choose the best chart_type (one of: Bar Chart, Pie Chart, Line "
        "Chart, Scatter Plot) and the x_column and y_column. Use only columns that "
        "exist in the summary."
    )
    user = (
        f"Dataset summary:\n{json.dumps(summary, default=str)}\n\n"
        f"Chart request: {request}"
    )
    response = _client().messages.parse(
        model=_model(),
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=ChartPlan,
    )
    return response.parsed_output


def explain(question: str, plan: QueryPlan, result) -> str:
    """Ask Claude to explain an already-computed result in plain language."""
    payload = {
        "question": question,
        "plan": plan.model_dump(exclude_none=True),
        "result": result.to_dict() if hasattr(result, "to_dict") else result,
    }
    system = (
        "You are a data analyst. Explain the result using ONLY the provided "
        "execution output. Give a direct, concise answer (2-4 sentences) with "
        "specific numbers. If the result does not answer the question, say so."
    )
    response = _client().messages.create(
        model=_model(),
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": json.dumps(payload, default=str)}
        ],
    )
    return next((b.text for b in response.content if b.type == "text"), "")
```

- [ ] **Step 2: Sanity-check syntax**

Run: `python -c "import ast; ast.parse(open('webapp/ai.py').read()); print('ok')"`
Expected: prints `ok` (syntax valid; we don't import it directly because it imports streamlit at module load).

- [ ] **Step 3: Commit**

```bash
git add webapp/ai.py
git commit -m "feat: add Claude AI layer with structured query and chart plans"
```

---

## Task 5: Streamlit UI — `app.py`

**Files:**
- Create: `webapp/app.py`

- [ ] **Step 1: Create `webapp/app.py`**

```python
"""Streamlit UI for the AI CSV reader, backed by Claude."""

import anthropic
import streamlit as st

import ai
from data import load_csv, summarize
from execute import execute_plan, generate_graph

st.set_page_config(page_title="CSV Data Analyzer", page_icon="📊", layout="wide")
st.title("📊 CSV Data Analyzer")
st.caption("Upload a CSV, explore the data, and ask questions — powered by Claude.")

if "ANTHROPIC_API_KEY" not in st.secrets:
    st.error(
        "The app admin needs to set the ANTHROPIC_API_KEY secret in the "
        "Streamlit Cloud settings before AI features will work."
    )

uploaded = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = load_csv(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read that CSV: {e}")
        st.stop()

    st.session_state["df"] = df
    st.subheader("Data preview")
    st.write(f"Columns: {', '.join(df.columns)}")
    st.dataframe(df.head(10))

df = st.session_state.get("df")


def _friendly_ai_error(e: Exception) -> str:
    if isinstance(e, anthropic.RateLimitError):
        return "Claude is rate-limited right now — please wait a moment and retry."
    if isinstance(e, anthropic.AuthenticationError):
        return "The ANTHROPIC_API_KEY secret is missing or invalid."
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error ({e.status_code}). Please retry."
    return f"Something went wrong: {e}"


if df is not None:
    st.divider()
    st.subheader("💬 Ask a question")
    question = st.text_input(
        "Your question", placeholder="e.g., What is the average sales by category?"
    )
    if st.button("Ask Claude", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                plan = ai.plan_query(question, summarize(df))
                result, err = execute_plan(df, plan.model_dump(exclude_none=True))
                if err:
                    st.error(f"Could not run that query: {err}")
                else:
                    st.markdown(ai.explain(question, plan, result))
            except Exception as e:  # noqa: BLE001
                st.error(_friendly_ai_error(e))

    st.divider()
    st.subheader("📈 Create a chart")
    chart_request = st.text_input(
        "Describe the chart", placeholder="e.g., Bar chart of total sales by category"
    )
    if st.button("Make chart", type="primary") and chart_request:
        with st.spinner("Building chart..."):
            try:
                chart = ai.plan_chart(chart_request, summarize(df))
                fig = generate_graph(df, chart.chart_type, chart.x_column, chart.y_column)
                if fig is None:
                    st.error(
                        f"Couldn't build a {chart.chart_type} from "
                        f"{chart.x_column} / {chart.y_column}."
                    )
                else:
                    st.caption(
                        f"{chart.chart_type}: x={chart.x_column}, y={chart.y_column}"
                    )
                    st.pyplot(fig)
            except Exception as e:  # noqa: BLE001
                st.error(_friendly_ai_error(e))
```

- [ ] **Step 2: Verify the app parses without a syntax error**

Run: `python -c "import ast; ast.parse(open('webapp/app.py').read()); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add webapp/app.py
git commit -m "feat: add Streamlit UI wiring data, AI, and execution layers"
```

---

## Task 6: Test suite + gitignore (automatable parts)

**Files:**
- Create: `.gitignore`

The live smoke test (running the app with a real key) is done by the user; this task covers the parts that can be automated here.

- [ ] **Step 1: Create or append `.gitignore` at repo root**

```
.streamlit/secrets.toml
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest webapp/ -v`
Expected: all tests in `test_data.py` (5) and `test_execute.py` (4) PASS.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore local secrets and caches"
```

---

## Task 7: Update README and push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a Streamlit section to `README.md`**

Add to the end of `README.md`:

```markdown
## Streamlit web app (Claude-powered)

A Streamlit version lives in `webapp/`. It uses Claude (Anthropic) and sends
only a schema + summary statistics to the model — never raw rows.

**Run locally:**
1. `pip install -r requirements.txt`
2. Put your key in `.streamlit/secrets.toml`:
   `ANTHROPIC_API_KEY = "sk-ant-..."`
3. `streamlit run webapp/app.py`

**Deploy on Streamlit Community Cloud:**
1. Connect this repo at https://share.streamlit.io
2. Set the main file to `webapp/app.py`
3. Add `ANTHROPIC_API_KEY` (and optional `ANTHROPIC_MODEL`) in the app's
   Secrets settings.
```

- [ ] **Step 2: Push the branch**

```bash
git add README.md
git commit -m "docs: document the Streamlit app and deployment"
git push -u origin streamlit-claude-port
```

Expected: push succeeds to `YutaB123/CSV-Reader`.

---

## Task 8: Deploy on Streamlit Community Cloud (manual, user-driven)

**Files:** none (external service)

This task is done by the user in the browser — it cannot be automated from here. The plan documents the exact steps.

- [ ] **Step 1: Sign in to Streamlit Community Cloud**

Go to https://share.streamlit.io and sign in with the GitHub account `YutaB123`.

- [ ] **Step 2: Create the app**

Click "Create app" → "Deploy a public app from GitHub" → select repo `YutaB123/CSV-Reader`, branch `main` (after merge) or `streamlit-claude-port`, main file path `webapp/app.py`.

- [ ] **Step 3: Add secrets**

In "Advanced settings" → "Secrets", paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-...your-key..."
ANTHROPIC_MODEL = "claude-haiku-4-5"
```

- [ ] **Step 4: Deploy and verify**

Click "Deploy". Wait for the build. When it loads, upload a CSV, ask a question, and request a chart. The public URL (`https://<name>.streamlit.app`) is the working link.

---

## Self-Review Notes

- **Spec coverage:** data layer (Task 2), AI layer (Task 4), execution+UI (Tasks 3, 5), schema-only-to-AI (Task 2 + test_summarize_excludes_raw_rows), deployment to same repo subfolder (Tasks 1, 7, 8), error handling (Task 5 `_friendly_ai_error`, missing-key notice), testing (Tasks 2, 3, 6), Haiku default (Task 4 `DEFAULT_MODEL`). All covered.
- **Type consistency:** `QueryPlan` / `ChartPlan` defined in Task 4 are consumed in Task 5 via `.model_dump(exclude_none=True)` and attribute access (`chart.chart_type`); `execute_plan` (Task 3) consumes the dict form; `generate_graph` returns a Figure consumed by `st.pyplot`. Consistent.
- **Import scheme:** folder is `webapp/` (not `streamlit/`) to avoid shadowing the library; all intra-app imports are flat; `webapp/conftest.py` makes pytest resolve them.
- **Anthropic API:** uses `messages.parse(..., output_format=Model)` returning `.parsed_output`, and plain `messages.create(...)` for `explain`, per the current SDK. Model default `claude-haiku-4-5`.
