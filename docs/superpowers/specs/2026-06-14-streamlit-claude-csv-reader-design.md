# CSV Reader → Streamlit + Claude — Design

**Date:** 2026-06-14
**Status:** Approved

## Goal

Port the existing Gradio-based AI CSV analyzer to a Streamlit app deployed on
Streamlit Community Cloud, giving the project a live public URL, and switch the
AI backend from OpenAI to Claude (Anthropic).

## Background

The current app (`csv_reader/csv_reader.py`) is a Gradio UI with three features:

1. CSV upload + preview (columns detected, first 10 rows).
2. Ask-a-question: an LLM plans a pandas operation → Python executes it → an LLM
   explains the computed result.
3. AI graph generation: the user describes a chart, an LLM picks chart type +
   columns, matplotlib renders it.

It runs on OpenAI `gpt-4o-mini` and, in the planning and graph steps, sends the
**entire CSV** to the model. There is no deployment.

## Decisions

- **AI provider:** Claude via the official `anthropic` Python SDK.
- **Model:** configurable via `ANTHROPIC_MODEL` secret; **defaults to
  `claude-haiku-4-5`** ($1/$5 per MTok) for cost, since the public link spends
  the owner's credits and the tasks (plan a query, pick a chart, explain a
  number) are simple. Can be set to `claude-opus-4-8` for max quality.
- **API key:** owner's key, stored as a Streamlit secret
  (`st.secrets["ANTHROPIC_API_KEY"]`). The public link spends the owner's
  credits (accepted tradeoff).
- **Data sent to AI:** schema + summary statistics only — column names, dtypes,
  numeric means, categorical top-value counts. **Raw rows never leave the
  machine.**
- **Repo location:** new app lives in a `streamlit/` subfolder of the existing
  `CSV-Reader` repo. The old Gradio `csv_reader/csv_reader.py` is left untouched.

## Architecture

Three modules under `streamlit/`:

### `data.py` — data layer (no AI)
- `load_csv(file) -> DataFrame` — parse an uploaded CSV with pandas.
- `summarize(df) -> dict` — produce the *schema + stats* payload that is the
  ONLY thing the AI ever sees:
  - `meta`: total rows, numeric columns, categorical columns (object/category
    dtypes plus low-cardinality numerics treated as categorical).
  - `overall_means` for numeric columns.
  - `group_counts` (top-K value counts) for categorical columns.
  No raw rows are included.

### `ai.py` — AI layer (Anthropic SDK)
- Client reads the key from `st.secrets["ANTHROPIC_API_KEY"]`; model from
  `st.secrets.get("ANTHROPIC_MODEL", "claude-haiku-4-5")`.
- `plan_query(question, summary) -> QueryPlan` — uses `client.messages.parse()`
  with a Pydantic `QueryPlan` schema (`filter`, `aggregation`, `metric`,
  `group_by`, `ranking`). Structured outputs replace the current
  parse-JSON-from-a-code-fence approach.
- `explain(question, plan, result) -> str` — plain-text Claude call that
  explains the already-computed result in 2-4 sentences. Sees only the result,
  never the data.
- `plan_chart(request, summary) -> ChartPlan` — structured
  `{chart_type, x_column, y_column}` via `messages.parse()`.
- Errors (`anthropic.RateLimitError`, `anthropic.APIStatusError`, refusal stop
  reason) are caught and returned as friendly strings.

### `app.py` — execution + UI (Streamlit)
- Streamlit UI: file uploader, data preview, ask-a-question box + answer,
  chart-request box + rendered chart.
- Reuses provider-independent pandas/matplotlib logic from the original,
  refactored into:
  - `execute_plan(df, plan) -> (result, error)` — runs the plan on the FULL
    DataFrame locally (filter → group → aggregate → rank).
  - `generate_graph(df, chart_type, x_col, y_col) -> figure` — matplotlib chart.
- Uses `st.session_state` to hold the parsed DataFrame across interactions.

## Data flow (Q&A)

```
upload CSV → summarize() ──► user question
                              │
                  plan_query(question, summary)   [Claude sees summary only]
                              │  QueryPlan
                  execute_plan(df, plan)           [runs on full df in Python]
                              │  result
                  explain(question, plan, result)  [Claude sees result only]
                              │
                           display answer
```

## Deployment

1. Commit `streamlit/` (`app.py`, `data.py`, `ai.py`), `requirements.txt`, and
   `.streamlit/config.toml` to the `CSV-Reader` repo; push.
2. On Streamlit Community Cloud: connect the repo, set main file to
   `streamlit/app.py`.
3. Add `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) in the app's
   Secrets UI.
4. Streamlit builds and serves a public `*.streamlit.app` URL.

`requirements.txt`: `streamlit`, `pandas`, `matplotlib`, `anthropic`, `pydantic`.

## Error handling

- Bad/unparseable CSV → friendly inline message.
- Missing `ANTHROPIC_API_KEY` secret → clear notice that the app admin must set
  it.
- Claude rate limit / API error / refusal → caught, surfaced as a readable
  message, app stays usable.
- Plan execution failure (bad column, missing metric) → shown inline.

## Testing

- Unit tests (`streamlit/test_data.py`, `streamlit/test_execute.py`) for
  `summarize()` and `execute_plan()` — pure pandas, no API key needed.
- AI layer smoke-tested manually after deploy with a sample CSV.

## Out of scope (YAGNI)

- Authentication / login / password gate.
- Multi-file upload, data persistence, history.
- Porting the original dark-theme CSS (Streamlit provides its own theming).
- Removing or modifying the original Gradio app.
