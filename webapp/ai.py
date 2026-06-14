"""AI layer: Claude calls that return structured plans.

Reads the API key and model from configuration that works on both Streamlit
Community Cloud (st.secrets) and Render/other hosts (environment variables).
The only data passed to Claude is the schema+stats summary from
data.summarize() and already-computed results — never raw rows.
"""

import json
import os
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


def _config(name: str, default=None):
    """Read config from Streamlit secrets (Streamlit Cloud) or env vars (Render).

    st.secrets raises if no secrets file exists, so guard the lookup and fall
    back to the environment.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001 - no secrets file present (e.g. on Render)
        pass
    return os.environ.get(name, default)


def api_key_configured() -> bool:
    """True if an Anthropic API key is available from secrets or env."""
    return bool(_config("ANTHROPIC_API_KEY"))


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=_config("ANTHROPIC_API_KEY"))


def _model() -> str:
    return _config("ANTHROPIC_MODEL", DEFAULT_MODEL)


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
    if response.parsed_output is None:
        raise ValueError("Claude did not return a usable query plan.")
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
    if response.parsed_output is None:
        raise ValueError("Claude did not return a usable chart plan.")
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
