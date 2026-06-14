"""Streamlit UI for the AI CSV reader, backed by Claude."""

import anthropic
import matplotlib.pyplot as plt
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
                    plt.close(fig)
            except Exception as e:  # noqa: BLE001
                st.error(_friendly_ai_error(e))
