import gradio as gr
import pandas as pd
from openai import OpenAI
import json
import tiktoken
import matplotlib.pyplot as plt
import io
from PIL import Image
import re

# OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI()

# -------------------------
# CSV UPLOAD
# -------------------------
def upload_csv(file):
    if file is None:
        return "No file uploaded.", None, None

    try:
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        return (
            f"Columns: {', '.join(cols)}",
            df.head(10),
            df
        )
    except Exception as e:
        return f"Error loading CSV: {e}", None, None


# -------------------------
# COMPUTATION (PYTHON SIDE)
# -------------------------
def compute_statistics(question: str, df: pd.DataFrame) -> dict:
    n_rows = int(len(df))

    # Identify numeric and categorical columns dynamically
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Start with object/category dtypes
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Treat low-cardinality numeric columns as categorical as well (e.g., 0/1 flags)
    low_card_num_as_cat = [
        c for c in numeric_cols if df[c].nunique(dropna=True) <= 20
    ]
    categorical_cols = list(dict.fromkeys(cat_cols + low_card_num_as_cat))

    # Overall means for numeric columns
    overall_means = {}
    if numeric_cols:
        means = df[numeric_cols].mean(numeric_only=True)
        overall_means = {c: (float(means[c]) if pd.notna(means[c]) else None) for c in numeric_cols}

    # Grouped means for each categorical column (limit categories to top N by frequency)
    group_means = {}
    TOP_K = 30
    for cat in categorical_cols:
        # Determine top-k categories by frequency to control payload size
        vc = df[cat].astype("string").value_counts(dropna=True).head(TOP_K)
        top_cats = vc.index.tolist()
        sub = df[df[cat].astype("string").isin(top_cats)]
        means_by_num = {}
        for num in numeric_cols:
            grp = (
                sub.groupby(sub[cat].astype("string"))[num]
                .mean(numeric_only=True)
                .dropna()
            )
            means_by_num[num] = {str(k): float(v) for k, v in grp.to_dict().items()}
        group_means[cat] = {
            "top_categories_by_frequency": [str(v) for v in top_cats],
            "means": means_by_num,
        }

    # Grouped counts for each categorical column (helps with "most" questions)
    group_counts = {}
    for cat in categorical_cols:
        vc = df[cat].astype("string").value_counts(dropna=True).head(TOP_K)
        group_counts[cat] = {str(idx): int(cnt) for idx, cnt in vc.to_dict().items()}

    computed = {
        "meta": {
            "total_rows": n_rows,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        "overall_means": overall_means,
        "group_means": group_means,
        "group_counts": group_counts,
    }

    return computed


# -------------------------
# DECISION AGENT (LLM) - PLANNING ONLY
# -------------------------
def decide_plan(question: str, df: pd.DataFrame) -> dict:
    schema = {
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist(),
    }
    
    # Convert full CSV to JSON format for OpenAI
    full_data = df.to_dict(orient='records')

    system_prompt = """
You are a data decision agent.
Decide WHAT data operations are required to answer the user’s request. Never compute results. Do not load or execute code. Output only a JSON plan.

You must decide:
- filter (optional): {"column": str, "value": any}
- aggregation (optional): one of [mean, sum, count, max, min]
- metric (optional): the numeric column to aggregate
- group_by (optional): column name or list of columns
- ranking (optional): {"order": "desc"|"asc", "top_n": int}
- chart_type (optional): one of [bar, pie, line, scatter]

Rules:
- Use only columns that exist.
- If the request is unclear, return {"error": "cannot_determine"}.
- Do NOT compute numbers or assume values.
"""

    user_prompt = (
        "Full Dataset:\n" + json.dumps(full_data, ensure_ascii=False, default=str) + "\n\n" +
        "Schema:\n" + json.dumps(schema, ensure_ascii=False) + "\n\n" +
        f"User question: {question}\n" +
        "Return only a JSON plan with keys from the rules."
    )

    print("\n[DEBUG] Data sent to OpenAI (decide_plan):")
    print(f"  - Full CSV data sent: {len(df)} rows")
    print(f"  - Columns: {schema['columns']}")
    print(f"  - Total data points: {len(df) * len(df.columns)}\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    plan_text = response.choices[0].message.content.strip()
    # Extract JSON if inside code fences
    if "```json" in plan_text:
        plan_text = plan_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in plan_text:
        plan_text = plan_text.split("```", 1)[1].split("```", 1)[0].strip()

    return json.loads(plan_text)


# -------------------------
# EXECUTION (PYTHON) - APPLY PLAN
# -------------------------
def execute_plan(df: pd.DataFrame, plan: dict):
    try:
        working = df.copy()

        # Filtering
        flt = plan.get("filter")
        if flt:
            col = flt.get("column")
            val = flt.get("value")
            if col not in working.columns:
                return None, "Filter column not found"
            working = working[working[col] == val]

        # Determine grouping
        group_by = plan.get("group_by")
        if group_by and isinstance(group_by, str):
            group_by = [group_by]

        aggregation = plan.get("aggregation")
        metric = plan.get("metric")

        result = None

        if aggregation:
            agg_func = aggregation
            if group_by:
                grouped = working.groupby(group_by)
                if agg_func == "count":
                    if metric and metric in working.columns:
                        result = grouped[metric].count()
                    else:
                        result = grouped.size()
                else:
                    if not metric or metric not in working.columns:
                        return None, "Metric missing for aggregation"
                    result = grouped[metric].agg(agg_func)
            else:
                if agg_func == "count":
                    result = len(working) if not metric else working[metric].count()
                else:
                    if not metric or metric not in working.columns:
                        return None, "Metric missing for aggregation"
                    result = getattr(working[metric], agg_func)()
        else:
            # Default to row count if no aggregation specified
            result = len(working)

        # Ranking (on aggregated result)
        ranking = plan.get("ranking")
        if ranking and hasattr(result, "sort_values"):
            order = ranking.get("order", "desc")
            top_n = ranking.get("top_n", 10)
            ascending = order == "asc"
            result = result.sort_values(ascending=ascending).head(top_n)

        return result, None

    except Exception as e:
        return None, str(e)


# -------------------------
# OPENAI EXPLANATION (LLM SIDE)
# -------------------------
def get_answer_from_openai(question: str, columns, execution_payload: dict):
    system_prompt = f"""You are a data analyst. Explain the result using only the provided execution output from a CSV dataset.

Dataset columns: {', '.join(columns)}

Rules:
- Use only the provided execution result and plan
- Give direct, concise answers (2-4 sentences)
- If data is missing to answer the question, say so clearly
- Include specific numbers or categories from the execution result
"""

    user_prompt = (
        "Use only this execution result to answer concisely:\n" +
        json.dumps(execution_payload, ensure_ascii=False)
    )

    print("\n[DEBUG] Data sent to OpenAI (get_answer):")
    print(f"  - Execution payload keys: {list(execution_payload.keys())}")
    print(f"  - Result type: {type(execution_payload.get('result'))}")
    print(f"  - No raw CSV data sent\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()



# -------------------------  
# CLEAR HISTORY
# -------------------------
def clear_history():
    return "", ""

# -------------------------
# GRAPH GENERATION
# -------------------------
def generate_graph_with_ai(df, user_request):
    """Uses AI to determine the best chart type and columns based on user request"""
    if df is None:
        return None, "Please upload a CSV first."
    
    if not user_request or user_request.strip() == "":
        return None, "Please describe what graph you want to create."

    # Parse a requested "top N" if present
    top_n = None
    match = re.search(r"top\s+(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        try:
            top_n = int(match.group(1))
        except ValueError:
            top_n = None

    try:
        # Get column info with FULL dataset
        columns_info = {
            "columns": list(df.columns),
            "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "full_data": df.to_dict(orient='records')  # Send ALL rows
        }
        
        print("\n[DEBUG] Data sent to OpenAI (generate_graph):")
        print(f"  - Full CSV rows sent: {len(df)}")
        print(f"  - Columns sent: {columns_info['columns']}")
        print(f"  - Total data points: {len(df) * len(df.columns)}\n")
        
        system_prompt = """
You are a data visualization expert. Based on the user's request and the FULL dataset provided, determine:
1. The best chart type (Bar Chart, Pie Chart, Line Chart, or Scatter Plot)
2. The X-axis column
3. The Y-axis column

Respond ONLY with a JSON object in this exact format:
{
  "chart_type": "Bar Chart",
  "x_column": "column_name",
  "y_column": "column_name"
}

Rules:
- Choose columns that exist in the dataset
- Match the chart type to the user's intent and data types
- For categorical comparisons, use Bar Chart or Pie Chart
- For trends over time/sequence, use Line Chart
- For numeric relationships, use Scatter Plot
- You have access to the complete dataset
"""
        
        user_prompt = f"""
Dataset Info:
{json.dumps(columns_info, indent=2)}

User Request: {user_request}

Provide the JSON response for the best visualization.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        
        # Parse AI response
        ai_response = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in ai_response:
            ai_response = ai_response.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_response:
            ai_response = ai_response.split("```")[1].split("```")[0].strip()
            
        config = json.loads(ai_response)
        
        chart_type = config.get("chart_type", "Bar Chart")
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        
        if not x_col or not y_col:
            return None, "AI couldn't determine appropriate columns."
        
        if x_col not in df.columns or y_col not in df.columns:
            return None, f"Columns not found: {x_col}, {y_col}"
        
        # Generate the graph
        img = generate_graph(df, chart_type, x_col, y_col, top_n=top_n)
        
        if img is None:
            return None, "Failed to generate graph."
        
        status = f"Created {chart_type}: X={x_col}, Y={y_col}"
        return img, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def generate_graph(df, chart_type, x_col, y_col, top_n=None):
    if df is None:
        return None
    
    if not x_col or not y_col:
        return None
    
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    try:
        plt.figure(figsize=(10, 6))

        if chart_type == "Bar Chart":
            palette = list(plt.cm.tab20.colors)
            if df[y_col].dtype in ['int64', 'float64']:
                grouped = df.groupby(x_col)[y_col].sum()
                grouped = grouped.sort_values(ascending=False)
                if top_n:
                    grouped = grouped.head(top_n)
                colors = palette[: len(grouped)]
                grouped.plot(kind='bar', color=colors)
            else:
                counts = df[x_col].value_counts()
                counts = counts.sort_values(ascending=False)
                if top_n:
                    counts = counts.head(top_n)
                colors = palette[: len(counts)]
                counts.plot(kind='bar', color=colors)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} by {x_col}")

        elif chart_type == "Pie Chart":
            if df[y_col].dtype in ['int64', 'float64']:
                grouped = df.groupby(x_col)[y_col].sum()
                grouped.plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('')
            else:
                df[x_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('')
            plt.title(f"{y_col} Distribution by {x_col}")

        elif chart_type == "Line Chart":
            if df[y_col].dtype in ['int64', 'float64']:
                grouped = df.groupby(x_col)[y_col].mean()
                grouped.plot(kind='line', marker='o')
            else:
                df[x_col].value_counts().sort_index().plot(kind='line', marker='o')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} by {x_col}")

        elif chart_type == "Scatter Plot":
            if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                plt.scatter(df[x_col], df[y_col], alpha=0.6, c=plt.cm.tab20.colors[0])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{y_col} vs {x_col}")
            else:
                return None

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img

    except Exception as e:
        print(f"Graph error: {e}")
        plt.close()
        return None

# -------------------------
def ask_question(question, df):
    if df is None:
        return "Please upload a CSV first."

    try:
        plan = decide_plan(question, df)
        if plan.get("error"):
            return "This question cannot be answered because the required plan could not be determined."

        execution_result, err = execute_plan(df, plan)
        if err:
            return f"Error executing plan: {err}"

        payload = {
            "question": question,
            "plan": plan,
            "result": execution_result.to_dict() if hasattr(execution_result, "to_dict") else execution_result,
        }

        answer = get_answer_from_openai(question, list(df.columns), payload)
        return answer

    except Exception as e:
        return f"Error: {e}"


# -------------------------
# GRADIO UI
# -------------------------
# Custom dark theme with better styling
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    panel_background_fill="*neutral_900",
    panel_background_fill_dark="*neutral_900",
    block_background_fill="*neutral_800",
    block_label_background_fill="*neutral_900",
    input_background_fill="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_hover="*neutral_600",
)

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
}
button {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary {
    min-height: 50px !important;
}
.gr-button-secondary {
    min-height: 45px !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}
.gr-box {
    border-radius: 12px !important;
    border: 1px solid #333 !important;
}
textarea, input[type="text"] {
    background: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
textarea::placeholder, input::placeholder {
    color: #888 !important;
}
.gr-form {
    background: #1f1f1f !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.gr-input textarea, .gr-textbox textarea, .gr-input input {
    color: #ffffff !important;
    background: #2a2a2a !important;
}

"""

with gr.Blocks(title="CSV Data Analyzer") as demo:
    gr.Markdown("""
    # 📊 CSV Data Analyzer
    ### Upload your CSV file, explore the data, and ask questions powered by AI
    """, elem_classes="header-text")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Your Data")
            file_input = gr.File(label="Select CSV File", file_types=[".csv"], elem_id="file-upload")
            upload_btn = gr.Button("📤 Upload & Analyze", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Data Preview")
            columns_output = gr.Textbox(label="Columns Detected", interactive=False, lines=2)
            preview_output = gr.Dataframe(label="First 10 Rows Preview", interactive=False)
    
    df_state = gr.State()

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💬 Ask a Question")
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What is the average price?",
                lines=2
            )
            with gr.Row():
                ask_btn = gr.Button("🤖 Ask AI", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", scale=1)
    
    with gr.Row():
        result_output = gr.Textbox(label="📝 AI Answer", lines=10, interactive=False)
    
    with gr.Row():
        copy_btn = gr.Button("📋 Copy Answer", variant="secondary", size="lg")

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Create Graph")
            graph_request = gr.Textbox(
                label="Describe the graph you want", 
                placeholder="e.g., Show me a bar chart of sales by category",
                lines=2
            )
            graph_btn = gr.Button("📈 Make Graph with AI", variant="primary", size="lg")
    
    graph_output = gr.Image(label="Generated Graph", type="pil")
    graph_status = gr.Textbox(label="Graph Info", interactive=False, lines=1)

    upload_btn.click(
        upload_csv,
        inputs=file_input,
        outputs=[columns_output, preview_output, df_state],
    )

    ask_btn.click(
        ask_question,
        inputs=[question_input, df_state],
        outputs=result_output,
    )

    clear_btn.click(
        clear_history,
        inputs=[],
        outputs=[question_input, result_output],
    )

    copy_btn.click(
        lambda x: x,
        inputs=[result_output],
        outputs=[],
        js="(x) => {navigator.clipboard.writeText(x); return x;}"
    )

    graph_btn.click(
        generate_graph_with_ai,
        inputs=[df_state, graph_request],
        outputs=[graph_output, graph_status],
    )

demo.launch(server_name="127.0.0.1", server_port=7862, theme=theme, css=custom_css)

