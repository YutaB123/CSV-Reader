# CSV-Reader
This project is an AI-enabled CSV Reader and Data Explorer. It parses CSV files, displays data, summarizes key statistics, and generates visualizations. Users can ask questions such as “Which month had the highest sales?” or “Show a bar chart grouped by category” and get answers automatically. Ideal for students, analysts.

The original Gradio + OpenAI prototype lives in `csv_reader/`. A deployable Streamlit + Claude version lives in `webapp/`.

## Streamlit web app (Claude-powered)

The Streamlit app in `webapp/` uses Claude (Anthropic) and sends only a schema +
summary statistics to the model — never raw rows. The AI plans a query, Python
runs it locally on the full data, and the AI explains the computed result.

**Run locally:**
1. `pip install -r requirements.txt`
2. Put your key in `.streamlit/secrets.toml`:
   `ANTHROPIC_API_KEY = "sk-ant-..."`
3. `streamlit run webapp/app.py`

**Deploy on Streamlit Community Cloud:**
1. Connect this repo at https://share.streamlit.io
2. Set the main file to `webapp/app.py`
3. Add `ANTHROPIC_API_KEY` (and optional `ANTHROPIC_MODEL`, default
   `claude-haiku-4-5`) in the app's Secrets settings.
