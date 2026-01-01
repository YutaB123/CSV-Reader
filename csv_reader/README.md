# 📊 CSV Data Analyzer

An AI-powered CSV analysis tool built with Gradio and OpenAI. Upload your CSV files, explore data, ask natural language questions, and generate visualizations.

## ✨ Features

- **📁 CSV Upload & Preview**: Load CSV files and instantly preview columns and data
- **💬 AI-Powered Q&A**: Ask natural language questions about your data
- **📈 Smart Graph Generation**: Describe the chart you want, AI picks the best visualization
- **🎨 Multiple Chart Types**: Bar charts, pie charts, line charts, scatter plots
- **🌙 Dark Theme UI**: Modern, sleek interface

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd csv_reader
pip install gradio pandas openai tiktoken matplotlib pillow
```

### 2. Set OpenAI API Key

Set your API key as an environment variable:

```bash
# Windows
set OPENAI_API_KEY=sk-your-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the App

```bash
python csv_reader.py
```

Open http://127.0.0.1:7862 in your browser.

## 🔧 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GRADIO UI                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │CSV Upload│  │ Question Box │  │ Graph Request        │  │
│  └──────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING PIPELINE                       │
│                                                             │
│  1. PLAN (LLM)        2. EXECUTE (Python)   3. EXPLAIN (LLM)│
│  ┌─────────────┐     ┌─────────────────┐   ┌─────────────┐ │
│  │ Decide what │ ──▶ │ Filter, Group,  │ ──▶│ Natural     │ │
│  │ operations  │     │ Aggregate data  │   │ language    │ │
│  └─────────────┘     └─────────────────┘   │ answer      │ │
│                                            └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### AI Decision Agent

The app uses a two-step AI process:

1. **Planning**: GPT-4o-mini analyzes your question and creates a structured plan:
   - Filter conditions
   - Aggregation type (mean, sum, count, max, min)
   - Grouping columns
   - Ranking parameters

2. **Execution**: Python executes the plan on your data locally (no data sent to AI)

3. **Explanation**: AI generates a natural language answer from the results

## 📊 Example Questions

- "What is the average price?"
- "Which category has the most sales?"
- "Show me the top 10 products by revenue"
- "What's the total count by region?"

## 📈 Example Graph Requests

- "Show me a bar chart of sales by category"
- "Create a pie chart of market share"
- "Line chart of revenue over time"
- "Top 5 products by quantity as a bar chart"

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **UI** | Gradio |
| **Data Processing** | Pandas |
| **AI** | OpenAI GPT-4o-mini |
| **Visualization** | Matplotlib |
| **Token Counting** | tiktoken |

## 📁 Project Structure

```
csv_reader/
├── csv_reader.py    # Main application
└── README.md        # This file
```

## 🔒 Privacy

- Your CSV data is processed locally
- Only schema and aggregated results are sent to OpenAI for analysis
- No raw data is stored

## 📝 License

MIT License - Feel free to use and modify!
