# 🕵️‍♂️ Deep Research Agent

A powerful, autonomous research assistant built with **LangGraph**, **Gemini 2.5 Flash**, and **Streamlit**. This agent takes a user topic, plans a multi-step research strategy, crawls the web for information, and synthesizes a professional Markdown report (with PDF export).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/AI-LangGraph-orange)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.5-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 🚀 Features

*   **Autonomous Planning**: Breaks complex topics into distinct, search-optimized queries.
*   **Deep Web Scraping**: Uses **DuckDuckGo** for privacy-focused searching and **Trafilatura** for robust content extraction.
*   **Cyclic Research Workflow**: Implements a feedback loop that continues researching until the plan is complete.
*   **AI Synthesis**: Uses **Gemini 2.5 Flash** to read, summarize, and compile findings into a cohesive report.
*   **Interactive UI**: A beautiful Streamlit chat interface with real-time progress tracking.
*   **PDF Export**: Download your final research reports directly as PDFs.
*   **Cost-Effective**: Designed to run entirely with a **free** Google API Key (no OpenAI or paid search APIs required).

## 🛠️ Architecture

The agent is built as a state machine using **LangGraph**:

1.  **Planner Node**:
    *   Input: User Topic.
    *   Action: Generates a 3-step search plan.
2.  **Research Node** (Loop):
    *   Action: Executes the current search query.
    *   Search: DuckDuckGo (Top 3 results).
    *   Scrape: Extracts main text content from URLs.
    *   Summarize: Gemini condenses the scraped content.
    *   State Update: Appends summary to context, advances index.
3.  **Manager Logic**:
    *   Check: Are there more queries in the plan?
    *   Routing: If yes $\rightarrow$ Loop back to *Research Node*. If no $\rightarrow$ Proceed to *Writer Node*.
4.  **Writer Node**:
    *   Action: Compiles all summaries into a final Markdown report.

## 📦 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/deep-research-agent.git
    cd deep-research-agent
    ```

2.  **Create a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Run

### Streamlit UI
```bash
streamlit run app.py
```

### CLI
```bash
python main.py
```

## Test (validated)

The current repository test suite passes with:
```bash
python -m pytest -q
```

## Project layout

```text
.
├── app.py
├── main.py
├── requirements.txt
├── tests/
│   ├── test_agent.py
│   ├── integration/
│   └── unit/
└── final_report.md   (runtime output)
```

## Limitations (code-backed)

- Requires `GOOGLE_API_KEY` (from sidebar state or environment); otherwise execution stops/errors.
- Planner uses at most 3 generated queries.
- Search uses DuckDuckGo top 3 results per query.
- Scraping only processes absolute `http/https` URLs.
- Scraped text per source is truncated to 15000 characters.
- Search and scrape exceptions are handled by skipping failed items and continuing; output may be partial.
- Final report quality depends on external search/scraped content and model response.
- Streamlit state is in-session memory (`st.session_state`); no persistent database/storage layer in code.

<p align="center">Made with ❤️ by Ahmad Mujtaba</p>
