# Deep Research Agent

This repository contains a LangGraph-based research workflow with two entry points:
- CLI runner in `main.py`
- Streamlit UI in `app.py`

The model configured in code is `gemini-2.5-flash`.

## Architecture (from code)

Core workflow is implemented in `main.py` as a LangGraph state machine over `AgentState`.

State fields:
- `topic`
- `api_key`
- `plan`
- `current_query_index`
- `summaries`
- `final_report`

Nodes:
1. `planner_node`
   - Uses Gemini to generate query lines.
   - Cleans each query (`_clean_query`) and keeps up to 3 non-empty queries.
   - Raises `ValueError` if the resulting plan is empty.
2. `research_node`
   - Runs DuckDuckGo search (`DDGS().text`) with `max_results=3`.
   - Scrapes each result concurrently (`ThreadPoolExecutor(max_workers=3)`).
   - Accepts only absolute `http/https` URLs (`_is_valid_web_url`).
   - Extracts content with `trafilatura` and truncates each source text to 15000 chars.
   - Summarizes combined scraped content with Gemini.
   - If nothing is scraped, returns a fallback summary (includes search error text when present).
3. `writer_node`
   - Combines all summaries and asks Gemini to produce final Markdown report text.
4. `manager_logic`
   - Routes `researcher -> researcher` while `current_query_index < len(plan)`.
   - Routes to `writer` when research loop is complete.

Graph wiring:
- Entry: `planner`
- Edge: `planner -> researcher`
- Conditional: `researcher -> researcher|writer`
- Edge: `writer -> END`

## Execution flow

### CLI (`main.py`)
1. Loads environment variables via `python-dotenv` (`load_dotenv()`).
2. If `GOOGLE_API_KEY` is missing, prompts user input for key.
3. Prompts research topic and strips whitespace.
4. Builds initial graph state and calls `app.invoke(initial_state)`.
5. Prints final report and writes `final_report.md`.

### Streamlit (`app.py`)
1. Reads API key from sidebar `st.text_input`.
2. Accepts topic via `st.chat_input`.
3. Builds initial state and streams graph events via `graph_app.stream(initial_state)`.
4. Displays planner/research progress in status UI.
5. Stores final report in `st.session_state`.
6. Converts Markdown to PDF on demand path and exposes `st.download_button` (`research_report.pdf`).

PDF conversion (`convert_markdown_to_pdf`):
- Markdown -> HTML (`markdown.markdown`)
- Sanitization (`nh3.clean` with an allowlist of tags)
- PDF rendering (`xhtml2pdf.pisa.CreatePDF`)

## Dependencies

From `requirements.txt`:
- `langchain-google-genai`
- `langgraph`
- `langchain-core`
- `duckduckgo-search`
- `trafilatura`
- `python-dotenv`
- `streamlit`
- `markdown`
- `xhtml2pdf`
- `nh3`
- `pytest`
- `pytest-cov`

## Environment setup (venv)

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
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
