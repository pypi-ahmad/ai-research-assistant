# TEST REPORT

## 1) Codebase Summary

- Core orchestrator is the LangGraph workflow in [main.py](main.py#L24-L269), with state defined in [AgentState](main.py#L24-L34).
- Graph nodes and routing are implemented in:
  - [planner_node](main.py#L73-L109)
  - [research_node](main.py#L112-L201)
  - [writer_node](main.py#L205-L226)
  - [manager_logic](main.py#L230-L236)
- CLI entry path is [main.py](main.py#L272-L309), including report output to `final_report.md`.
- Streamlit UI entry path is [app.py](app.py#L53-L188), including event streaming from `graph_app.stream(initial_state)` in [app.py](app.py#L139).
- PDF conversion path is [convert_markdown_to_pdf](app.py#L10-L50), with sanitization via `nh3.clean` in [app.py](app.py#L23).
- Declared dependencies are listed in [requirements.txt](requirements.txt#L1-L12).

## 2) Issues Found (with file + line refs)

### Critical

1. **Global API-key mutation risk (session isolation/security)**
   - Evidence (pre-fix area in UI config): key input and runtime key handling in [app.py](app.py#L85-L113).
   - Fixed behavior evidence (state-passed key + per-call LLM key): [app.py](app.py#L130), [main.py](main.py#L41-L48), [main.py](main.py#L86), [main.py](main.py#L190), [main.py](main.py#L224).

### Major

2. **Planner could retain empty queries after sanitization**
   - Fixed filtering now occurs post-cleaning at [main.py](main.py#L93-L94).

3. **Unvalidated fetch targets from search results**
   - URL validation helper added at [main.py](main.py#L66-L69).
   - Invalid URL skip guard in scraping path at [main.py](main.py#L143-L150).

4. **Search failure visibility and diagnostics**
   - Error capture in research stage at [main.py](main.py#L129-L137).
   - Fallback summary includes captured search error at [main.py](main.py#L194-L195).

5. **State contract consistency (`final_report`)**
   - Agent state declares `final_report` in [main.py](main.py#L34).
   - Initial state now includes `final_report` in CLI at [main.py](main.py#L294).
   - Initial state now includes `final_report` in UI at [app.py](app.py#L134).

### Minor

6. **PDF regeneration on every UI rerun**
   - Cache keys/state fields in [app.py](app.py#L101-L104).
   - Conditional PDF regeneration in [app.py](app.py#L181-L184).

7. **Raw exception text shown to end users**
   - User-safe error message now used in [app.py](app.py#L174).
   - Internal print retained for diagnostics in [app.py](app.py#L173).

8. **Unused imports in tests**
   - Current cleaned test imports at [tests/test_agent.py](tests/test_agent.py#L21-L22).

## 3) Tests Created

- Unit tests (main workflow behaviors): [tests/unit/test_main_unit.py](tests/unit/test_main_unit.py)
  - [test_writer_node_returns_final_report_from_llm](tests/unit/test_main_unit.py#L4)
  - [test_planner_node_drops_empty_items_after_cleaning](tests/unit/test_main_unit.py#L33)
- Unit tests (PDF conversion behavior): [tests/unit/test_app_pdf_unit.py](tests/unit/test_app_pdf_unit.py)
  - [test_convert_markdown_to_pdf_returns_bytes_and_sanitizes_script](tests/unit/test_app_pdf_unit.py#L53)
  - [test_convert_markdown_to_pdf_returns_none_on_pdf_error](tests/unit/test_app_pdf_unit.py#L76)
- Integration tests (compiled graph flow): [tests/integration/test_graph_integration.py](tests/integration/test_graph_integration.py)
  - [test_compiled_graph_invoke_happy_path](tests/integration/test_graph_integration.py#L28)
  - [test_compiled_graph_bubbles_planner_failure](tests/integration/test_graph_integration.py#L53)
- Existing comprehensive suite remains in [tests/test_agent.py](tests/test_agent.py), including class groups:
  - [TestGetLlm](tests/test_agent.py#L54)
  - [TestPlannerNode](tests/test_agent.py#L86)
  - [TestResearchNode](tests/test_agent.py#L122)
  - [TestCleanQuery](tests/test_agent.py#L262)
  - [TestManagerLogic](tests/test_agent.py#L291)
  - [TestPdfHtmlSanitization](tests/test_agent.py#L346)

## 4) Failures Detected

- One failure was detected during test-generation validation (before final fixes):
  - **Test**: `tests/unit/test_app_pdf_unit.py::test_convert_markdown_to_pdf_returns_none_on_pdf_error`
  - **Failure output**: `TypeError ... fake_create_pdf() got an unexpected keyword argument 'dest'`
  - **Stack mapping**:
    - call site in app: [app.py](app.py#L46)
    - failing test function: [tests/unit/test_app_pdf_unit.py](tests/unit/test_app_pdf_unit.py#L76-L88)
  - **Resolution evidence**: mock signature updated to accept `dest` in [tests/unit/test_app_pdf_unit.py](tests/unit/test_app_pdf_unit.py#L82).

## 5) Fixes Applied (diff summary)

- **main.py**
  - `get_llm` now accepts optional explicit key and resolves per-call key first: [main.py](main.py#L41-L48).
  - Added URL validation helper and guard: [main.py](main.py#L66-L69), [main.py](main.py#L143-L150).
  - Planner now removes empty post-sanitization entries: [main.py](main.py#L93-L94).
  - Research fallback includes search-error detail: [main.py](main.py#L129-L137), [main.py](main.py#L194-L195).
  - CLI initial state now includes `final_report`: [main.py](main.py#L294).

- **app.py**
  - UI passes `api_key` into graph state: [app.py](app.py#L130).
  - UI initial state includes `final_report`: [app.py](app.py#L134).
  - User-facing exception text hardened: [app.py](app.py#L173-L174).
  - PDF output caching to reduce rerun overhead: [app.py](app.py#L101-L104), [app.py](app.py#L181-L184).

- **tests**
  - Added new unit/integration files: [tests/unit/test_main_unit.py](tests/unit/test_main_unit.py), [tests/unit/test_app_pdf_unit.py](tests/unit/test_app_pdf_unit.py), [tests/integration/test_graph_integration.py](tests/integration/test_graph_integration.py).
  - Updated planner edge-case expectation in [tests/unit/test_main_unit.py](tests/unit/test_main_unit.py#L33-L46).
  - Removed unused imports in [tests/test_agent.py](tests/test_agent.py#L21-L22).

## 6) Final Test Status

- Latest full-suite result: **49 passed**.
- Evidence command used repeatedly in this audit: `python -m pytest -q`.
- Test framework dependencies present: [requirements.txt](requirements.txt#L11-L12).

## 7) Risk Assessment

- **Residual external dependency risk (Major)**
  - Runtime depends on external model/search/web content availability and quality:
    - LLM calls in [main.py](main.py#L86), [main.py](main.py#L190), [main.py](main.py#L224)
    - Search calls in [main.py](main.py#L131-L134)
    - Scraping in [main.py](main.py#L153-L161)
- **Residual partial-output risk (Minor)**
  - Search/scrape exceptions continue execution with fallback summaries: [main.py](main.py#L137), [main.py](main.py#L166-L167), [main.py](main.py#L193-L196).
- **Residual content truncation risk (Minor)**
  - Per-source text capped to 15000 chars: [main.py](main.py#L158-L161).
- **Overall stability status**
  - No current failing tests; latest phase validation showed consecutive all-pass runs.
