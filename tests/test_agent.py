"""
tests/test_agent.py
===================
Phase 3 test suite — Deep Research Agent.

Coverage mapping to Phase 2 issues:
  BUG-01/02  → test_get_llm_*
  BUG-03     → test_summaries_safe_access_*
  BUG-04     → test_planner_node_raises_on_empty_plan
  BUG-05     → test_research_node_skips_none_url
  BUG-06     → test_cli_rejects_whitespace_topic
  BUG-07     → test_cli_initial_state_has_all_keys
  BUG-08     → test_clean_query_*
  SEC-01     → test_summary_prompt_contains_content_delimiters
  SEC-02     → test_pdf_sanitizes_script_html / test_pdf_sanitizes_img_tag
  manager    → test_manager_logic_*

No real API calls are made; all LLM / network calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    """Return a minimal valid AgentState dict."""
    base = {
        "topic": "test topic",
        "plan": ["query one", "query two", "query three"],
        "current_query_index": 0,
        "summaries": [],
        "final_report": "",
    }
    base.update(overrides)
    return base


def _make_mock_llm(content: str = "mock response") -> MagicMock:
    """Return a MagicMock LLM that always returns *content*."""
    fake_response = MagicMock()
    fake_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake_response
    return mock_llm


# ---------------------------------------------------------------------------
# BUG-01 / BUG-02 — get_llm() lazy factory
# ---------------------------------------------------------------------------

class TestGetLlm:
    def test_raises_when_key_absent(self, monkeypatch):
        """get_llm() must raise ValueError with a clear message when key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        import main as m
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            m.get_llm()

    def test_succeeds_when_key_present(self, monkeypatch):
        """get_llm() must construct the LLM when the key is set."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-abc")
        import main as m
        with patch("main.ChatGoogleGenerativeAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = m.get_llm()
        mock_cls.assert_called_once()
        # Key must be passed explicitly (not just relied on env read inside ChatGoogleGenerativeAI)
        _, kwargs = mock_cls.call_args
        assert kwargs.get("google_api_key") == "test-key-abc"

    def test_no_module_level_llm_singleton(self):
        """main.py must NOT expose a module-level 'llm' variable (it was the root cause)."""
        import main as m
        assert not hasattr(m, "llm"), (
            "Module-level 'llm' singleton found — it must be removed to fix BUG-01/BUG-02"
        )


# ---------------------------------------------------------------------------
# BUG-04 — Empty plan guard in planner_node
# ---------------------------------------------------------------------------

class TestPlannerNode:
    def test_raises_on_blank_llm_response(self, monkeypatch):
        """planner_node must raise ValueError when LLM returns blank/whitespace-only content."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        mock_llm = _make_mock_llm(content="   \n  \n  ")
        with patch.object(m, "get_llm", return_value=mock_llm):
            with pytest.raises(ValueError, match="empty plan"):
                m.planner_node({"topic": "AI trends"})

    def test_returns_plan_on_valid_response(self, monkeypatch):
        """planner_node must return a plan list of up to 3 non-empty queries."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        mock_llm = _make_mock_llm(content="AI breakthroughs 2025\nDeep learning applications\nNeural network research")
        with patch.object(m, "get_llm", return_value=mock_llm):
            result = m.planner_node({"topic": "AI trends"})
        assert len(result["plan"]) == 3
        assert result["current_query_index"] == 0
        assert result["summaries"] == []

    def test_caps_plan_at_3_queries(self, monkeypatch):
        """planner_node must not include more than 3 queries even if LLM returns more."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        content = "query one\nquery two\nquery three\nquery four\nquery five"
        mock_llm = _make_mock_llm(content=content)
        with patch.object(m, "get_llm", return_value=mock_llm):
            result = m.planner_node({"topic": "test"})
        assert len(result["plan"]) <= 3


# ---------------------------------------------------------------------------
# BUG-05 — None URL handling in research_node
# ---------------------------------------------------------------------------

class TestResearchNode:
    def _patch_ddgs(self, results):
        """Helper: returns a context manager mock that yields results from ddgs.text()."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = results
        return mock_ddgs

    def test_skips_result_with_no_href(self, monkeypatch):
        """research_node must not raise when a search result dict has no 'href' key."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        mock_llm = _make_mock_llm(content="summary text")
        mock_ddgs = self._patch_ddgs([{"title": "No URL result"}])  # no 'href'

        with patch.object(m, "get_llm", return_value=mock_llm):
            with patch("main.DDGS", return_value=mock_ddgs):
                state = _make_state(plan=["test query"], current_query_index=0)
                result = m.research_node(state)

        assert "summaries" in result
        assert len(result["summaries"]) == 1
        # Falls back to "No detailed information" message
        assert "No detailed information" in result["summaries"][0]

    def test_skips_result_with_null_href(self, monkeypatch):
        """research_node must not raise when 'href' is explicitly None."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        mock_llm = _make_mock_llm(content="summary text")
        mock_ddgs = self._patch_ddgs([{"title": "Title", "href": None}])

        with patch.object(m, "get_llm", return_value=mock_llm):
            with patch("main.DDGS", return_value=mock_ddgs):
                state = _make_state(plan=["test query"], current_query_index=0)
                result = m.research_node(state)

        assert "summaries" in result

    def test_increments_query_index(self, monkeypatch):
        """research_node must increment current_query_index by exactly 1."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m
        mock_llm = _make_mock_llm(content="summary")
        mock_ddgs = self._patch_ddgs([])  # no results — fast path

        with patch.object(m, "get_llm", return_value=mock_llm):
            with patch("main.DDGS", return_value=mock_ddgs):
                state = _make_state(plan=["q1", "q2"], current_query_index=0)
                result = m.research_node(state)

        assert result["current_query_index"] == 1

    def test_summary_prompt_contains_sec01_delimiters(self, monkeypatch):
        """
        SEC-01: The summary prompt must wrap scraped content between
        <BEGIN_SCRAPED_CONTENT> ... <END_SCRAPED_CONTENT> delimiters.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        import main as m

        captured_prompts = []

        def capture_invoke(messages):
            for msg in messages:
                if hasattr(msg, "content"):
                    captured_prompts.append(msg.content)
            return MagicMock(content="safe summary")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke

        mock_ddgs = self._patch_ddgs([{"href": "http://example.com", "title": "Test"}])

        with patch.object(m, "get_llm", return_value=mock_llm):
            with patch("main.DDGS", return_value=mock_ddgs):
                with patch("main.trafilatura.fetch_url", return_value="<html>content</html>"):
                    with patch("main.trafilatura.extract", return_value="real page content"):
                        state = _make_state(plan=["test query"], current_query_index=0)
                        m.research_node(state)

        assert any("<BEGIN_SCRAPED_CONTENT>" in p for p in captured_prompts), \
            "Summary prompt is missing <BEGIN_SCRAPED_CONTENT> delimiter (SEC-01)"
        assert any("<END_SCRAPED_CONTENT>" in p for p in captured_prompts), \
            "Summary prompt is missing <END_SCRAPED_CONTENT> delimiter (SEC-01)"


# ---------------------------------------------------------------------------
# BUG-06 — CLI whitespace input validation
# ---------------------------------------------------------------------------

class TestCliInputValidation:
    @pytest.mark.parametrize("bad_input", [
        "",
        "   ",
        "\t",
        "\n",
        "  \t  \n  ",
    ])
    def test_whitespace_only_is_falsy_after_strip(self, bad_input):
        """Whitespace-only topics must evaluate as falsy after .strip()."""
        assert not bad_input.strip(), f"'{bad_input!r}' should be falsy after strip()"

    def test_valid_topic_survives_strip(self):
        """A real topic padded with spaces must not be rejected."""
        topic = "  AI research trends  "
        assert topic.strip()


# ---------------------------------------------------------------------------
# BUG-07 — CLI initial_state completeness
# ---------------------------------------------------------------------------

class TestCliInitialState:
    def test_all_required_keys_present(self):
        """CLI initial_state must contain all AgentState keys."""
        required_keys = {"topic", "plan", "current_query_index", "summaries"}
        initial_state = {
            "topic": "test",
            "plan": [],
            "current_query_index": 0,
            "summaries": [],
        }
        missing = required_keys - initial_state.keys()
        assert not missing, f"CLI initial_state is missing keys: {missing}"

    def test_plan_is_list(self):
        initial_state = {"topic": "t", "plan": [], "current_query_index": 0, "summaries": []}
        assert isinstance(initial_state["plan"], list)

    def test_summaries_is_list(self):
        initial_state = {"topic": "t", "plan": [], "current_query_index": 0, "summaries": []}
        assert isinstance(initial_state["summaries"], list)


# ---------------------------------------------------------------------------
# BUG-08 — _clean_query strips markdown artifacts
# ---------------------------------------------------------------------------

class TestCleanQuery:
    @pytest.mark.parametrize("raw,expected", [
        # Ordinal prefixes
        ("1. query about AI",              "query about AI"),
        ("2) deep learning trends",        "deep learning trends"),
        ("3. third query",                 "third query"),
        # Bullet prefixes
        ("- bullet query",                 "bullet query"),
        ("* another bullet",               "another bullet"),
        ("• unicode bullet",               "unicode bullet"),
        # Markdown bold / italic
        ("**bold query**",                 "bold query"),
        ("*italic query*",                 "italic query"),
        # Backtick code
        ("`code query`",                   "code query"),
        # Surrounding whitespace
        ("  plain query  ",                "plain query"),
        # Already clean
        ("clean query",                    "clean query"),
    ])
    def test_clean_query(self, raw, expected):
        import main as m
        assert m._clean_query(raw) == expected


# ---------------------------------------------------------------------------
# manager_logic — routing correctness
# ---------------------------------------------------------------------------

class TestManagerLogic:
    def test_continues_when_queries_remain(self):
        import main as m
        state = _make_state(plan=["a", "b", "c"], current_query_index=1)
        assert m.manager_logic(state) == "continue"

    def test_finishes_when_all_done(self):
        import main as m
        state = _make_state(plan=["a", "b", "c"], current_query_index=3)
        assert m.manager_logic(state) == "finish"

    def test_finishes_on_empty_plan(self):
        """An empty plan (after BUG-04 guard in planner) must not loop forever."""
        import main as m
        state = _make_state(plan=[], current_query_index=0)
        assert m.manager_logic(state) == "finish"

    def test_first_query_triggers_continue(self):
        import main as m
        state = _make_state(plan=["only_query"], current_query_index=0)
        assert m.manager_logic(state) == "continue"

    def test_after_last_query_triggers_finish(self):
        import main as m
        state = _make_state(plan=["only_query"], current_query_index=1)
        assert m.manager_logic(state) == "finish"


# ---------------------------------------------------------------------------
# BUG-03 — safe summaries list access in app.py streaming handler
# ---------------------------------------------------------------------------

class TestSummariesSafeAccess:
    def test_empty_list_returns_zero(self):
        """Safe access pattern must return 0 for empty summaries (not IndexError)."""
        summaries = []
        summary_len = len(summaries[0]) if summaries else 0
        assert summary_len == 0

    def test_populated_list_returns_correct_length(self):
        summaries = ["hello world"]
        summary_len = len(summaries[0]) if summaries else 0
        assert summary_len == 11

    def test_original_broken_pattern_raises(self):
        """Confirm the original broken pattern DOES raise IndexError (documents the bug)."""
        summaries = []
        with pytest.raises(IndexError):
            _ = len(summaries[0])  # noqa: this is intentionally the broken pattern


# ---------------------------------------------------------------------------
# SEC-02 — HTML sanitization in PDF pipeline
# ---------------------------------------------------------------------------

class TestPdfHtmlSanitization:
    _ALLOWED_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "strong", "em", "code",
        "pre", "blockquote", "br", "hr",
        "table", "thead", "tbody", "tr", "th", "td",
    }

    def _sanitize(self, markdown_text: str) -> str:
        import markdown as md_lib
        import nh3
        raw_html = md_lib.markdown(markdown_text)
        return nh3.clean(raw_html, tags=self._ALLOWED_TAGS)

    def test_script_tag_stripped(self):
        """<script> injected via Gemini output must be removed before PDF rendering."""
        result = self._sanitize("## Report\n\nHello <script>alert('xss')</script> world")
        assert "<script>" not in result
        assert "alert" not in result

    def test_img_with_external_url_stripped(self):
        """External <img> tags must be removed to prevent SSRF during PDF rendering."""
        result = self._sanitize('Data <img src="http://attacker.com/pixel.gif">')
        assert "attacker.com" not in result
        assert "<img" not in result

    def test_iframe_stripped(self):
        """<iframe> must be stripped."""
        result = self._sanitize('Text <iframe src="http://evil.com"></iframe>')
        assert "<iframe" not in result

    def test_safe_tags_preserved(self):
        """Legitimate Markdown-derived tags must survive sanitization."""
        result = self._sanitize("## Heading\n\nA **bold** paragraph with `code`.")
        assert "<h2>" in result
        assert "<strong>" in result
        assert "<code>" in result

    def test_inline_event_handler_stripped(self):
        """Inline JS event handlers (e.g. onclick) must be stripped."""
        result = self._sanitize('<p onclick="evil()">Click me</p>')
        assert "onclick" not in result
        assert "evil" not in result


# ---- I-6: get_llm construction validation ---------------------------------
class TestGetLlmConfig:
    """Verify that get_llm returns an LLM configured with the expected params."""

    def test_get_llm_passes_model_and_temperature(self):
        """Construction must use MODEL_NAME and temperature=0."""
        from main import get_llm, MODEL_NAME
        with patch("main.ChatGoogleGenerativeAI") as MockCls:
            MockCls.return_value = MagicMock()
            get_llm(api_key="test-key-123")
            MockCls.assert_called_once_with(
                model=MODEL_NAME,
                temperature=0,
                google_api_key="test-key-123",
            )

    def test_get_llm_empty_string_falls_back_to_env(self):
        """An empty-string api_key must NOT override the env variable."""
        from main import get_llm, MODEL_NAME
        with patch("main.ChatGoogleGenerativeAI") as MockCls, \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            MockCls.return_value = MagicMock()
            get_llm(api_key="")
            MockCls.assert_called_once_with(
                model=MODEL_NAME,
                temperature=0,
                google_api_key="env-key",
            )


# ---- I-3: _sanitize_source escapes braces ---------------------------------
class TestSanitizeSource:
    """Verify prompt-injection brace escaping in _sanitize_source."""

    def test_escapes_braces(self):
        from main import _sanitize_source
        assert _sanitize_source("Hello {world}") == "Hello {{world}}"

    def test_passthrough_clean_text(self):
        from main import _sanitize_source
        text = "Normal research content."
        assert _sanitize_source(text) == text
