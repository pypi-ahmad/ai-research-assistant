from unittest.mock import MagicMock, patch


def _llm_side_effect(messages):
    """Deterministic LLM behavior based on prompt content (targets main.py graph flow)."""
    text = "\n".join(getattr(m, "content", "") for m in messages)
    response = MagicMock()
    if "research planner" in text:
        response.content = "query one\nquery two\nquery three"
    elif "technical writer" in text:
        response.content = "# Integrated Report\n\nDone"
    else:
        response.content = "summary for query"
    return response


def _make_ddgs_with_urls():
    ddgs = MagicMock()
    ddgs.__enter__.return_value = ddgs
    ddgs.__exit__.return_value = False
    ddgs.text.return_value = [
        {"title": "A", "href": "https://example.com/a"},
        {"title": "B", "href": "https://example.com/b"},
    ]
    return ddgs


def test_compiled_graph_invoke_happy_path(monkeypatch):
    """Target: main.app compiled graph in main.py."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import main as m

    llm = MagicMock()
    llm.invoke.side_effect = _llm_side_effect

    with patch.object(m, "get_llm", return_value=llm):
        with patch("main.DDGS", return_value=_make_ddgs_with_urls()):
            with patch("main.trafilatura.fetch_url", return_value="<html>ok</html>"):
                with patch("main.trafilatura.extract", return_value="extracted content"):
                    result = m.app.invoke(
                        {
                            "topic": "integration topic",
                            "plan": [],
                            "current_query_index": 0,
                            "summaries": [],
                        }
                    )

    assert "final_report" in result
    assert result["final_report"].startswith("# Integrated Report")


def test_compiled_graph_bubbles_planner_failure(monkeypatch):
    """Target: main.app compiled graph in main.py (failure scenario)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import main as m

    planner_blank_llm = MagicMock()
    planner_blank_llm.invoke.return_value = MagicMock(content="  \n  ")

    with patch.object(m, "get_llm", return_value=planner_blank_llm):
        try:
            m.app.invoke(
                {
                    "topic": "integration topic",
                    "plan": [],
                    "current_query_index": 0,
                    "summaries": [],
                }
            )
            assert False, "Expected ValueError from planner_node for empty plan"
        except ValueError as exc:
            assert "empty plan" in str(exc)
