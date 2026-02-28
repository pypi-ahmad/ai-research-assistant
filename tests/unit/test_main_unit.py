from unittest.mock import MagicMock, patch


def test_writer_node_returns_final_report_from_llm(monkeypatch):
    """Target: main.writer_node in main.py"""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import main as m

    mock_response = MagicMock()
    mock_response.content = "# Final Report\n\nContent"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    state = {
        "topic": "agentic systems",
        "plan": ["q1"],
        "current_query_index": 1,
        "summaries": ["summary one", "summary two"],
        "final_report": "",
    }

    with patch.object(m, "get_llm", return_value=mock_llm):
        result = m.writer_node(state)

    assert result == {"final_report": "# Final Report\n\nContent"}
    called_messages = mock_llm.invoke.call_args.args[0]
    assert any("agentic systems" in msg.content for msg in called_messages)
    assert any("summary one" in msg.content for msg in called_messages)
    assert any("summary two" in msg.content for msg in called_messages)


def test_planner_node_drops_empty_items_after_cleaning(monkeypatch):
    """Target: main.planner_node in main.py (sanitization + filtering)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import main as m

    mock_response = MagicMock()
    mock_response.content = "*\nquery two\nquery three"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch.object(m, "get_llm", return_value=mock_llm):
        result = m.planner_node({"topic": "test topic"})

    assert result["plan"] == ["query two", "query three"]
