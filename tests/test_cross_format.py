"""Equivalent sessions from different agents must produce the same semantic sequence of common events."""

from __future__ import annotations

from agent_traces.model import semantic_view
from conftest import report_of, trace_of

EXPECTED = [
    ("message", "user", "text"),
    ("message", "assistant", "text"),
    ("tool_call", 1, "file_read"),
    ("tool_result", 1, "file_read", "success"),
    ("message", "assistant", "text"),
    ("tool_call", 2, "file_write"),
    ("tool_result", 2, "file_write", "success"),
    ("message", "assistant", "text"),
]


def test_claude_code_and_cursor_cli_tool_sessions_are_semantically_equal():
    claude = semantic_view(trace_of("claude-code/tool-calls.jsonl").events)
    cursor = semantic_view(trace_of("cursor-cli/tool-calls.jsonl").events)
    assert claude == cursor == EXPECTED


def test_equivalent_sessions_have_equal_core_counts():
    claude = report_of("claude-code/tool-calls.jsonl")
    cursor = report_of("cursor-cli/tool-calls.jsonl")
    for section, key in [
        ("counts", "messages"),
        ("counts", "tool_calls"),
        ("counts", "tool_results"),
        ("counts", "errors"),
        ("tools", "categories"),
        ("relations", "paired_calls"),
    ]:
        assert getattr(claude, section)[key] == getattr(cursor, section)[key], key


def test_equal_outcomes_keep_their_different_evidence():
    """Same successes, different evidence: Claude Code omits is_error (protocol default), Cursor states it."""
    claude = report_of("claude-code/tool-calls.jsonl").counts["result_outcomes"]
    cursor = report_of("cursor-cli/tool-calls.jsonl").counts["result_outcomes"]
    total = lambda outcomes: outcomes["success_confirmed"] + outcomes["success_protocol_default"]  # noqa: E731
    assert total(claude) == total(cursor) == 2
    assert (claude["success_protocol_default"], cursor["success_confirmed"]) == (2, 2)
