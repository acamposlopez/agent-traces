"""Cursor CLI stream-json adapter: documented events, call correlation and partial-output dedup."""

from __future__ import annotations

from agent_traces.model import (
    Message,
    ResultStatus,
    Role,
    SessionMarker,
    SessionPhase,
    TerminalState,
    ToolCall,
    ToolCategory,
    ToolResult,
    Unrecognized,
    UnrecognizedReason,
)
from conftest import trace_of

SESSION = {"session_id": "s-1"}


def assistant(text):
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    return {"type": "assistant", "message": message, **SESSION}


def test_simple_session():
    trace = trace_of("cursor-cli/simple.jsonl")
    start, user, answer, end = trace.events
    assert isinstance(start, SessionMarker) and start.phase is SessionPhase.START
    assert (user.role, answer.role) == (Role.USER, Role.ASSISTANT)
    assert isinstance(end, SessionMarker) and end.status is ResultStatus.SUCCESS
    assert trace.metadata.terminal_state is TerminalState.COMPLETED
    assert trace.metadata.models.values == ["Claude 4 Sonnet"]
    assert trace.metadata.session_ids.values == ["c0a10000-5e55-4000-8000-000000000001"]


def test_started_and_completed_events_are_correlated_by_call_id():
    trace = trace_of("cursor-cli/tool-calls.jsonl")
    calls = [e for e in trace.events if isinstance(e, ToolCall)]
    results = [e for e in trace.events if isinstance(e, ToolResult)]
    assert [c.call_id for c in calls] == [r.call_id for r in results]
    assert [(c.tool_name, c.category) for c in calls] == [
        ("readToolCall", ToolCategory.FILE_READ),
        ("writeToolCall", ToolCategory.FILE_WRITE),
    ]
    assert all(r.status is ResultStatus.SUCCESS for r in results)
    assert results[1].output["linesCreated"] == 1
    assert calls[0].arguments == {"path": "README.md"}
    # The terminal result repeats the assistant text: it is kept apart, not turned into a message.
    assert sum(isinstance(e, Message) for e in trace.events) == 4


def test_failed_run():
    trace = trace_of("cursor-cli/failed-run.jsonl")
    calls = [e for e in trace.events if isinstance(e, ToolCall)]
    results = [e for e in trace.events if isinstance(e, ToolResult)]
    assert calls[0].category is ToolCategory.UNKNOWN and calls[0].tool_name == "run_terminal_command"
    assert isinstance(calls[0].arguments, str)  # function arguments are documented as a string
    # Undocumented non-success shape: outcome unknown, never reinterpreted as an error.
    assert len(results) == 1 and results[0].status is ResultStatus.UNKNOWN
    assert calls[1].call_id not in {r.call_id for r in results}
    assert trace.metadata.terminal_state is TerminalState.MISSING


def test_partial_output_deltas_are_merged_and_flushes_skipped():
    trace = trace_of("cursor-cli/sensitive-data.jsonl")
    assistant = [e for e in trace.events if isinstance(e, Message) and e.role is Role.ASSISTANT]
    assert [m.content for m in assistant] == ["I'll write the .env file.", "Done. CANARY secret stored."]
    assert [(m.source.line, m.source.line_end) for m in assistant] == [(3, 4), (11, 12)]
    assert assistant[0].timestamp is not None
    assert dict(trace.metadata.normalization) == {
        "cursor_partial_flush_before_tool": 1,
        "cursor_partial_final_flush": 1,
    }


def test_complete_messages_without_partial_output_are_kept(jsonl):
    path = jsonl(
        [
            assistant("a"),
            assistant("b"),
        ]
    )
    events = trace_of(path, fmt="cursor-cli").events
    assert [e.content for e in events] == ["a", "b"]


def test_undocumented_assistant_variant_is_not_interpreted(jsonl):
    path = jsonl([{**assistant("x"), "model_call_id": "m"}])
    (event,) = trace_of(path, fmt="cursor-cli").events
    assert isinstance(event, Unrecognized) and event.reason is UnrecognizedReason.UNKNOWN_SUBTYPE


def test_result_with_is_error_marks_the_run_as_failed(jsonl):
    path = jsonl([{"type": "result", "subtype": "error", "is_error": True, "result": "x", **SESSION}])
    trace = trace_of(path, fmt="cursor-cli")
    assert trace.events[0].status is ResultStatus.ERROR
    assert trace.metadata.terminal_state is TerminalState.FAILED


def test_terminal_state_follows_the_last_run(jsonl):
    init = {"type": "system", "subtype": "init", "apiKeySource": "env", "model": "GPT-5", **SESSION}
    done = {"type": "result", "subtype": "success", "is_error": False, **SESSION}
    assert trace_of(jsonl([init, done]), fmt="cursor-cli").metadata.terminal_state is TerminalState.COMPLETED
    assert trace_of(jsonl([init, done, init]), fmt="cursor-cli").metadata.terminal_state is TerminalState.MISSING


def test_incomplete_records():
    trace = trace_of("edge-cases/incomplete-records.jsonl")
    unrecognized = [e for e in trace.events if isinstance(e, Unrecognized)]
    assert [e.reason for e in unrecognized] == [
        UnrecognizedReason.INCOMPLETE,  # user without message
        UnrecognizedReason.INCOMPLETE,  # message that is not an object
        UnrecognizedReason.INCOMPLETE,  # non-text content block (only text is documented)
        UnrecognizedReason.INCOMPLETE,  # tool_call that is not an object
        UnrecognizedReason.UNKNOWN_SUBTYPE,  # tool_call subtype
        UnrecognizedReason.UNKNOWN_SUBTYPE,  # system subtype
    ]
    call = next(e for e in trace.events if isinstance(e, ToolCall))
    assert call.call_id is None


def test_unknown_event_types_are_kept_as_unrecognized():
    trace = trace_of("cursor-cli/sensitive-data.jsonl")
    unknown = [e for e in trace.events if isinstance(e, Unrecognized)]
    assert len(unknown) == 1 and unknown[0].reason is UnrecognizedReason.UNKNOWN_TYPE
    assert unknown[0].source.record_type is None
    vault = [e for e in trace.events if isinstance(e, ToolCall) and e.category is ToolCategory.UNKNOWN]
    assert len(vault) == 1
