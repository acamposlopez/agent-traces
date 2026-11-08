"""Claude Code adapter: conversion to the canonical model, relations, errors and unknown records."""

from __future__ import annotations

from agent_traces.model import (
    ContentKind,
    ErrorEvent,
    ErrorScope,
    FieldState,
    Message,
    Notice,
    Reasoning,
    ResultStatus,
    Role,
    ToolCall,
    ToolCategory,
    ToolResult,
    Unparseable,
    Unrecognized,
    UnrecognizedReason,
    state_of,
)
from conftest import cc_record, trace_of


def kinds(events):
    return [type(event).__name__ for event in events]


def test_simple_session():
    trace = trace_of("claude-code/simple.jsonl")
    assert kinds(trace.events) == ["Notice", "Message", "Reasoning", "Message", "Message", "Message"]
    snapshot, first_user, reasoning, answer, follow_up, example = trace.events
    assert snapshot.notice_type == "file-history-snapshot"
    assert first_user.role is Role.USER and isinstance(first_user.content, str)
    assert answer.role is Role.ASSISTANT and answer.model == "claude-sonnet-4-5"
    # One API message split over two lines: the reasoning block and the text share message.id.
    assert reasoning.message_id == answer.message_id == "msg_c1a_01"
    assert follow_up.role is Role.USER and example.message_id == "msg_c1a_02"
    meta = trace.metadata
    assert meta.session_ids.values == ["c1a00000-5e55-4000-8000-000000000001"]
    assert meta.agent_versions.values == ["2.1.150"]
    assert meta.models.values == ["claude-sonnet-4-5"]
    assert [cwd.startswith("/home/") for cwd in meta.context["cwd"]] == [True]


def test_reasoning_content_is_never_stored():
    assert "content" not in Reasoning.__dataclass_fields__
    trace = trace_of("claude-code/secrets.jsonl")
    reasoning = [e for e in trace.events if isinstance(e, Reasoning)]
    assert len(reasoning) == 1 and not reasoning[0].redacted


def test_order_and_parent_links_are_preserved():
    trace = trace_of("claude-code/tool-calls.jsonl")
    assert [e.seq for e in trace.events] == list(range(len(trace.events)))
    lines = [e.source.line for e in trace.events]
    assert lines == sorted(lines)
    for previous, event in zip(trace.events, trace.events[1:]):
        assert event.parent_id == previous.event_id  # each line points to the one before it


def test_tool_calls_pair_with_results():
    trace = trace_of("claude-code/tool-calls.jsonl")
    calls = [e for e in trace.events if isinstance(e, ToolCall)]
    results = [e for e in trace.events if isinstance(e, ToolResult)]
    assert [c.call_id for c in calls] == [r.call_id for r in results] == ["toolu_canary_c2b_01", "toolu_canary_c2b_02"]
    assert [c.category for c in calls] == [ToolCategory.FILE_READ, ToolCategory.FILE_WRITE]
    assert [(r.category, r.tool_name, r.status) for r in results] == [
        (ToolCategory.FILE_READ, "Read", ResultStatus.SUCCESS),
        (ToolCategory.FILE_WRITE, "Write", ResultStatus.SUCCESS),
    ]
    assert results[0].details["type"] == "text"  # structured toolUseResult kept for later stages
    assert calls[0].arguments["file_path"].endswith("README.md")


def test_errors_and_recovery():
    trace = trace_of("claude-code/errors-recovery.jsonl")
    results = [e for e in trace.events if isinstance(e, ToolResult)]
    assert [r.status for r in results] == [
        ResultStatus.ERROR,
        ResultStatus.SUCCESS,
        ResultStatus.SUCCESS,
        ResultStatus.SUCCESS,
    ]
    errors = [e for e in trace.events if isinstance(e, ErrorEvent)]
    assert [(e.scope, e.code) for e in errors] == [
        (ErrorScope.API, "api_error"),
        (ErrorScope.API, "api_error_message"),
    ]
    assert "<synthetic>" not in trace.metadata.models.values
    # The run recovers: the last tool call succeeds and the last event is an assistant message.
    assert isinstance(trace.events[-1], Message) and trace.events[-1].role is Role.ASSISTANT


def test_mcp_and_unknown_tools_are_categorised_without_trusting_names():
    trace = trace_of("claude-code/secrets.jsonl")
    calls = [e for e in trace.events if isinstance(e, ToolCall)]
    assert [c.category for c in calls] == [ToolCategory.SHELL, ToolCategory.MCP, ToolCategory.UNKNOWN]


def test_unsafe_values_are_tracked_but_kept_apart():
    meta = trace_of("claude-code/secrets.jsonl").metadata
    assert len(meta.models.values) == 2 and len(meta.agent_versions.values) == 2


def test_subagent_links():
    main = trace_of("claude-code/subagent-main.jsonl")
    call = next(e for e in main.events if isinstance(e, ToolCall))
    result = next(e for e in main.events if isinstance(e, ToolResult))
    assert call.category is ToolCategory.SUBAGENT and call.tool_name == "Agent"
    assert result.subagent_ref == "a5e0c0de"
    agent = trace_of("claude-code/subagent-agent.jsonl")
    assert all(e.is_sidechain and e.agent_id == "a5e0c0de" for e in agent.events)
    assert agent.metadata.session_ids.values == main.metadata.session_ids.values  # parent session id


def test_unknown_fields_records_blocks_and_subtypes():
    trace = trace_of("edge-cases/unknown-fields.jsonl")
    unrecognized = [e for e in trace.events if isinstance(e, Unrecognized)]
    assert [(e.reason, e.source.record_type) for e in unrecognized] == [
        (UnrecognizedReason.UNKNOWN_BLOCK, "assistant"),
        (UnrecognizedReason.UNKNOWN_SUBTYPE, "system"),
        (UnrecognizedReason.UNKNOWN_TYPE, None),  # unknown type names are never echoed
    ]
    assert unrecognized[0].source.block == 0
    notices = [e.notice_type for e in trace.events if isinstance(e, Notice)]
    assert notices == ["system/compact_boundary", "queue-operation/enqueue"]
    summary = [e for e in trace.events if isinstance(e, Message) and e.injected]
    assert len(summary) == 1 and summary[0].role is Role.USER


def test_unparseable_fields_are_distinguished_from_absent_ones(jsonl):
    path = jsonl(
        [
            cc_record("user", "u1", None, {"role": "user", "content": [{"type": "text", "text": 42}]},
                      timestamp="ayer"),
            cc_record("user", "u2", "u1", {"role": "user", "content": "hola"}, timestamp=None),
            cc_record("assistant", "a1", "u2", {"id": "m1", "role": "assistant", "content": [
                {"type": "tool_use", "id": 7, "name": "Read", "input": {}},
                {"type": "image", "source": {}},
            ]}),
        ]
    )
    events = trace_of(path).events
    assert isinstance(events[0].content, Unparseable) and isinstance(events[0].timestamp, Unparseable)
    assert events[1].timestamp is None
    assert isinstance(events[2].call_id, Unparseable)
    assert isinstance(events[3], Unrecognized)  # images are only expected in user content
    assert [state_of(e.timestamp) for e in events[:2]] == [FieldState.UNPARSEABLE, FieldState.ABSENT]


def test_user_content_kinds_and_incomplete_records(jsonl):
    path = jsonl(
        [
            cc_record("user", "u1", None, {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                {"type": "document", "source": {}},
            ]}),
            cc_record("user", "u2", "u1", None),
            cc_record("assistant", "a1", "u2", {"role": "assistant", "content": []}),
        ]
    )
    events = trace_of(path, fmt="claude-code").events
    assert [e.content_kind for e in events[:2]] == [ContentKind.IMAGE, ContentKind.DOCUMENT]
    assert [e.reason for e in events[2:]] == [UnrecognizedReason.INCOMPLETE, UnrecognizedReason.INCOMPLETE]


def test_tool_result_with_unexpected_is_error_value(jsonl):
    path = jsonl(
        [
            cc_record("assistant", "a1", None, {"id": "m1", "role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}),
            cc_record("user", "u1", "a1", {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x", "is_error": "maybe"}]}),
        ]
    )
    result = trace_of(path).events[1]
    assert result.status is ResultStatus.UNKNOWN and result.category is ToolCategory.SHELL


# Phase 1.6: signatures, origins and metadata classification ----------------------------------------------------

from agent_traces.adapters.base import Signal  # noqa: E402
from agent_traces.adapters.claude_code import ClaudeCodeAdapter  # noqa: E402
from agent_traces.model import ExportPolicy, MessageOrigin, MetadataStructure, export_policy  # noqa: E402


def test_sniff_uses_positive_signatures_only():
    adapter = ClaudeCodeAdapter()
    assert {member.name for member in Signal} == {"NONE", "WEAK", "STRONG"}
    record = cc_record("user", "u1", None, {"role": "user", "content": "x"})
    assert adapter.sniff(record) is Signal.STRONG
    assert adapter.sniff({**record, "session_id": "s"}) is Signal.STRONG  # an extra snake_case key is neutral
    assert adapter.sniff({"session_id": "s", "type": "result"}) is Signal.WEAK
    assert adapter.sniff({"session_id": "s"}) is Signal.NONE


def test_user_message_origins(jsonl):
    def user(uuid, content, **extra):
        return cc_record("user", uuid, None, {"role": "user", "content": content}, **extra)

    events = trace_of(jsonl([
        user("u1", "  please fix the bug"),
        user("u2", "<local-command-caveat>Caveat</local-command-caveat>", isMeta=True),
        user("u3", "<command-name>/model</command-name>"),
        user("u4", "<bash-stdout>ok</bash-stdout>"),
        user("u5", "[Request interrupted by user]"),
        user("u6", "summary of the earlier conversation", isCompactSummary=True),
        user("u7", "skill body", sourceToolUseID="toolu_1"),
    ])).events
    assert [e.origin for e in events] == [
        MessageOrigin.PROMPT,
        MessageOrigin.INJECTED,
        MessageOrigin.COMMAND,
        MessageOrigin.COMMAND_OUTPUT,
        MessageOrigin.INTERRUPTION,
        MessageOrigin.INJECTED,
        MessageOrigin.INJECTED,
    ]
    assert [export_policy(e) for e in events].count(ExportPolicy.NOT_BY_DEFAULT) == 4


def test_text_travelling_with_a_tool_result_is_never_a_prompt(jsonl):
    record = cc_record("user", "u1", None, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        {"type": "text", "text": "looks like a human sentence"},
    ]})
    events = trace_of(jsonl([record])).events
    assert isinstance(events[0], ToolResult) and events[1].origin is MessageOrigin.INJECTED


def test_metadata_records_are_classified_without_overclaiming(jsonl):
    events = trace_of(jsonl([
        cc_record("user", "u1", None, {"role": "user", "content": "hi"}),
        {"type": "progress", "data": {"type": "hook_progress"}, "toolUseID": "t1", "parentToolUseID": "t1"},
        {"type": "progress", "data": {"type": "never-heard-of"}},
        {"type": "progress"},
        {"type": "attachment", "attachment": {"type": "queued_command"}},
        {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-03-14T10:00:00Z"},
        {"type": "custom-title"},
        {"type": "last-prompt"},
        cc_record("system", "s1", "u1", None, subtype="compact_boundary", compactMetadata={"trigger": "auto"}),
        cc_record("system", "s2", "s1", None, subtype="turn_duration"),
    ]), fmt="claude-code").events[1:]
    described = [(e.notice_type, e.metadata_class, e.structure) for e in events]
    assert described == [
        ("progress/hook_progress", "progress", MetadataStructure.VERIFIED),
        ("progress/other", "progress", MetadataStructure.NAME_ONLY),  # unknown subtype: not verified
        ("progress", "progress", MetadataStructure.NAME_ONLY),  # no data: structure not present
        ("attachment/queued_command", "attachment", MetadataStructure.VERIFIED),
        ("queue-operation/enqueue", "queue", MetadataStructure.VERIFIED),
        ("custom-title", "title", MetadataStructure.NAME_ONLY),
        ("last-prompt", "prompt", MetadataStructure.NAME_ONLY),
        ("system/compact_boundary", "compaction", MetadataStructure.VERIFIED),
        ("system/turn_duration", "timing", MetadataStructure.NAME_ONLY),
    ]
    assert {export_policy(e) for e in events} == {ExportPolicy.NOT_BY_DEFAULT}
