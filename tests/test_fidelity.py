"""Fidelity of the canonical model: relations come only from the source, technical metadata is preserved
and documented defaults are never presented as facts."""

from __future__ import annotations

import json

import pytest

from agent_traces.adapters.base import native_id
from agent_traces.engine import load_trace
from agent_traces.model import (
    KNOWN_ATTRIBUTES,
    ErrorEvent,
    Message,
    Notice,
    SessionMarker,
    SessionPhase,
    StatusBasis,
    ToolCall,
    ToolResult,
    Unrecognized,
)
from conftest import CLAUDE_FIXTURES, CURSOR_FIXTURES, EDGE_FIXTURES, fixture, trace_of


def _reject(name):
    raise ValueError(name)


def source_records(path) -> dict[int, dict]:
    """Parsed source objects by line number (same numbering and JSON rules as the reader)."""
    records = {}
    for line_no, line in enumerate(path.read_bytes().split(b"\n"), start=1):
        try:
            value = json.loads(line, parse_constant=_reject)
        except ValueError:
            continue
        if isinstance(value, dict):
            records[line_no] = value
    return records


CLAUDE_FILES = CLAUDE_FIXTURES + [fixture("edge-cases/invalid-json.jsonl"), fixture("edge-cases/unknown-fields.jsonl")]
CURSOR_FILES = CURSOR_FIXTURES + [fixture("edge-cases/incomplete-records.jsonl")]


@pytest.mark.parametrize("path", CLAUDE_FILES, ids=lambda p: p.name)
def test_claude_code_relations_come_only_from_the_source(path):
    records = source_records(path)
    for event in trace_of(path, fmt="claude-code").events:
        record = records.get(event.source.line)
        if record is None:  # reader-level problem (malformed line): nothing to compare
            assert isinstance(event, Unrecognized) and event.raw is None
            continue
        assert event.event_id == native_id(record.get("uuid"))
        assert event.parent_id == native_id(record.get("parentUuid"))
        assert event.agent_id == native_id(record.get("agentId"))
        sidechain = record.get("isSidechain")
        assert event.is_sidechain == (sidechain if isinstance(sidechain, bool) else None)
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        if record.get("type") == "assistant" and message:
            assert event.message_id == native_id(message.get("id"))
        else:
            assert event.message_id is None
        if isinstance(event, (ToolCall, ToolResult)):
            block = message["content"][event.source.block]
            assert event.call_id == native_id(block["id" if isinstance(event, ToolCall) else "tool_use_id"])


@pytest.mark.parametrize("path", CURSOR_FILES, ids=lambda p: p.name)
def test_cursor_cli_relations_come_only_from_the_source(path):
    records = source_records(path)
    for event in trace_of(path, fmt="cursor-cli").events:
        # The format has no record, parent, message or agent ids: none may appear.
        assert (event.event_id, event.parent_id, event.message_id, event.agent_id, event.is_sidechain) == (
            None, None, None, None, None,
        )
        if isinstance(event, (ToolCall, ToolResult)):
            assert event.call_id == native_id(records[event.source.line].get("call_id"))


@pytest.mark.parametrize("path", CLAUDE_FIXTURES + CURSOR_FIXTURES + EDGE_FIXTURES, ids=lambda p: p.name)
def test_attributes_use_known_names_only(path):
    fmt = {"ambiguous-mixed.jsonl": "cursor-cli", "unknown-format.jsonl": "claude-code",
           "claude-agent-sdk-stream.jsonl": "cursor-cli"}.get(path.name, "auto")
    trace = load_trace(path, fmt=fmt)  # None for files without records
    for event in trace.events if trace else []:
        assert set(event.attributes) <= KNOWN_ATTRIBUTES


def test_claude_code_technical_metadata_is_preserved_verbatim():
    path = fixture("claude-code/tool-calls.jsonl")
    records = source_records(path)
    assistant = [e for e in trace_of(path).events if e.source.record_type == "assistant"]
    assert assistant
    for event in assistant:
        record = records[event.source.line]
        assert event.attributes == {
            "request_id": record["requestId"],
            "stop_reason": record["message"]["stop_reason"],
            "usage": record["message"]["usage"],
        }
    user = [e for e in trace_of(path).events if e.source.record_type == "user"]
    assert all(e.attributes == {} for e in user)  # nothing is filled in when the source has nothing


def test_compaction_relation_and_flags_are_kept():
    path = fixture("edge-cases/unknown-fields.jsonl")
    records = source_records(path)
    events = trace_of(path).events
    boundary = next(e for e in events if isinstance(e, Notice) and e.notice_type == "system/compact_boundary")
    assert boundary.parent_id is None  # the source sets parentUuid to null here...
    assert boundary.attributes["logical_parent_id"] == records[boundary.source.line]["logicalParentUuid"]
    summary = next(e for e in events if isinstance(e, Message) and e.injected)
    assert summary.attributes == {"is_compact_summary": True}


def test_cursor_result_metadata_is_preserved_verbatim():
    path = fixture("cursor-cli/tool-calls.jsonl")
    records = source_records(path)
    end = next(e for e in trace_of(path).events if isinstance(e, SessionMarker) and e.phase is SessionPhase.END)
    record = records[end.source.line]
    assert end.attributes == {key: record[key] for key in ("duration_ms", "duration_api_ms", "request_id")}


def test_result_status_basis_separates_facts_from_defaults():
    claude = [e for e in trace_of("claude-code/errors-recovery.jsonl").events if isinstance(e, ToolResult)]
    assert [e.status_basis for e in claude] == [
        StatusBasis.EXPLICIT,  # is_error: true
        StatusBasis.EXPLICIT,  # is_error: false
        StatusBasis.PROTOCOL_DEFAULT,  # is_error absent: Anthropic API default
        StatusBasis.EXPLICIT,
    ]
    cursor_ok = [e for e in trace_of("cursor-cli/tool-calls.jsonl").events if isinstance(e, ToolResult)]
    assert {e.status_basis for e in cursor_ok} == {StatusBasis.EXPLICIT}
    cursor_failed = [e for e in trace_of("cursor-cli/failed-run.jsonl").events if isinstance(e, ToolResult)]
    assert [e.status_basis for e in cursor_failed] == [StatusBasis.UNDETERMINED]


def test_no_event_is_created_without_a_source_record():
    """Every event points to a real line of the file; aggregated events point to their first and last line."""
    for path in CLAUDE_FIXTURES + CURSOR_FIXTURES:
        lines = path.read_bytes().count(b"\n")
        for event in trace_of(path).events:
            assert 1 <= event.source.line <= lines
            if event.source.line_end is not None:
                assert event.source.line < event.source.line_end <= lines
            assert not isinstance(event, ErrorEvent) or event.code in {"api_error", "api_error_message"}
