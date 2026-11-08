"""Format detection: decide from the whole set of structural signals, never from one key, and never guess."""

from __future__ import annotations

import pytest

from agent_traces.adapters import DEFAULT_ADAPTERS, KNOWN_UNSUPPORTED
from agent_traces.detection import DetectionStatus, detect
from agent_traces.engine import FormatError, resolve_format
from agent_traces.reader import Limits
from conftest import CLAUDE_FIXTURES, CURSOR_FIXTURES, cc_record, fixture


def detection_of(path, limits=Limits()):
    return detect(path, DEFAULT_ADAPTERS, limits, KNOWN_UNSUPPORTED)


def cursor_call(call_id="c1", **extra):
    return {"type": "tool_call", "subtype": "started", "call_id": call_id, "session_id": "s",
            "tool_call": {"readToolCall": {"args": {"path": "a"}}}, **extra}


@pytest.mark.parametrize("path", CLAUDE_FIXTURES, ids=lambda p: p.name)
def test_claude_code_fixtures_are_detected(path):
    result = detection_of(path)
    assert result.status is DetectionStatus.DETECTED and result.adapter.name == "claude-code"


@pytest.mark.parametrize("path", CURSOR_FIXTURES, ids=lambda p: p.name)
def test_cursor_cli_fixtures_are_detected(path):
    result = detection_of(path)
    assert result.status is DetectionStatus.DETECTED and result.adapter.name == "cursor-cli"


def test_snake_case_session_id_does_not_contradict_claude_code(jsonl):
    """Real Claude Code sessions mix camelCase and snake_case keys: an extra session_id is neutral."""
    path = jsonl(
        [
            {"type": "canary-unknown-record", "session_id": "s-1", "timestamp": "2026-03-14T10:00:00Z"},
            {"type": "custom-title", "session_id": "s-1"},
            cc_record("user", "u1", None, {"role": "user", "content": "hi"}, session_id="s-1"),
            cc_record("assistant", "a1", "u1", {"id": "m1", "role": "assistant", "content": []}),
        ]
    )
    result = detection_of(path)
    assert result.status is DetectionStatus.DETECTED and result.adapter.name == "claude-code"


def test_cursor_events_with_future_fields_are_still_cursor(jsonl):
    """Cursor may add fields (consumers must ignore unknown ones): uuid or tools do not rule it out."""
    path = jsonl(
        [
            {"type": "system", "subtype": "init", "apiKeySource": "login", "session_id": "s", "model": "x",
             "cwd": "/w", "permissionMode": "default", "tools": ["read"], "mcp_servers": [], "uuid": "u-1"},
            cursor_call(uuid="u-2"),
        ]
    )
    result = detection_of(path)
    assert result.status is DetectionStatus.DETECTED and result.adapter.name == "cursor-cli"


def test_valid_records_still_decide_when_some_lines_are_broken():
    result = detection_of(fixture("edge-cases/invalid-json.jsonl"))
    assert result.status is DetectionStatus.DETECTED and result.adapter.name == "claude-code"


def test_mixed_formats_are_contradictory_and_refused():
    path = fixture("edge-cases/ambiguous-mixed.jsonl")
    result = detection_of(path)
    assert result.status is DetectionStatus.CONTRADICTORY
    assert set(result.matched) == {"claude-code", "cursor-cli"}
    with pytest.raises(FormatError) as error:
        resolve_format(path)
    assert error.value.detection.adapter is None


def test_a_record_matching_two_signatures_is_contradictory(jsonl):
    hybrid = {**cc_record("system", "s1", None, None, subtype="init"), "apiKeySource": "login", "session_id": "s"}
    assert detection_of(jsonl([hybrid])).status is DetectionStatus.CONTRADICTORY


def test_known_unsupported_format_is_reported_not_mistaken():
    # The Claude Agent SDK stream-json output resembles Cursor's: it must not be taken for cursor-cli.
    result = detection_of(fixture("edge-cases/claude-agent-sdk-stream.jsonl"))
    assert result.status is DetectionStatus.UNSUPPORTED
    assert result.unsupported.name == "claude-agent-sdk-stream" and result.adapter is None


def test_unknown_format_has_insufficient_evidence():
    result = detection_of(fixture("edge-cases/unknown-format.jsonl"))
    assert result.status is DetectionStatus.INSUFFICIENT and result.likely == []


@pytest.mark.parametrize("name", ["empty.jsonl", "blank-lines.jsonl"])
def test_files_without_records_are_empty(name):
    assert detection_of(fixture(f"edge-cases/{name}")).status is DetectionStatus.EMPTY


def test_only_malformed_lines_is_insufficient(jsonl):
    assert detection_of(jsonl(["{bad", "also bad"])).status is DetectionStatus.INSUFFICIENT


def test_weak_evidence_gives_a_hint_but_no_decision(jsonl):
    session = {"session_id": "s"}
    path = jsonl(
        [
            {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}, **session},
            {"type": "assistant", "message": {"role": "assistant", "content": []}, **session},
        ]
    )
    result = detection_of(path)
    assert result.status is DetectionStatus.INSUFFICIENT and result.likely == ["cursor-cli"]


def test_sampling_extends_while_no_format_is_proven(jsonl):
    """A session that starts with many metadata records is still detected (bounded extra sampling)."""
    metadata = [{"type": "mode", "sessionId": "s"}] * 120
    path = jsonl([*metadata, cc_record("user", "u1", None, {"role": "user", "content": "hi"})])
    result = detection_of(path)
    assert result.status is DetectionStatus.DETECTED and result.records_examined == 121


def test_explicit_format_skips_detection():
    path = fixture("edge-cases/ambiguous-mixed.jsonl")
    result = resolve_format(path, "cursor-cli")
    assert result.status is DetectionStatus.EXPLICIT and result.adapter.name == "cursor-cli"


def test_detection_examines_a_bounded_number_of_records(jsonl):
    path = jsonl([cursor_call(f"c{i}") for i in range(500)])
    assert detection_of(path, Limits(sniff_records=20)).records_examined == 20
    neutral = jsonl([{"event": "x"}] * 3000, name="neutral.jsonl")
    assert detection_of(neutral, Limits(sniff_max_records=200)).records_examined == 200
