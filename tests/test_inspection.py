"""Inspection report: counts, relations, field coverage and integrity status."""

from __future__ import annotations

import pytest

from conftest import cc_record, fixture, report_of


def issue_codes(report):
    return {issue["code"]: issue["count"] for issue in report.integrity["issues"]}


def test_claude_code_tool_session_report():
    report = report_of("claude-code/tool-calls.jsonl")
    assert report.source["format"] == "claude-code" and report.source["detection"] == "detected"
    assert report.input["records"] == 8
    assert report.counts["messages"] == {"total": 4, "assistant": 3, "user": 1}
    assert report.counts["tool_calls"] == report.counts["tool_results"] == 2
    assert report.tools == {
        "categories": {"file_read": 1, "file_write": 1},
        "known_tools": {"Read": 1, "Write": 1},
        "names_not_shown": 0,
        "calls_without_name": 0,
    }
    assert report.relations["paired_calls"] == 2 and report.relations["parent_refs_unresolved"] == 0
    assert report.session["terminal_state"] == "not_applicable"
    assert report.integrity == {"status": "ok", "issues": []}


def test_split_assistant_message_is_counted_once():
    report = report_of("claude-code/simple.jsonl")
    assert report.counts["messages"] == {"total": 4, "assistant": 2, "user": 2}
    assert report.counts["reasoning_blocks"] == 1
    assert report.records["notices"] == {"file-history-snapshot": 1}


def test_errors_are_counted_by_origin():
    report = report_of("claude-code/errors-recovery.jsonl")
    assert report.counts["errors"] == {"total": 3, "tool": 1, "api": 2, "run_failed": 0}
    # A result without is_error is never presented as a confirmed success.
    assert report.counts["result_outcomes"] == {
        "success_confirmed": 2, "success_protocol_default": 1, "error_confirmed": 1, "unknown": 0,
    }
    assert report.tools["categories"] == {"file_read": 1, "file_write": 1, "shell": 2}


def test_cursor_failed_run_report():
    report = report_of("cursor-cli/failed-run.jsonl")
    assert report.session["terminal_state"] == "missing"
    assert report.relations["calls_without_result"] == 1
    assert issue_codes(report) == {
        "missing_terminal_event": 1,
        "calls_without_result": 1,
        "unknown_result_outcomes": 1,
    }
    assert report.integrity["status"] == "degraded"
    assert report.tools["names_not_shown"] == 1


def test_partial_output_report_counts_messages_once():
    report = report_of("cursor-cli/sensitive-data.jsonl")
    assert report.input["records"] == 14 and report.counts["events"] == 10
    assert report.counts["messages"] == {"total": 3, "assistant": 2, "user": 1}
    assert report.records["normalization"] == {
        "cursor_partial_final_flush": 1,
        "cursor_partial_flush_before_tool": 1,
    }


def test_subagent_relations():
    main = report_of("claude-code/subagent-main.jsonl")
    assert main.relations["subagent_calls"] == 1 and main.relations["subagent_refs"] == 1
    agent = report_of("claude-code/subagent-agent.jsonl")
    assert agent.relations["agent_ids"] == 1
    assert agent.relations["sidechain_events"] == agent.counts["events"]


@pytest.mark.parametrize(
    ("name", "fmt", "status"),
    [
        ("edge-cases/empty.jsonl", "auto", "empty"),
        ("edge-cases/blank-lines.jsonl", "auto", "empty"),
        ("edge-cases/empty.jsonl", "cursor-cli", "empty"),
        ("edge-cases/invalid-json.jsonl", "auto", "degraded"),
        ("edge-cases/unknown-fields.jsonl", "auto", "warnings"),
        ("edge-cases/incomplete-records.jsonl", "auto", "degraded"),
        ("edge-cases/unknown-format.jsonl", "claude-code", "unusable"),
        ("cursor-cli/simple.jsonl", "auto", "ok"),
        ("cursor-cli/tool-calls.jsonl", "auto", "ok"),
    ],
)
def test_integrity_status(name, fmt, status):
    assert report_of(name, fmt).integrity["status"] == status


def test_empty_file_report():
    report = report_of("edge-cases/empty.jsonl")
    assert report.source["format"] is None and report.source["detection"] == "empty"
    assert report.input["records"] == 0 and report.counts["events"] == 0


def test_malformed_input_report():
    report = report_of("edge-cases/invalid-json.jsonl")
    # The unterminated final fragment is an incomplete tail (the file may be being written), not a record.
    assert issue_codes(report) == {"malformed_json": 3, "not_an_object": 1, "incomplete_final_record": 1}
    assert report.records["interpreted_events"] == 3 and report.input["records"] == 7
    assert report.input["ends_with_newline"] is False and report.input["incomplete_tail_bytes"] > 0


def test_field_coverage_distinguishes_states():
    report = report_of("cursor-cli/sensitive-data.jsonl")
    assert report.fields["timestamp"] == {"present": 2, "absent": 7, "unparseable": 0}
    assert report.fields["message.content"] == {"present": 3, "absent": 0, "unparseable": 0}


def test_pairing_anomalies(jsonl):
    def call(uuid, parent, call_id):
        return cc_record("assistant", uuid, parent, {"id": uuid, "role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "Bash", "input": {}}]})

    def result(uuid, parent, call_id):
        return cc_record("user", uuid, parent, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": "ok"}]})

    path = jsonl(
        [
            result("u0", None, "t0"),  # result before any call
            call("a1", "u0", "t1"),
            call("a2", "a1", "t1"),  # duplicate call id
            result("u1", "a2", "t1"),
            result("u2", "u1", "t1"),  # duplicate result
            call("a3", "missing-parent", "t3"),  # never answered, unknown parent
        ]
    )
    report = report_of(path)
    assert issue_codes(report) == {
        "results_without_call": 1,
        "duplicate_call_ids": 1,
        "duplicate_results": 1,
        "calls_without_result": 1,
        "unresolved_parent_refs": 1,
    }
    assert report.relations["paired_calls"] == 1


def test_timestamp_and_session_anomalies(jsonl):
    path = jsonl(
        [
            cc_record("user", "u1", None, {"role": "user", "content": "a"}, timestamp="2026-03-14T10:00:05Z"),
            cc_record("user", "u2", "u1", {"role": "user", "content": "b"}, timestamp="2026-03-14T10:00:01Z"),
            cc_record("user", "u3", "u2", {"role": "user", "content": "c"}, timestamp="not a date",
                      sessionId="s-2"),
            cc_record("user", "u4", "u3", {"role": "user", "content": "d"}, timestamp="2026-03-14T10:00:09Z",
                      sessionId=12345),
        ]
    )
    report = report_of(path)
    assert issue_codes(report) == {
        "conversation_out_of_order": 1,
        "unparseable_timestamps": 1,
        "multiple_sessions": 1,
        "unparseable_session_ids": 1,
    }
    assert report.session["distinct_sessions"] == 2


def test_text_rendering_lists_every_section():
    from agent_traces.processors import render_text

    text = render_text(report_of(fixture("claude-code/errors-recovery.jsonl")))
    for heading in ("Source", "Session", "Counts", "Tools", "Relations", "Records", "Integrity: ok", "Limitations"):
        assert heading in text
    assert "errors         3 (tool 1, api 2, run failed 0)" in text


@pytest.mark.parametrize(
    ("value", "safe"),
    [
        ("claude-sonnet-4-5", True),
        ("claude-sonnet-4-5-20250929", True),
        ("Claude 4 Sonnet", True),
        ("gpt-5", True),
        ("o3-mini", True),
        ("gemini-2.5-pro", True),
        ("canary-private-model", False),  # unknown family
        ("claude-sk7Hq2LmN9xPz4RtVb8W", False),  # long random-looking token
        ("claude\nsecret", False),
        ("claude/" + "x" * 80, False),
    ],
)
def test_model_values_are_only_shown_when_safe(value, safe):
    from agent_traces.processors.inspection import _is_safe_model

    assert _is_safe_model(value) is safe


@pytest.mark.parametrize("flags", [{}, {"include_file_hash": True, "include_models": True}], ids=["defaults", "opt-in"])
def test_reports_are_ascii_only(flags):
    """ASCII output cannot fail on any terminal encoding (which would print a traceback)."""
    from agent_traces.engine import run as run_engine
    from agent_traces.processors import InspectionProcessor, render_text
    from conftest import CLAUDE_FIXTURES, CURSOR_FIXTURES

    for path in CLAUDE_FIXTURES + CURSOR_FIXTURES:
        processor = InspectionProcessor(**flags)
        report = run_engine(path, [processor], compute_hash=bool(flags))["inspection"]
        assert render_text(report).isascii() and report.to_json().isascii()


def test_result_basis_is_reported():
    from agent_traces.processors import render_text

    report = report_of("claude-code/errors-recovery.jsonl")
    assert "success_confirmed 2, success_protocol_default 1, error_confirmed 1" in render_text(report)


def test_unparseable_identifiers_are_flagged(jsonl):
    path = jsonl(
        [
            cc_record("user", "u" * 300, None, {"role": "user", "content": "a"}),  # longer than any real id
            cc_record("user", "u2", ["not", "an", "id"], {"role": "user", "content": "b"}),
        ]
    )
    report = report_of(path)
    assert {"code": "unparseable_ids", "severity": "warning", "count": 2} in report.integrity["issues"]


# Coverage, origins, outcomes and policies (phase 1.6) ----------------------------------------------------------


def test_coverage_separates_semantic_metadata_and_unknown_records():
    coverage = report_of("edge-cases/unknown-fields.jsonl").coverage
    assert coverage["records"] == 8
    assert [coverage[k]["records"] for k in ("semantic", "metadata_verified", "metadata_name_only", "unknown",
                                             "skipped")] == [3, 2, 0, 3, 0]
    assert coverage["semantically_interpreted_percent"] == 37.5
    assert coverage["structurally_recognized_percent"] == 62.5
    assert coverage["unknown_blocks_in_interpreted_records"] == 1


def test_documented_duplicates_are_counted_as_skipped():
    coverage = report_of("cursor-cli/sensitive-data.jsonl").coverage
    counts = tuple(coverage[key]["records"] for key in ("semantic", "skipped", "unknown"))
    assert counts == (11, 2, 1)
    assert coverage["skipped"]["percent"] == 14.3


def test_user_messages_are_counted_by_origin(jsonl):
    def user(uuid, content, **extra):
        return cc_record("user", uuid, None, {"role": "user", "content": content}, **extra)

    path = jsonl(
        [
            user("u1", "please fix the bug"),
            user("u2", "<local-command-caveat>Caveat: generated by commands</local-command-caveat>", isMeta=True),
            user("u3", "<command-name>/model</command-name>\n<command-args></command-args>"),
            user("u4", "<local-command-stdout>model set</local-command-stdout>"),
            user("u5", "[Request interrupted by user for tool use]"),
            user("u6", "expanded skill instructions", sourceToolUseID="toolu_1"),
            cc_record("user", "u7", None, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t9", "content": "ok"},
                {"type": "text", "text": "<system-reminder>context</system-reminder>"}]}),
        ]
    )
    counts = report_of(path).counts
    assert counts["messages"]["user"] == 7
    assert counts["user_messages_by_origin"] == {
        "prompt": 1, "command": 1, "command_output": 1, "interruption": 1, "injected": 3, "unknown": 0,
    }


def test_agent_version_and_models_are_hidden_by_default():
    from agent_traces.engine import run
    from agent_traces.processors import InspectionProcessor

    path = fixture("claude-code/tool-calls.jsonl")
    hidden = report_of(path).session
    assert (hidden["agent_versions"], hidden["agent_versions_not_shown"], hidden["agent_versions_visible"]) == (
        [], 1, False)
    shown = run(path, [InspectionProcessor(include_agent_version=True)])["inspection"].session
    assert shown["agent_versions"] == ["2.1.150"] and shown["agent_versions_not_shown"] == 0


def test_export_policy_is_conservative():
    policy = report_of("edge-cases/unknown-fields.jsonl").records["export_policy"]
    # 3 unrecognised -> never; 2 metadata notices + 1 injected compaction summary -> not by default.
    assert policy == {"after_scrubbing": 2, "not_by_default": 3, "never": 3}


def test_metadata_timestamps_do_not_count_as_conversation_disorder(jsonl):
    path = jsonl(
        [
            cc_record("user", "u1", None, {"role": "user", "content": "a"}, timestamp="2026-03-14T10:00:05Z"),
            {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-03-14T09:00:00Z", "sessionId": "s"},
            {"type": "queue-operation", "operation": "dequeue", "timestamp": "2026-03-14T08:00:00Z", "sessionId": "s"},
            cc_record("user", "u2", "u1", {"role": "user", "content": "b"}, timestamp="2026-03-14T10:00:06Z"),
        ]
    )
    report = report_of(path)
    assert report.fields["out_of_order"] == {"conversation": 0, "metadata": 1}
    assert report.integrity["status"] == "ok"


def test_unterminated_tail_is_a_warning_not_a_failure(jsonl):
    path = jsonl([cc_record("user", "u1", None, {"role": "user", "content": "a"})])
    path.write_bytes(path.read_bytes() + b'{"parentUuid": "u1", "type": "assis')
    report = report_of(path)
    assert issue_codes(report) == {"incomplete_final_record": 1}
    assert report.integrity["status"] == "warnings" and report.input["records"] == 1
