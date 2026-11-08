"""The case study (CASE_STUDY.md) rests on three synthetic examples in ``examples/``. The counts it documents
were fixed by hand from the files (lines, blocks and references), not copied from the program, and every report
must stay free of the examples' own text. If an example or the inspector changes, this test says so."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent_traces import cli
from test_privacy import ENUM_LIKE, allowed_vocabulary

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"

# Text of each example that must never reach a report: prompts, claims, paths, commands, identifiers.
MUST_NOT_APPEAR = {
    "case-a-complete-task.jsonl": ["dry-run", "sync.py", "example-app", "pytest -q", "aaaa0001", "msg_example_a", "toolu_example"],
    "case-b-missing-result.jsonl": ["migrat", "staging", "run_terminal_command", "example-app", "bbbb0000", "call_example", "req-example"],
    "case-c-unverified-claim.jsonl": ["health endpoint", "200 OK", "deployed", "example-app", "cccc0001", "msg_example_c"],
}

# Hand-derived expectations. Case A: 9 records; 1 user prompt; assistant messages are counted from text blocks,
# one per distinct message id (lines 4 and 9), while records holding only tool_use blocks (lines 2, 5 and 7) are
# calls, not messages, so the assistant count is 2; three tool calls (Bash, Edit, Bash)
# all answered: one explicit error, one explicit success, one success by protocol default (no is_error); every
# record after the first points at the previous uuid (8 resolved parents). Case B: 6 records, a started call with
# no completed event (1 call without result, warning), a documented terminal result. Case C: 2 records, no tools.
EXPECTED: dict[str, dict[str, Any]] = {
    "case-a-complete-task.jsonl": {
        "source": {"format": "claude-code", "detection": "detected"},
        "input": {"records": 9},
        "counts": {
            "events": 9,
            "messages": {"total": 3, "assistant": 2, "user": 1},
            "user_messages_by_origin": {"prompt": 1},
            "tool_calls": 3,
            "tool_results": 3,
            "result_outcomes": {"success_confirmed": 1, "success_protocol_default": 1, "error_confirmed": 1, "unknown": 0},
            "errors": {"total": 1, "tool": 1, "run_failed": 0},
        },
        "tools": {"categories": {"shell": 2, "file_write": 1}, "known_tools": {"Bash": 2, "Edit": 1}, "names_not_shown": 0},
        "relations": {"paired_calls": 3, "calls_without_result": 0, "results_without_call": 0,
                      "parent_refs_resolved": 8, "parent_refs_unresolved": 0},
        "integrity": {"status": "ok", "issues": []},
    },
    "case-b-missing-result.jsonl": {
        "source": {"format": "cursor-cli", "detection": "detected"},
        "input": {"records": 6},
        "session": {"terminal_state": "completed", "distinct_sessions": 1},
        "counts": {
            "events": 6,
            "messages": {"total": 3, "assistant": 2, "user": 1},
            "tool_calls": 1,
            "tool_results": 0,
            "errors": {"total": 0, "tool": 0, "run_failed": 0},
        },
        "tools": {"categories": {"unknown": 1}, "known_tools": {}, "names_not_shown": 1},
        "relations": {"paired_calls": 0, "calls_without_result": 1, "results_without_call": 0},
        "records": {"session_markers": {"start": 1, "end": 1}},
        "integrity": {"status": "warnings", "issues": [{"code": "calls_without_result", "severity": "warning", "count": 1}]},
    },
    "case-c-unverified-claim.jsonl": {
        "source": {"format": "claude-code", "detection": "detected"},
        "input": {"records": 2},
        "counts": {
            "events": 2,
            "messages": {"total": 2, "assistant": 1, "user": 1},
            "tool_calls": 0,
            "tool_results": 0,
            "errors": {"total": 0, "tool": 0, "run_failed": 0},
        },
        "tools": {"known_tools": {}, "names_not_shown": 0},
        "relations": {"paired_calls": 0, "calls_without_result": 0, "parent_refs_resolved": 1, "parent_refs_unresolved": 0},
        "integrity": {"status": "ok", "issues": []},
    },
}


def run(path: pathlib.Path, *flags: str, capsys) -> tuple[int, str, str]:
    code = cli.main(["inspect", str(path), *flags])
    out = capsys.readouterr()
    return code, out.out, out.err


def assert_subset(expected: Any, actual: Any, where: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        for key, value in expected.items():
            assert key in actual, f"{where}.{key} missing"
            assert_subset(value, actual[key], f"{where}.{key}")
    else:
        assert expected == actual, f"{where}: expected {expected!r}, got {actual!r}"


def string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_case_study_reports_match_the_hand_derived_counts(name, capsys):
    code, out, err = run(EXAMPLES / name, "--output", "json", capsys=capsys)
    assert code == 0 and err == ""
    assert_subset(EXPECTED[name], json.loads(out), name)


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("output", ["text", "json"])
def test_case_study_reports_contain_none_of_the_examples_text(name, output, capsys):
    path = EXAMPLES / name
    code, out, _ = run(path, "--output", output, "--discover-schema", capsys=capsys)
    assert code == 0
    assert [needle for needle in MUST_NOT_APPEAR[name] if needle.lower() in out.lower()] == []
    # Stronger: no string value of 8+ characters from the file, unless it is a reviewed name of the format
    # (role and block names, enumerations) that the report may print by design.
    vocabulary = allowed_vocabulary()
    values = set()
    for line in path.read_text().splitlines():
        values.update(string_values(json.loads(line)))
    leaked = [v for v in values if len(v) >= 8 and v not in vocabulary and not ENUM_LIKE.fullmatch(v) and v in out]
    assert leaked == [], f"{len(leaked)} value(s) of the example appear in the report"
