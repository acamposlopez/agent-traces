"""Properties that must hold whatever the fixture: the expectation is always an independent reference (the
report of the untouched file, or the structure of the parsed value), never the production algorithm.

- renaming every identifier consistently leaves the inspection report unchanged;
- changing every fictitious secret leaves the report unchanged and the new value never appears;
- unknown fields at any level never change the default report;
- a line over the limit is always reported, never silently dropped from a report that says ``ok``;
- usage errors never echo the offending value, whatever it looks like.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import pytest

from agent_traces import cli
from conftest import CLAUDE_FIXTURES, CURSOR_FIXTURES, fixture

SUPPORTED = CLAUDE_FIXTURES + CURSOR_FIXTURES
ID_KEYS = frozenset({
    "uuid", "parentUuid", "logicalParentUuid", "sessionId", "session_id", "agentId", "id", "tool_use_id", "call_id",
    "leafUuid", "messageId", "requestId", "request_id", "toolUseID", "parentToolUseID", "sourceToolUseID",
    "sourceToolAssistantUUID", "model_call_id", "toolCallId", "parentSessionId", "parentLastUuid",
})
# A bijection on the characters identifiers are made of: same length, so byte counts stay equal.
RENAME = str.maketrans("0123456789abcdef", "fedcba9876543210")


def widen(value: Any) -> Any:
    """Add an unknown field (with a nested unknown field) to every object at every level."""
    if isinstance(value, dict):
        return {**{k: widen(v) for k, v in value.items()}, "canary_extra": {"canary_inner": "CANARY-EXTRA"}}
    if isinstance(value, list):
        return [widen(item) for item in value]
    return value


def run_json(path: pathlib.Path, *flags: str, capsys) -> dict[str, Any]:
    code = cli.main(["inspect", str(path), "--output", "json", *flags])
    out = capsys.readouterr().out
    assert code == 0
    return json.loads(out)


def rewrite(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {k: (v.translate(RENAME) if k in keys and isinstance(v, str) else rewrite(v, keys)) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(item, keys) for item in value]
    return value


def transformed_copy(path: pathlib.Path, tmp_path: pathlib.Path, transform) -> pathlib.Path:
    lines = []
    for raw in path.read_bytes().split(b"\n"):
        try:
            lines.append(json.dumps(transform(json.loads(raw)), ensure_ascii=False).encode("utf-8"))
        except ValueError:
            lines.append(raw)  # blank or unparseable line: kept as is
    copy = tmp_path / path.name
    copy.write_bytes(b"\n".join(lines))
    return copy


def without_sizes(report: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(json.dumps(report))
    for key in ("bytes", "snapshot_bytes"):
        report["input"].pop(key, None)
    return report


@pytest.mark.parametrize("path", SUPPORTED, ids=lambda p: p.name)
def test_renaming_identifiers_consistently_keeps_the_report(path, tmp_path, capsys):
    original = transformed_copy(path, tmp_path / "a", lambda v: v) if False else path  # the untouched file
    renamed = transformed_copy(path, tmp_path, lambda v: rewrite(v, ID_KEYS))
    assert renamed.read_bytes() != path.read_bytes()
    assert without_sizes(run_json(renamed, capsys=capsys)) == without_sizes(run_json(original, capsys=capsys))


@pytest.mark.parametrize("name", ["claude-code/secrets.jsonl", "cursor-cli/sensitive-data.jsonl"])
def test_changing_every_secret_leaves_the_report_unchanged_and_hidden(name, tmp_path, capsys):
    path = fixture(name)
    changed = tmp_path / path.name
    changed.write_bytes(path.read_bytes().replace(b"CANARY", b"ZEBRA6").replace(b"canary", b"zebra6"))
    assert b"ZEBRA6" in changed.read_bytes()
    reports = []
    for flags in ((), ("--discover-schema",)):
        before = run_json(path, *flags, capsys=capsys)
        after = run_json(changed, *flags, capsys=capsys)
        assert before == after
        reports.append(after)
    cli.main(["inspect", str(changed), "--discover-schema"])
    text = capsys.readouterr().out
    assert "zebra6" not in (json.dumps(reports) + text).lower()


@pytest.mark.parametrize("path", CLAUDE_FIXTURES, ids=lambda p: p.name)
def test_unknown_fields_never_change_the_default_report(path, tmp_path, capsys):
    widened = transformed_copy(path, tmp_path, widen)
    assert without_sizes(run_json(widened, capsys=capsys)) == without_sizes(run_json(path, capsys=capsys))


@pytest.mark.parametrize("path", SUPPORTED, ids=lambda p: p.name)
def test_a_line_over_the_limit_is_always_reported(path, tmp_path, capsys):
    over = tmp_path / path.name
    over.write_bytes(path.read_bytes() + b'{"type": "user", "canary_padding": "' + b"x" * 5000 + b'"}\n')
    fmt = "claude-code" if path in CLAUDE_FIXTURES else "cursor-cli"
    report = run_json(over, "--format", fmt, "--max-line-bytes", "4096", capsys=capsys)
    assert report["records"]["unrecognized"].get("oversized", 0) >= 1
    assert report["integrity"]["status"] in ("degraded", "unusable")
    assert any(issue["code"] == "oversized" and issue["severity"] == "error" for issue in report["integrity"]["issues"])


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "x.jsonl", "--format", "claude-codeéCANARY"],
        ["inspect", "x.jsonl", "--format", "/home/CANARY-user/private.jsonl"],
        ["inspect", "x.jsonl", "--output", "json CANARY"],
        ["inspect", "x.jsonl", "--output", "%s%n%CANARY"],
        ["inspect", "x.jsonl", "--max-line-bytes", "1024CANARY"],
        ["inspect", "x.jsonl", "--max-line-bytes", "-5CANARY"],
        ["inspect", "x.jsonl", "--max-line-bytes", "99999999999999999999CANARY"],
        ["inspect", "x.jsonl", "--CANARY=/home/CANARY"],
        ["inspect", "--", "--CANARY-looks-like-a-flag"],
    ],
)
def test_usage_errors_never_echo_odd_values(argv, capsys):
    try:
        code = cli.main(argv)  # a value taken as the file name is an input error, returned, not raised
    except SystemExit as error:
        code = error.code
    err = capsys.readouterr().err
    assert code in (cli.EXIT_USAGE, cli.EXIT_INPUT) and "error:" in err
    assert "canary" not in err.lower() and not re.search(r"/home/", err)
