"""Command line: options, output formats and exit codes."""

from __future__ import annotations

import json
import subprocess

import pytest

from agent_traces import cli
from conftest import fixture


def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_no_command_prints_help(capsys):
    code, out, _ = run_cli(capsys)
    assert code == 0 and "inspect" in out


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("claude-code/tool-calls.jsonl", "claude-code"), ("cursor-cli/tool-calls.jsonl", "cursor-cli")],
)
def test_inspect_text_and_json(capsys, name, fmt):
    code, out, err = run_cli(capsys, "inspect", str(fixture(name)))
    assert code == 0 and err == ""
    assert f"format   {fmt} (auto-detected" in out and "Integrity: ok" in out
    code, out, _ = run_cli(capsys, "inspect", str(fixture(name)), "--format", fmt, "--output", "json")
    report = json.loads(out)
    assert code == 0 and report["source"] == {
        "format": fmt,
        "format_name": report["source"]["format_name"],
        "adapter_version": "0.1.0",
        "detection": "explicit",
        "records_examined": 0,
    }


def test_explicit_format_overrides_detection(capsys):
    path = fixture("edge-cases/ambiguous-mixed.jsonl")
    code, out, _ = run_cli(capsys, "inspect", str(path), "--format", "claude-code")
    assert code == 0 and "set with --format" in out


@pytest.mark.parametrize("name", ["edge-cases/ambiguous-mixed.jsonl", "edge-cases/unknown-format.jsonl"])
def test_undeterminable_format_asks_for_it(capsys, name):
    code, out, err = run_cli(capsys, "inspect", str(fixture(name)))
    assert code == cli.EXIT_FORMAT and out == ""
    assert "specify it with --format (claude-code, cursor-cli)" in err


def test_weak_evidence_hint(capsys, jsonl):
    message = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    path = jsonl([{"type": "assistant", "message": message, "session_id": "s"}])
    code, _, err = run_cli(capsys, "inspect", str(path))
    assert code == cli.EXIT_FORMAT and "insufficient evidence" in err
    assert "weak evidence suggests: cursor-cli (not enough to decide)" in err


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("edge-cases/ambiguous-mixed.jsonl", "contradictory evidence: records match the signatures of several"),
        ("edge-cases/claude-agent-sdk-stream.jsonl", "known but unsupported format: Claude Agent SDK stream-json"),
        ("edge-cases/unknown-format.jsonl", "insufficient evidence: no supported format signature in 3 record(s)"),
    ],
)
def test_refusals_explain_why(capsys, name, expected):
    code, out, err = run_cli(capsys, "inspect", str(fixture(name)))
    assert code == cli.EXIT_FORMAT and out == "" and expected in err


def test_missing_file(capsys, tmp_path):
    code, out, err = run_cli(capsys, "inspect", str(tmp_path / "nope.jsonl"))
    assert code == cli.EXIT_INPUT and out == "" and "input file not found" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect"],
        ["inspect", "x", "--format", "codex"],
        ["inspect", "x", "--max-line-bytes", "10"],
        ["inspect", "x", "--max-line-bytes", str(512 * 2**20)],  # above the 256 MiB ceiling
    ],
)
def test_usage_errors(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == cli.EXIT_USAGE


def test_max_line_bytes_option(capsys, jsonl):
    path = jsonl(["{" + '"k": "' + "v" * 3000 + '"}', {"type": "result", "subtype": "success", "session_id": "s"}])
    code, out, _ = run_cli(capsys, "inspect", str(path), "--format", "cursor-cli", "--max-line-bytes", "2048",
                           "--output", "json")
    report = json.loads(out)
    assert code == 0 and report["records"]["unrecognized"] == {"oversized": 1}


def test_package_metadata_describes_inspection_not_redaction():
    """The package only inspects traces: its docstring and description must not announce redaction."""
    import pathlib
    import tomllib

    import agent_traces

    pyproject = tomllib.loads((pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())
    for text in (agent_traces.__doc__ or "", pyproject["project"]["description"]):
        assert "redact" not in text.lower() and "inspect" in text.lower()


def test_installed_entry_point():
    out = subprocess.run(
        ["agent-traces", "inspect", str(fixture("cursor-cli/simple.jsonl")), "--output", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(out.stdout)["integrity"]["status"] == "ok"
