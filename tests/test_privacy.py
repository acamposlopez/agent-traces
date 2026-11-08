"""Privacy guarantees: reports, errors, exceptions and reprs never expose fixture content, defaults are
private, reasoning never survives anywhere, unknown data is never trusted and nothing leaves the machine.

Every fictitious sensitive value in the fixtures contains the marker "CANARY". Assertion messages only
report counts so that a failure does not print the leaked values either.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import socket
import subprocess
import sys
from typing import Any

import pytest

from agent_traces import cli, reader
from agent_traces.adapters import DEFAULT_ADAPTERS, get_adapter
from agent_traces.detection import DetectionStatus, explicit
from agent_traces.engine import RunContext
from agent_traces.adapters.claude_code_vocabulary import METADATA_SPECS, SYSTEM_SPECS
from agent_traces.model import (
    ContentKind,
    ErrorScope,
    ExportPolicy,
    FieldState,
    Message,
    MessageOrigin,
    MetadataStructure,
    RecordCategory,
    ResultStatus,
    Role,
    SessionPhase,
    SourceRef,
    StatusBasis,
    TerminalState,
    ToolCategory,
    Unrecognized,
    UnrecognizedReason,
    semantic_view,
)
from agent_traces.processors.inspection import (
    GENERAL_LIMITATIONS,
    ISSUE_SEVERITY,
    TRACKED_FIELDS,
    InspectionProcessor,
    IntegrityStatus,
    Severity,
    _is_safe_model,
    _is_safe_version,
)
from agent_traces.reader import Limits, ReadStats
from conftest import (
    CANARY,
    cc_record,
    CLAUDE_FIXTURES,
    CURSOR_FIXTURES,
    EDGE_FIXTURES,
    fixture,
    report_of,
    trace_of,
    walk_strings,
)

SRC = pathlib.Path(__file__).parents[1] / "src" / "agent_traces"
ALL_FIXTURES = CLAUDE_FIXTURES + CURSOR_FIXTURES + EDGE_FIXTURES
FORMAT_FOR_UNDETECTABLE = {"ambiguous-mixed.jsonl": "cursor-cli", "unknown-format.jsonl": "claude-code",
                           "claude-agent-sdk-stream.jsonl": "cursor-cli"}
OPT_IN = ("--include-file-hash", "--include-models", "--include-agent-version")
DISCOVER = ("--discover-schema",)
OPAQUE_CODE = re.compile(r"^[TKSB](\d+|\*)$")
HEX64 = re.compile(r"[0-9a-f]{64}")


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def format_for(path) -> str:
    return FORMAT_FOR_UNDETECTABLE.get(path.name, "auto")


def string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def fixture_values(path) -> set[str]:
    """String values (not keys) of every parseable fixture line."""
    strings: set[str] = set()
    for line in path.read_bytes().split(b"\n"):
        try:
            strings.update(string_values(json.loads(line)))
        except ValueError:
            continue
    return strings


ENUM_LIKE = re.compile(r"[a-z]+(_[a-z]+)+")  # format enums such as subtypes ("compact_boundary", "end_turn")


# Reports ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "flags", [(), OPT_IN, DISCOVER, OPT_IN + DISCOVER], ids=["defaults", "opt-in", "discover", "all"]
)
@pytest.mark.parametrize("output", ["text", "json"])
def test_reports_never_contain_canary_values(capsys, path, flags, output):
    code, out, err = run_cli(capsys, "inspect", str(path), "--format", format_for(path), "--output", output, *flags)
    assert code == 0
    leaks = (out + err).lower().count(CANARY)
    assert leaks == 0, f"{leaks} canary marker(s) leaked"


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_reports_never_contain_original_string_values(capsys, path):
    """Stronger than the canary check: no original string value of 8+ characters may appear in the report,
    except allowlisted names, format enums and agent versions that pass the safe-pattern filter.
    (Unknown keys are covered by the canary check: every unknown key in the fixtures contains the marker.)"""
    code, out, _ = run_cli(capsys, "inspect", str(path), "--format", format_for(path), "--output", "json")
    assert code == 0
    vocabulary = allowed_vocabulary()
    leaked = [
        value
        for value in fixture_values(path)
        if len(value) >= 8
        and value not in vocabulary
        and not ENUM_LIKE.fullmatch(value)
        and not _is_safe_version(value)
        and value in out
    ]
    assert not leaked, f"{len(leaked)} original value(s) leaked"


def allowed_vocabulary() -> set[str]:
    vocabulary: set[str] = set(ISSUE_SEVERITY) | set(GENERAL_LIMITATIONS) | set(TRACKED_FIELDS)
    for enum in (ContentKind, ErrorScope, ExportPolicy, FieldState, MessageOrigin, MetadataStructure, RecordCategory,
                 ResultStatus, Role, SessionPhase, StatusBasis, TerminalState, ToolCategory, UnrecognizedReason,
                 DetectionStatus, IntegrityStatus, Severity):
        vocabulary.update(member.value for member in enum)
    vocabulary.update(spec.metadata_class for spec in (*METADATA_SPECS.values(), *SYSTEM_SPECS.values()))
    for adapter in DEFAULT_ADAPTERS:
        vocabulary.update({adapter.name, adapter.display_name, adapter.version, *adapter.known_tools,
                           *adapter.known_record_types, *adapter.limitations})
        vocabulary.update(adapter.vocabulary.discriminators)  # reviewed names, safe to show by design
    return vocabulary


def keys_and_values(node: Any, keys: list[str], values: list[str]) -> None:
    if isinstance(node, dict):
        for key, item in node.items():
            keys.append(key)
            keys_and_values(item, keys, values)
    elif isinstance(node, list):
        for item in node:
            keys_and_values(item, keys, values)
    elif isinstance(node, str):
        values.append(node)


DYNAMIC_TEXT = re.compile(
    r"^(\d+ reasoning block|\d+ record\(s\) or content block|\d+ distinct tool name|\d+ model and \d+ agent version"
    r"|\d+ tracked identifiers reached|\d+ bytes inspected: the file grew|\d+ bytes at the end form)"
)
# Notice types ("type" or "type/discriminator" from the vocabulary), normalisation codes and numbers.
STRUCTURAL = re.compile(r"^([a-z-]+/[A-Za-z_]+|(cursor|claude)_[a-z_]+|\d+(\.\d+)*)$")


def unexpected_strings(report: dict, *, allow_hash: bool = False, allow_models: bool = False):
    schema_keys: list[str] = ["code", "severity", "count", "sha256"]  # issue items and the opt-in hash
    keys_and_values(report_of(fixture("edge-cases/empty.jsonl")).to_dict(), schema_keys, [])
    vocabulary = allowed_vocabulary()
    keys: list[str] = []
    values: list[str] = []
    keys_and_values(report, keys, values)
    bad_keys = [k for k in keys if k not in vocabulary and k not in schema_keys and not STRUCTURAL.match(k)]
    bad_values = [
        v
        for v in values
        if not (
            v in vocabulary
            or STRUCTURAL.match(v)
            or DYNAMIC_TEXT.match(v)
            or _is_safe_version(v)
            or (allow_models and _is_safe_model(v))
            or (allow_hash and HEX64.fullmatch(v))
        )
    ]
    return bad_keys, bad_values


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_report_strings_come_from_a_closed_vocabulary(path):
    """Privacy by construction: every key and string value of the report is code-defined or allowlisted."""
    bad_keys, bad_values = unexpected_strings(report_of(path, format_for(path)).to_dict())
    assert not bad_keys, f"{len(bad_keys)} unexpected key(s) in the report"
    assert not bad_values, f"{len(bad_values)} unexpected value(s) in the report"


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_opt_in_report_only_adds_the_hash_and_safe_models(capsys, path):
    code, out, _ = run_cli(capsys, "inspect", str(path), "--format", format_for(path), "--output", "json", *OPT_IN)
    assert code == 0
    bad_keys, bad_values = unexpected_strings(json.loads(out), allow_hash=True, allow_models=True)
    assert not bad_keys and not bad_values, f"{len(bad_keys) + len(bad_values)} unexpected string(s)"


# Private defaults: file hash and model identifiers ------------------------------------------------------------


@pytest.mark.parametrize("output", ["text", "json"])
def test_file_hash_is_omitted_by_default(capsys, output):
    code, out, _ = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")), "--output", output)
    assert code == 0 and "sha256" not in out and not HEX64.search(out)


def test_file_hash_only_on_request(capsys):
    path = fixture("claude-code/simple.jsonl")
    code, out, _ = run_cli(capsys, "inspect", str(path), "--output", "json", "--include-file-hash")
    assert code == 0 and json.loads(out)["input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    code, out, _ = run_cli(capsys, "inspect", str(path), "--include-file-hash")
    assert code == 0 and re.search(r"^  sha256 +[0-9a-f]{64}$", out, re.MULTILINE)


@pytest.mark.parametrize("name", ["claude-code/tool-calls.jsonl", "cursor-cli/tool-calls.jsonl"])
def test_model_identifiers_are_omitted_by_default(capsys, name):
    code, text, _ = run_cli(capsys, "inspect", str(fixture(name)))
    code_json, out, _ = run_cli(capsys, "inspect", str(fixture(name)), "--output", "json")
    assert code == code_json == 0
    assert "claude-sonnet-4-5" not in text + out and "Claude 4 Sonnet" not in text + out
    session = json.loads(out)["session"]
    assert (session["models"], session["models_not_shown"], session["models_visible"]) == ([], 1, False)
    assert "not shown; use --include-models" in text


def test_model_identifiers_on_request_still_hide_unsafe_values(capsys):
    code, out, _ = run_cli(capsys, "inspect", str(fixture("claude-code/secrets.jsonl")), "--output", "json",
                           "--include-models")
    session = json.loads(out)["session"]
    assert code == 0 and (session["models"], session["models_not_shown"]) == (["claude-sonnet-4-5"], 1)


# stderr and exceptions --------------------------------------------------------------------------------------


def test_error_messages_never_echo_paths_or_content(capsys, tmp_path):
    missing = tmp_path / "CANARY-private" / "session.jsonl"
    code, out, err = run_cli(capsys, "inspect", str(missing))
    assert code == cli.EXIT_INPUT and out == "" and CANARY not in err.lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "trace.jsonl", "/home/CANARY-user/clients/extra.jsonl"],
        ["inspect", "trace.jsonl", "--format", "CANARY-format"],
        ["inspect", "trace.jsonl", "--output", "CANARY"],
        ["inspect", "trace.jsonl", "--max-line-bytes", "CANARY"],
        ["--CANARY-flag=/home/CANARY-user"],
    ],
)
def test_usage_errors_never_echo_argument_values(capsys, argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    err = capsys.readouterr().err
    assert error.value.code == cli.EXIT_USAGE and "error:" in err and CANARY not in err.lower()


def test_internal_errors_hide_exception_messages(capsys, monkeypatch):
    adapter = get_adapter("claude-code")

    def explode(records, meta):
        for record in records:
            raise ValueError(f"CANARY secret from line {record.line}: {record.data}")
        yield

    monkeypatch.setattr(adapter, "parse", explode)
    code, out, err = run_cli(capsys, "inspect", str(fixture("claude-code/secrets.jsonl")))
    assert code == cli.EXIT_INTERNAL and out == ""
    assert "ValueError" in err and CANARY not in err.lower()


def test_read_errors_become_fixed_messages(capsys, monkeypatch):
    real = open(fixture("claude-code/simple.jsonl"), "rb")  # noqa: SIM115 - gives a real size to the snapshot

    class FailingHandle:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            real.close()
            return False

        def fileno(self):
            return real.fileno()

        def readline(self, size):
            raise OSError(5, "CANARY I/O error on /home/CANARY-user/session.jsonl")

    monkeypatch.setattr(reader, "open_readonly", lambda path: FailingHandle())
    code, out, err = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")), "--format", "claude-code")
    assert code == cli.EXIT_INPUT and out == ""
    assert "input file could not be read" in err and CANARY not in err.lower()


def test_rendering_errors_are_contained(capsys, monkeypatch):
    def broken(report):
        raise RuntimeError("CANARY rendering failure")

    monkeypatch.setattr(cli, "render_text", broken)
    code, out, err = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")))
    assert code == cli.EXIT_INTERNAL and out == "" and CANARY not in err.lower()


def test_interruption_exits_cleanly(capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupted)
    code, out, err = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")))
    assert code == cli.EXIT_INTERRUPTED and out == "" and err.strip() == "agent-traces: error: interrupted"


# Canonical model: reasoning, reprs, unknown data ------------------------------------------------------------


def test_thinking_content_never_enters_the_canonical_model():
    trace = trace_of("claude-code/secrets.jsonl")
    assert not any("CANARY-THINKING" in s or "CANARY-SIGNATURE" in s for s in walk_strings(trace.events))


def test_reasoning_nested_in_raw_payloads_is_removed(jsonl):
    from conftest import cc_record

    thinking = {"type": "thinking", "thinking": "CANARY-THINKING", "signature": "CANARY-SIGNATURE"}
    claude = jsonl(
        [
            {"type": "canary-future-record", "message": {"content": [thinking]}},  # unknown record type
            cc_record("assistant", "a1", None, {"id": "m1", "role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Agent", "input": {}},
                {"type": "canary_block", "inner": {"type": "redacted_thinking", "data": "CANARY-THINKING"}},
                thinking,
            ]}),
            cc_record("user", "u1", "a1", {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [thinking, {"type": "text", "text": "ok"}]}]},
                toolUseResult={"agentId": "x", "content": [thinking]}),
            cc_record("system", "s1", "u1", None, subtype="api_error", error={"detail": [thinking]}),
        ],
        name="claude.jsonl",
    )
    events = trace_of(claude, fmt="claude-code").events
    strings = list(walk_strings(events))
    assert not any("CANARY-THINKING" in s or "CANARY-SIGNATURE" in s for s in strings)
    assert strings.count("reasoning") >= 5  # content-free stubs mark where reasoning was

    session = {"session_id": "s"}
    cursor = jsonl(
        [
            {"type": "thinking", "text": "CANARY-THINKING", **session},  # suppressed in print mode, if it appeared
            {"type": "tool_call", "subtype": "completed", "call_id": "c1", **session,
             "tool_call": {"readToolCall": {"args": {}, "result": {"success": {"content": [thinking]}}}}},
        ],
        name="cursor.jsonl",
    )
    events = trace_of(cursor, fmt="cursor-cli").events
    assert not any("CANARY-THINKING" in s for s in walk_strings(events))


def test_reasoning_inside_unparseable_values_is_removed(jsonl):
    """A value the adapter cannot interpret keeps its raw form in ``Unparseable.raw``; reasoning nested in
    there (text that is not a string, object-valued ids or timestamps) must be removed like any raw payload."""
    thinking = {"type": "thinking", "thinking": "CANARY-THINKING", "signature": "CANARY-SIGNATURE"}
    path = jsonl(
        [
            cc_record("user", "u1", None, {"role": "user", "content": "hello"}),
            cc_record("assistant", "a1", "u1", {"id": "m1", "role": "assistant", "content": [
                {"type": "text", "text": [thinking]},  # text that is not a string
                {"type": "tool_use", "id": {"nested": [thinking]}, "name": "Bash", "input": {}},  # object as id
            ]}),
            cc_record("user", "u2", "a1", {"role": "user", "content": "x"}, timestamp={"t": [thinking]}),
            cc_record("user", {"id": [thinking]}, "u2", {"role": "user", "content": "y"}),  # object as uuid
            cc_record("user", "u3", "u2", {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t9", "content": "r"}]}, toolUseResult={"agentId": {"a": [thinking]}}),
        ]
    )
    strings = list(walk_strings(trace_of(path, fmt="claude-code").events))
    assert not any("CANARY-THINKING" in s or "CANARY-SIGNATURE" in s for s in strings)
    assert strings.count("reasoning") >= 5  # the stubs mark where the blocks were


@pytest.mark.parametrize("name", ["claude-code/secrets.jsonl", "cursor-cli/sensitive-data.jsonl"])
def test_reprs_never_show_content(name):
    trace = trace_of(name)
    text = repr(trace.metadata) + "".join(repr(event) for event in trace.events)
    assert CANARY not in text.lower()


@pytest.mark.parametrize("reason", list(UnrecognizedReason))
def test_unrecognized_records_are_never_treated_as_safe(reason):
    assert reason.value in ISSUE_SEVERITY  # every reason is at least a warning
    adapter = get_adapter("claude-code")
    context = RunContext(explicit(adapter), adapter, adapter.new_metadata(), ReadStats(records=2), Limits())
    processor = InspectionProcessor(include_file_hash=True, include_models=True)
    processor.start(context)
    processor.handle(Message(source=SourceRef(line=1, record_type="user"), role=Role.USER, content="hello"))
    unknown = Unrecognized(source=SourceRef(line=2), reason=reason, raw={"CANARY_field": "CANARY value"})
    processor.handle(unknown)
    report = processor.finish(context)
    assert report.integrity["status"] in ("warnings", "degraded")
    assert report.records["unrecognized"] == {reason.value: 1}
    assert CANARY not in report.to_json().lower()
    assert semantic_view([unknown]) == []


# Local-first: no writes, no network, no processes -----------------------------------------------------------


def test_inspection_writes_nothing_and_stays_offline(capsys, monkeypatch, tmp_path):
    source = fixture("claude-code/secrets.jsonl")
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(source.read_bytes())
    copy.chmod(0o400)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    code, _, _ = run_cli(capsys, "inspect", str(copy), "--output", "json")
    assert code == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert copy.read_bytes() == source.read_bytes()


AUDIT_HOOK = r"""
import contextlib, io, json, os, sys

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
FORBIDDEN_PREFIXES = ("socket.", "subprocess.", "ftplib.", "smtplib.", "urllib.", "http.", "webbrowser.",
                      "tempfile.", "shutil.")
FORBIDDEN_EVENTS = {"os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty", "os.remove",
                    "os.rename", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.symlink", "os.link"}
seen = set()

def hook(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writes = isinstance(mode, str) and any(c in mode for c in "wax+")
        if writes or (isinstance(flags, int) and flags & WRITE_FLAGS):
            seen.add("open-for-writing")
    elif event.startswith(FORBIDDEN_PREFIXES) or event in FORBIDDEN_EVENTS:
        seen.add(event)

sys.addaudithook(hook)
"""

AUDIT_SCRIPT = AUDIT_HOOK + r"""
from agent_traces import cli

codes = set()
for path in sys.argv[1:]:
    for extra in ([], ["--output", "json", "--include-file-hash", "--include-models", "--include-agent-version",
                       "--discover-schema"], ["--discover-schema"]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            codes.add(cli.main(["inspect", path, *extra]))
print(json.dumps({"events": sorted(seen), "codes": sorted(codes)}))
"""


def test_runtime_audit_detects_writes_sockets_processes_and_deletions(tmp_path):
    """Positive control: the same hook does report a file write, a socket, a child process and a deletion.
    This bounds what the audit proves: the hook sees these Python-level events, not raw system calls."""
    script = AUDIT_HOOK + (
        "open(sys.argv[1], 'w').close()\n"
        "import socket\n"
        "socket.socket().close()\n"
        "import subprocess\n"
        "subprocess.run([sys.executable, '-c', 'pass'], check=True)\n"
        "os.remove(sys.argv[1])\n"
        "print(json.dumps({'events': sorted(seen)}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path / "probe.txt")], capture_output=True, text=True, check=True
    )
    events = json.loads(result.stdout.strip().splitlines()[-1])["events"]
    assert "open-for-writing" in events and any(e.startswith("socket.") for e in events)
    assert any(e.startswith("subprocess.") for e in events) and "os.remove" in events


def test_runtime_audit_no_file_writes_network_or_processes():
    """Python audit hooks see every file open, socket and process the CLI could create, on every fixture."""
    result = subprocess.run(
        [sys.executable, "-B", "-c", AUDIT_SCRIPT, *map(str, ALL_FIXTURES)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    audit = json.loads(result.stdout.strip().splitlines()[-1])
    assert audit["events"] == []
    assert set(audit["codes"]) <= {0, 3}  # 3: undeterminable formats in auto mode


FORBIDDEN_MODULES = {
    "aiohttp", "asyncio", "ctypes", "ftplib", "http", "httpx", "imaplib", "logging", "multiprocessing", "pickle",
    "poplib", "requests", "shelve", "shutil", "smtplib", "socket", "sqlite3", "ssl", "subprocess", "telnetlib",
    "tempfile", "urllib", "webbrowser", "xmlrpc",
}
FORBIDDEN_OS_CALLS = re.compile(
    r"^(system|popen|fork|forkpty|exec\w*|spawn\w*|posix_spawn\w*|startfile|remove|unlink|rename|replace|mkdir|"
    r"makedirs|rmdir|removedirs|truncate|chmod|chown|symlink|link|write)$"
)


def test_source_has_no_network_process_temp_or_logging_code():
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                names = set()
            assert not names & FORBIDDEN_MODULES, path.name
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                assert not FORBIDDEN_OS_CALLS.match(node.attr), f"{path.name}: os.{node.attr}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "compile", "__import__"}, path.name


# Schema discovery ---------------------------------------------------------------------------------------------


def discovery_of(capsys, path, *flags):
    code, out, _ = run_cli(capsys, "inspect", str(path), "--format", format_for(path), "--output", "json",
                           "--discover-schema", *flags)
    assert code == 0
    return json.loads(out)["discovery"]


def test_discovery_is_off_by_default(capsys):
    code, text, _ = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")))
    _, out, _ = run_cli(capsys, "inspect", str(fixture("claude-code/simple.jsonl")), "--output", "json")
    assert code == 0 and "Schema discovery" not in text and "discovery" not in json.loads(out)


SPECIAL_LABELS = {"(absent)", "(unparseable line)", "(more fields)"} | {
    f"({kind})" for kind in ("number", "boolean", "array", "object", "null")
}
JSON_TYPES = {"string", "number", "boolean", "null", "object", "array", "other"}


def discovery_labels(result):
    for group in result["groups"]:
        yield group["type"]
        if group["discriminator"]:
            yield group["discriminator"]["value"]
        for stats in (group["fields"], *group["nested"].values()):
            for label, info in stats.items():
                yield label
                yield from info["types"]
        yield from group["nested"]
        yield from group["blocks"]


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_discovery_labels_come_from_the_vocabulary_or_are_opaque(capsys, path):
    result = discovery_of(capsys, path)
    names = set()
    for adapter in DEFAULT_ADAPTERS:
        vocab = adapter.vocabulary
        names |= {*vocab.record_types, *vocab.discriminators, *vocab.fields, *vocab.block_types}
    unexpected = [
        label for label in discovery_labels(result)
        if not (label in names or OPAQUE_CODE.match(label) or label in SPECIAL_LABELS or label in JSON_TYPES
                or label in {"problem", *UnrecognizedReason})
    ]
    assert not unexpected, f"{len(unexpected)} unexpected label(s)"


def test_innocent_looking_unknown_names_stay_opaque(capsys):
    """A short, harmless-looking key outside the vocabulary is still never shown."""
    result = discovery_of(capsys, fixture("edge-cases/discovery-hostile.jsonl"))
    rendered = json.dumps(result)
    assert '"merger"' not in rendered and "password" not in rendered
    assert result["opaque_codes"]["fields"] == 64 and result["truncated"] is True


def test_opaque_codes_are_local_order_codes_not_hashes(capsys, jsonl):
    """Different unknown names in the same order get the same codes: nothing about the name is encoded."""
    def run(names):
        path = jsonl([cc_record("user", "u1", None, {"role": "user", "content": "x"})]
                     + [{"type": name, name: 1} for name in names], name=f"{names[0]}.jsonl")
        groups = discovery_of(capsys, path)["groups"]
        return [(g["type"], sorted(g["fields"])) for g in groups if g["category"] == "unknown"]

    assert run(["alpha-secret", "beta-secret"]) == run(["gamma-other", "delta-other"])


def test_discovery_reports_vocabulary_provenance(capsys):
    result = discovery_of(capsys, fixture("claude-code/errors-recovery.jsonl"))
    sources = result["vocabulary_sources"]
    assert "usage" in sources["documented"] and "parentUuid" in sources["observed"]
