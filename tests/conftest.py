"""Shared helpers. Tests only ever read the synthetic fixtures in tests/fixtures/ or files they create."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent_traces.engine import load_trace, run
from agent_traces.model import Trace
from agent_traces.processors import InspectionProcessor, InspectionReport

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CANARY = "canary"  # every fictitious sensitive value in the fixtures contains this marker

CLAUDE_FIXTURES = sorted((FIXTURES / "claude-code").glob("*.jsonl"))
CURSOR_FIXTURES = sorted((FIXTURES / "cursor-cli").glob("*.jsonl"))
EDGE_FIXTURES = sorted((FIXTURES / "edge-cases").glob("*.jsonl"))


def fixture(relative: str) -> pathlib.Path:
    path = FIXTURES / relative
    assert path.is_file(), relative
    return path


def trace_of(path: pathlib.Path | str, fmt: str = "auto") -> Trace:
    trace = load_trace(path if isinstance(path, pathlib.Path) else fixture(path), fmt=fmt)
    assert trace is not None
    return trace


def report_of(path: pathlib.Path | str, fmt: str = "auto") -> InspectionReport:
    target = path if isinstance(path, pathlib.Path) else fixture(path)
    return run(target, [InspectionProcessor()], fmt=fmt)["inspection"]


def write_jsonl(path: pathlib.Path, records: list[Any]) -> pathlib.Path:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def jsonl(tmp_path: pathlib.Path):
    """Write records to a temporary JSONL file and return its path."""

    def make(records: list[Any], name: str = "trace.jsonl") -> pathlib.Path:
        return write_jsonl(tmp_path / name, records)

    return make


def cc_record(rtype: str, uuid: str, parent: str | None, message: Any = None, **extra: Any) -> dict[str, Any]:
    """Minimal Claude Code style record for focused tests."""
    record = {
        "parentUuid": parent,
        "isSidechain": False,
        "sessionId": extra.pop("sessionId", "s-1"),
        "version": "2.1.150",
        "type": rtype,
        "uuid": uuid,
        "timestamp": extra.pop("timestamp", "2026-03-14T10:00:00.000Z"),
    }
    if message is not None:
        record["message"] = message
    record.update(extra)
    return record


def walk_strings(value: Any, seen: set[int] | None = None):
    """Yield every string reachable from canonical events (dataclass fields, dicts, lists, sets)."""
    import dataclasses

    if isinstance(value, str):
        yield value
        return
    seen = seen if seen is not None else set()
    if id(value) in seen:  # containers only: equal strings may share one object
        return
    seen.add(id(value))
    if dataclasses.is_dataclass(value):
        for item in dataclasses.fields(value):
            yield from walk_strings(getattr(value, item.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(key, seen)
            yield from walk_strings(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from walk_strings(item, seen)
