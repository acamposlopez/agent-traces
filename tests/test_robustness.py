"""Hostile or corrupted input: adapters and the inspector never raise, and memory stays bounded."""

from __future__ import annotations

import copy
import json
import tracemalloc

import pytest

from agent_traces.adapters import DEFAULT_ADAPTERS, claude_code
from agent_traces.adapters.cursor_cli import CursorCliAdapter
from agent_traces.detection import explicit
from agent_traces.engine import RunContext, run
from agent_traces.model import ToolCategory, ToolResult
from agent_traces.processors import DiscoveryProcessor, InspectionProcessor, inspection
from agent_traces.reader import Limits, RawRecord, ReadStats
from conftest import CLAUDE_FIXTURES, CURSOR_FIXTURES, EDGE_FIXTURES, cc_record

ODD_VALUES = [
    None, True, 0, -1, 10**30, 1.5, "", "x", "\ud800", "x" * 300, "y" * 10_000, [], ["x"], [{}], {},
    {"type": "text"}, {"type": "thinking", "thinking": "z"}, {"x": [1]},
]


def parsed_records(paths):
    for path in paths:
        for line in path.read_bytes().split(b"\n"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


def mutations(record):
    """Yield copies of the record with one field (top level, or one level down) replaced by an odd value."""
    for key, value in record.items():
        for odd in ODD_VALUES:
            mutated = dict(record)
            mutated[key] = odd
            yield mutated
        if isinstance(value, dict):
            for inner in value:
                for odd in ODD_VALUES:
                    mutated = copy.deepcopy(record)
                    mutated[key][inner] = odd
                    yield mutated
        if isinstance(value, list):
            for odd in ODD_VALUES:
                mutated = copy.deepcopy(record)
                mutated[key] = [odd, *value]
                yield mutated


@pytest.mark.parametrize("adapter", DEFAULT_ADAPTERS, ids=lambda a: a.name)
def test_adapters_and_inspector_never_raise_on_mutated_records(adapter):
    records = list(parsed_records(CLAUDE_FIXTURES + CURSOR_FIXTURES + EDGE_FIXTURES))
    shared_meta = adapter.new_metadata()
    context = RunContext(explicit(adapter), adapter, shared_meta, ReadStats(records=1), Limits())
    processors = [InspectionProcessor(include_models=True, include_agent_version=True), DiscoveryProcessor()]
    for processor in processors:  # one inspector and one discovery receive every odd record and event
        processor.start(context)
    line = 0
    for record in records:
        for mutated in mutations(record):
            line += 1
            adapter.sniff(mutated)
            raw = RawRecord(line, data=mutated)
            meta = adapter.new_metadata()
            for processor in processors:
                processor.observe(raw)
            events = list(adapter.parse([raw], meta))
            assert events or meta.normalization, "a record vanished without trace"
            for event in events:
                for processor in processors:
                    processor.handle(event)
    processors[0].finish(context).to_json()
    json.dumps(processors[1].finish(context))
    assert line > 5000


def peak_bytes(function) -> int:
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def inspect_file(path, discover=False):
    processors = [InspectionProcessor(), DiscoveryProcessor()] if discover else [InspectionProcessor()]
    return run(path, processors, fmt="claude-code")["inspection"]


def test_memory_does_not_grow_with_trace_content(tmp_path):
    path = tmp_path / "long.jsonl"
    with path.open("w") as handle:
        parent = None
        for i in range(3000):
            text = f"message {i} " + "lorem ipsum " * 340  # ~4 KB per record
            role = "user" if i % 2 == 0 else "assistant"
            message = {"role": role, "content": text} if role == "user" else {
                "id": f"m{i}", "role": role, "content": [{"type": "text", "text": text}]}
            handle.write(json.dumps(cc_record(role, f"u{i}", parent, message)) + "\n")
            parent = f"u{i}"
    size = path.stat().st_size
    peak = peak_bytes(lambda: inspect_file(path))
    path.unlink()
    assert size > 12 * 2**20
    assert peak < 4 * 2**20, f"peak {peak // 1024} KiB"


def test_hostile_identifiers_and_names_do_not_grow_memory(tmp_path):
    """Unique 24 KiB ids, names and session values on every record: none of them may be retained."""
    path = tmp_path / "hostile.jsonl"
    blob = "z" * 24 * 1024
    with path.open("w") as handle:
        parent = None
        for i in range(64):
            uid = f"{i}-{blob}"
            message = {"id": f"m{uid}", "role": "assistant", "content": [
                {"type": "tool_use", "id": f"t{uid}", "name": f"n{uid}", "input": {}}]}
            record = cc_record("assistant", uid, parent, message, sessionId=f"s{uid}", agentId=f"a{uid}",
                               version=f"v{uid}", cwd=f"/c{uid}", gitBranch=f"b{uid}")
            handle.write(json.dumps(record) + "\n")
            parent = uid
    size = path.stat().st_size
    report: list = []
    peak = peak_bytes(lambda: report.append(inspect_file(path, discover=True)))
    path.unlink()
    assert size > 10 * 2**20
    assert peak < 5 * 2**20, f"peak {peak // 1024} KiB"
    issues = {item["code"] for item in report[0].integrity["issues"]}
    assert {"unparseable_ids", "unparseable_session_ids"} <= issues


def test_deeply_nested_payloads_are_handled_iteratively(jsonl):
    nested: list = []
    for _ in range(900):  # close to the JSON parser's own nesting limit
        nested = [nested]
    path = jsonl([{"type": "canary-future-record", "payload": nested}, cc_record("user", "u1", None, {
        "role": "user", "content": "hi"})])
    report = run(path, [InspectionProcessor()], fmt="claude-code")["inspection"]
    assert report.records["unrecognized"] == {"unknown_type": 1}


def test_streamed_message_flood_is_split_without_losing_text():
    adapter = CursorCliAdapter()
    adapter.max_segment_chars = 1000
    delta = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "a" * 100}]},
             "session_id": "s"}
    records = [RawRecord(i + 1, data={**delta, "timestamp_ms": i}) for i in range(25)]
    meta = adapter.new_metadata()
    events = list(adapter.parse(records, meta))
    assert [len(event.content) for event in events] == [1000, 1000, 500]
    assert "".join(event.content for event in events) == "a" * 2500
    assert meta.normalization["cursor_partial_segment_split"] == 2


def test_pending_tool_calls_are_bounded(monkeypatch):
    monkeypatch.setattr(claude_code, "_MAX_PENDING_CALLS", 3)
    adapter = claude_code.ClaudeCodeAdapter()
    calls = [cc_record("assistant", f"a{i}", None, {"id": f"m{i}", "role": "assistant", "content": [
        {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]}) for i in range(5)]
    results = [cc_record("user", f"u{i}", None, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]}) for i in range(5)]
    meta = adapter.new_metadata()
    records = [RawRecord(n, data=r) for n, r in enumerate(calls + results, start=1)]
    tool_results = [e for e in adapter.parse(records, meta) if isinstance(e, ToolResult)]
    assert [e.category for e in tool_results] == [ToolCategory.UNKNOWN] * 2 + [ToolCategory.SHELL] * 3
    assert meta.normalization["claude_pending_call_forgotten"] == 2


def test_relation_tracking_degrades_gracefully_when_full(monkeypatch, jsonl):
    monkeypatch.setattr(inspection, "MAX_TRACKED_IDS", 10)
    records, parent = [], None
    for i in range(30):
        records.append(cc_record("user", f"u{i}", parent, {"role": "user", "content": "x"}))
        parent = f"u{i}"
    report = run(jsonl(records), [InspectionProcessor()])["inspection"]
    assert report.relations["tracking_truncated"] is True
    assert report.relations["parent_refs_unverified"] > 0
    assert {"code": "relation_tracking_truncated", "severity": "warning", "count": 1} in report.integrity["issues"]


def test_streamed_message_record_cap_bounds_the_classifier_window():
    adapter = CursorCliAdapter()
    adapter.max_segment_records = 5
    delta = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
             "session_id": "s"}
    records = [RawRecord(i + 1, data={**delta, "timestamp_ms": i}) for i in range(12)]
    meta = adapter.new_metadata()
    events = list(adapter.parse(records, meta))
    assert [(e.source.line, e.source.line_end) for e in events] == [(1, 5), (6, 10), (11, 12)]
    assert meta.normalization["cursor_partial_segment_split"] == 2


def test_discovery_structures_stay_bounded(jsonl):
    """Thousands of distinct unknown types and keys: codes, groups and fields stay within fixed limits."""
    records = [{"type": f"type-{i}", f"key-{i}": i, "nested": {f"inner-{i}": i}} for i in range(2000)]
    discovery = run(jsonl(records), [DiscoveryProcessor()], fmt="claude-code")["discovery"]
    assert discovery["opaque_codes"]["record_types"] == 64 and discovery["opaque_codes"]["fields"] == 64
    assert len(discovery["groups"]) <= 256 and discovery["truncated"] is True
    assert all(len(group["fields"]) <= 64 + 3 for group in discovery["groups"])
