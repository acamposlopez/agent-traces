"""A new source adapter plugs in without touching the engine, detection, inspection or CLI.

The test adapter below follows the event/item lifecycle that OpenAI documents for ``codex exec --json``
(https://developers.openai.com/codex/noninteractive): ``thread.started``, ``turn.*``, ``item.started`` /
``item.completed`` with an item ``id`` and ``type`` (agent_message, reasoning, command_execution...) and
``error``. It is NOT a supported Codex adapter: field details beyond that documented sample are illustrative.
It only proves that such a format fits the canonical model and the unchanged engine.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from agent_traces import cli
from agent_traces.adapters import DEFAULT_ADAPTERS
from agent_traces.adapters.base import Signal, SourceAdapter, copy_attributes, native_id, text_value, unrecognized
from agent_traces.engine import load_trace, run
from agent_traces.model import (
    ErrorEvent,
    ErrorScope,
    Event,
    Message,
    Notice,
    Reasoning,
    ResultStatus,
    Role,
    SessionMarker,
    SessionPhase,
    SourceRef,
    StatusBasis,
    TerminalState,
    ToolCall,
    ToolCategory,
    ToolResult,
    TraceMetadata,
    UnrecognizedReason,
    semantic_view,
)
from agent_traces.processors import InspectionProcessor
from agent_traces.reader import RawRecord
from conftest import walk_strings

SRC = pathlib.Path(__file__).parents[1] / "src" / "agent_traces"
CORE_MODULES = ["engine.py", "detection.py", "model.py", "reader.py", "processors/inspection.py", "adapters/base.py"]


class ItemLifecycleToyAdapter(SourceAdapter):
    name = "item-lifecycle-toy"
    display_name = "Item lifecycle toy format (tests only)"
    version = "0.0.1"
    has_terminal_event = True
    known_record_types = frozenset(
        {"thread.started", "turn.started", "turn.completed", "turn.failed", "item.started", "item.completed", "error"}
    )
    known_tools: Mapping[str, ToolCategory] = {
        "command_execution": ToolCategory.SHELL,
        "file_change": ToolCategory.FILE_WRITE,
        "mcp_tool_call": ToolCategory.MCP,
        "web_search": ToolCategory.WEB,
    }
    reasoning_block_types = frozenset({"reasoning"})

    def sniff(self, record: Mapping[str, Any]) -> Signal:
        rtype = text_value(record.get("type"))
        return Signal.STRONG if rtype in ("thread.started", "item.started", "item.completed") else Signal.NONE

    def parse(self, records: Iterable[RawRecord], meta: TraceMetadata) -> Iterator[Event]:
        terminal = None
        for record in records:
            data = record.data
            if data is None:
                yield unrecognized(record, self)
                continue
            rtype = text_value(data.get("type"))
            source = SourceRef(line=record.line, record_type=self.safe_record_type(rtype))
            if rtype == "thread.started":
                meta.session_ids.note(data.get("thread_id"))
                terminal = None
                yield SessionMarker(source=source, phase=SessionPhase.START)
            elif rtype == "turn.started":
                yield Notice(source=source, notice_type="turn.started")
            elif rtype == "turn.completed":
                terminal = TerminalState.COMPLETED
                yield SessionMarker(source=source, phase=SessionPhase.END, status=ResultStatus.SUCCESS,
                                    attributes=copy_attributes(data, (("usage", "usage"),)))
            elif rtype == "turn.failed":
                terminal = TerminalState.FAILED
                yield ErrorEvent(source=source, scope=ErrorScope.RUN, code="turn_failed", detail=data)
            elif rtype == "error":
                yield ErrorEvent(source=source, scope=ErrorScope.SYSTEM, code="error", detail=data)
            elif rtype in ("item.started", "item.completed"):
                yield self._item(record, source, data.get("item"), started=rtype == "item.started")
            else:
                yield unrecognized(record, self, UnrecognizedReason.UNKNOWN_TYPE)
        meta.terminal_state = terminal or TerminalState.MISSING

    def _item(self, record: RawRecord, source: SourceRef, item: Any, *, started: bool) -> Event:
        if not isinstance(item, dict):
            return unrecognized(record, self, UnrecognizedReason.INCOMPLETE)
        itype = text_value(item.get("type"))
        item_id = native_id(item.get("id"))
        if itype in self.reasoning_block_types:
            return Reasoning(source=source)  # content never stored
        if itype == "agent_message" and not started:
            return Message(source=source, role=Role.ASSISTANT, content=item.get("text"))
        if itype in self.known_tools:
            category = self.known_tools[itype]
            if started:
                return ToolCall(source=source, call_id=item_id, tool_name=itype, category=category, arguments=item)
            status = {"completed": ResultStatus.SUCCESS, "failed": ResultStatus.ERROR}.get(item.get("status"))
            return ToolResult(
                source=source,
                call_id=item_id,
                status=status or ResultStatus.UNKNOWN,
                status_basis=StatusBasis.EXPLICIT if status else StatusBasis.UNDETERMINED,
                tool_name=itype,
                category=category,
                output=item,
            )
        return unrecognized(record, self, UnrecognizedReason.UNKNOWN_SUBTYPE)


ADAPTERS = (*DEFAULT_ADAPTERS, ItemLifecycleToyAdapter())
SAMPLE = [
    {"type": "thread.started", "thread_id": "0199a213-81c0-7800-8aa1-canary000001"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "CANARY-THINKING plan"}},
    {"type": "item.started", "item": {"id": "item_1", "type": "command_execution",
                                      "command": "bash -lc ls /home/canary", "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_1", "type": "command_execution",
                                        "command": "bash -lc ls /home/canary", "status": "completed"}},
    {"type": "item.completed", "item": {"id": "item_2", "type": "future_item",
                                        "children": [{"type": "reasoning", "text": "CANARY-THINKING nested"}]}},
    {"type": "item.completed", "item": {"id": "item_3", "type": "agent_message", "text": "CANARY: docs and sdk."}},
    {"type": "turn.completed", "usage": {"input_tokens": 24763, "cached_input_tokens": 24448, "output_tokens": 122,
                                         "reasoning_output_tokens": 0}},
]


def test_new_adapter_is_detected_and_inspected_by_the_unchanged_engine(jsonl):
    report = run(jsonl(SAMPLE), [InspectionProcessor()], adapters=ADAPTERS)["inspection"]
    assert report.source["format"] == "item-lifecycle-toy" and report.source["detection"] == "detected"
    assert report.counts["messages"] == {"total": 1, "assistant": 1}
    assert (report.counts["tool_calls"], report.counts["tool_results"], report.counts["reasoning_blocks"]) == (1, 1, 1)
    assert report.tools["known_tools"] == {"command_execution": 1}
    assert report.relations["paired_calls"] == 1
    assert report.session["terminal_state"] == "completed"
    assert report.records["unrecognized"] == {"unknown_subtype": 1}
    assert report.integrity["status"] == "warnings"


def test_new_adapter_produces_the_common_semantic_sequence(jsonl):
    trace = load_trace(jsonl(SAMPLE), adapters=ADAPTERS)
    assert semantic_view(trace.events) == [
        ("tool_call", 1, "shell"),
        ("tool_result", 1, "shell", "success"),
        ("message", "assistant", "text"),
    ]
    end = next(e for e in trace.events if isinstance(e, SessionMarker) and e.phase is SessionPhase.END)
    assert end.attributes["usage"]["output_tokens"] == 122


def test_engine_removes_the_reasoning_types_declared_by_the_new_adapter(jsonl):
    trace = load_trace(jsonl(SAMPLE), adapters=ADAPTERS)
    assert not any("CANARY-THINKING" in s for s in walk_strings(trace.events))


def test_cli_offers_the_new_format(jsonl, capsys):
    code = cli.main(["inspect", str(jsonl(SAMPLE)), "--format", "item-lifecycle-toy"], adapters=ADAPTERS)
    out = capsys.readouterr().out
    assert code == 0 and "item-lifecycle-toy" in out and "canary" not in out.lower()


def test_core_modules_do_not_depend_on_specific_adapters():
    for relative in CORE_MODULES:
        tree = ast.parse((SRC / relative).read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        specific = {name for name in imported if name.startswith("agent_traces.adapters.")} - {
            "agent_traces.adapters.base"
        }
        assert not specific, relative
