"""Adapter for the Cursor Agent CLI ``--output-format stream-json`` output (NDJSON).

Built only from the official reference: https://cursor.com/docs/cli/reference/output-format
(formerly docs.cursor.com/en/cli/reference/output-format). It does NOT cover the local session storage
of the Cursor editor, which is a different integration.

Documented behaviour this adapter relies on:
- events: system (subtype init), user, assistant, tool_call (subtype started/completed), result;
- ``call_id`` correlates the started and completed events of one tool call;
- ``tool_call`` holds one key per tool type: ``readToolCall``, ``writeToolCall`` or ``function``;
- only ``result.success`` is documented for tool results;
- with ``--stream-partial-output`` only assistant events with ``timestamp_ms`` and without
  ``model_call_id`` carry new text; the other two kinds are duplicate flushes and are skipped;
- a run that fails may end without the terminal ``result`` event.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agent_traces.adapters.base import (
    Signal,
    Source,
    SourceAdapter,
    Vocabulary,
    copy_attributes,
    native_id,
    remember_context,
    text_value,
    unrecognized,
)
from agent_traces.model import (
    Event,
    Message,
    MessageOrigin,
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
    parse_epoch_ms,
)
from agent_traces.reader import RawRecord

_API_KEY_SOURCES = frozenset({"env", "flag", "login"})
_D = Source.DOCUMENTED
# Every name below appears in the official output-format reference.
_VOCABULARY = Vocabulary(
    record_types={name: _D for name in ("system", "user", "assistant", "tool_call", "result")},
    discriminators={name: _D for name in ("init", "started", "completed", "success")},
    fields={
        name: _D
        for name in (
            "type", "subtype", "session_id", "apiKeySource", "cwd", "model", "permissionMode", "message", "role",
            "content", "text", "timestamp_ms", "model_call_id", "call_id", "tool_call", "readToolCall",
            "writeToolCall", "function", "name", "arguments", "args", "path", "fileText", "toolCallId", "result",
            "success", "isEmpty", "exceededLimit", "totalLines", "totalChars", "linesCreated", "fileSize",
            "duration_ms", "duration_api_ms", "is_error", "request_id", "tools", "mcp_servers",
        )
    },
    block_types={"text": _D},
    containers=frozenset({"message", "tool_call"}),
)
_RESULT_ATTRIBUTES = (
    ("duration_ms", "duration_ms"),
    ("duration_api_ms", "duration_api_ms"),
    ("request_id", "request_id"),
)


@dataclass(slots=True)
class _Segment:
    """Text deltas of one assistant message streamed with --stream-partial-output."""

    first_line: int
    last_line: int
    timestamp: Any
    parts: list[str] = field(default_factory=list, repr=False)
    chars: int = 0

    def add(self, line: int, text: str) -> None:
        self.last_line = line
        self.parts.append(text)
        self.chars += len(text)

    def to_event(self) -> Message:
        return Message(
            source=SourceRef(line=self.first_line, line_end=self.last_line, record_type="assistant"),
            timestamp=self.timestamp,
            role=Role.ASSISTANT,
            content="".join(self.parts),
        )


class CursorCliAdapter(SourceAdapter):
    name = "cursor-cli"
    display_name = "Cursor Agent CLI (stream-json)"
    version = "0.1.0"
    has_terminal_event = True
    known_record_types = frozenset({"system", "user", "assistant", "tool_call", "result"})
    known_tools: Mapping[str, ToolCategory] = {
        "readToolCall": ToolCategory.FILE_READ,
        "writeToolCall": ToolCategory.FILE_WRITE,
    }
    vocabulary = _VOCABULARY
    # A streamed message longer than this (in characters or in delta records) is split into several messages
    # (counted in normalization) so that a flood of deltas cannot grow memory without limit.
    max_segment_chars: ClassVar[int] = 16 * 1024 * 1024
    max_segment_records: ClassVar[int] = 10_000
    limitations = (
        "cursor-cli: only the documented stream-json events are interpreted; the Cursor editor's local "
        "session storage is a different format and is not supported.",
        "cursor-cli: only readToolCall and writeToolCall are documented tool types; other tool types and "
        "'function' tools are counted in category 'unknown' and their names are not shown.",
        "cursor-cli: only successful tool results (result.success) are documented; any other result shape "
        "is reported with outcome 'unknown', not as an error.",
        "cursor-cli: print mode suppresses thinking events, so reasoning never appears in this format.",
        "cursor-cli: timestamps only exist on --stream-partial-output deltas.",
    )

    def sniff(self, record: Mapping[str, Any]) -> Signal:
        # Positive signatures only: documented events may gain fields over time (consumers must ignore
        # unknown fields), so no single extra key can rule this format out.
        if not isinstance(record.get("session_id"), str):
            return Signal.NONE
        rtype = text_value(record.get("type"))
        tool_call = record.get("tool_call")
        if (
            rtype == "tool_call"
            and record.get("subtype") in ("started", "completed")
            and isinstance(record.get("call_id"), str)
            and isinstance(tool_call, dict)
            and len(tool_call) == 1
        ):
            return Signal.STRONG
        if (
            rtype == "system"
            and record.get("subtype") == "init"
            and text_value(record.get("apiKeySource")) in _API_KEY_SOURCES
        ):
            return Signal.STRONG
        if rtype in ("user", "assistant") and isinstance(record.get("message"), dict):
            return Signal.WEAK
        if rtype in ("system", "result"):
            return Signal.WEAK
        return Signal.NONE

    def parse(self, records: Iterable[RawRecord], meta: TraceMetadata) -> Iterator[Event]:
        partial_mode = False
        segment: _Segment | None = None
        terminal: TerminalState | None = None
        for record in records:
            data = record.data
            is_delta = (
                data is not None
                and data.get("type") == "assistant"
                and "timestamp_ms" in data
                and "model_call_id" not in data
            )
            if segment is not None and not is_delta:
                yield segment.to_event()
                segment = None
            if data is None:
                yield unrecognized(record, self)
                continue
            rtype = text_value(data.get("type"))
            if rtype not in self.known_record_types:
                yield unrecognized(record, self, UnrecognizedReason.UNKNOWN_TYPE)
                continue
            meta.session_ids.note(data.get("session_id"))

            if rtype == "assistant":
                has_ts, has_call = "timestamp_ms" in data, "model_call_id" in data
                if has_ts and has_call:  # buffered flush before a tool call: duplicate text
                    partial_mode = True
                    meta.normalization["cursor_partial_flush_before_tool"] += 1
                    continue
                if has_call:  # undocumented combination: keep it, but do not interpret it
                    yield unrecognized(record, self, UnrecognizedReason.UNKNOWN_SUBTYPE)
                    continue
                if not has_ts and partial_mode:  # final flush at the end of a turn: duplicate text
                    meta.normalization["cursor_partial_final_flush"] += 1
                    continue
                text = _message_text(data)
                if text is None:
                    yield unrecognized(record, self, UnrecognizedReason.INCOMPLETE)
                    continue
                if has_ts:
                    partial_mode = True
                    if segment is not None and (
                        segment.chars + len(text) > self.max_segment_chars
                        or len(segment.parts) >= self.max_segment_records
                    ):
                        yield segment.to_event()
                        segment = None
                        meta.normalization["cursor_partial_segment_split"] += 1
                    if segment is None:
                        segment = _Segment(record.line, record.line, parse_epoch_ms(data["timestamp_ms"]))
                    segment.add(record.line, text)
                    continue
                yield Message(source=_ref(record, rtype), role=Role.ASSISTANT, content=text)

            elif rtype == "user":
                text = _message_text(data)
                if text is None:
                    yield unrecognized(record, self, UnrecognizedReason.INCOMPLETE)
                    continue
                # Documented as "the user's input prompt".
                yield Message(source=_ref(record, rtype), role=Role.USER, content=text, origin=MessageOrigin.PROMPT)

            elif rtype == "tool_call":
                event = self._tool_event(record, data)
                yield event if event is not None else unrecognized(record, self, UnrecognizedReason.INCOMPLETE)

            elif rtype == "system":
                if data.get("subtype") != "init":
                    yield unrecognized(record, self, UnrecognizedReason.UNKNOWN_SUBTYPE)
                    continue
                terminal = None
                meta.models.note(data.get("model"))
                remember_context(meta, "cwd", data.get("cwd"))
                yield SessionMarker(source=_ref(record, rtype), phase=SessionPhase.START, payload=data)

            elif rtype == "result":
                status, terminal = _result_status(data)
                yield SessionMarker(
                    source=_ref(record, rtype),
                    phase=SessionPhase.END,
                    status=status,
                    payload=data.get("result"),
                    attributes=copy_attributes(data, _RESULT_ATTRIBUTES),
                )
        if segment is not None:
            yield segment.to_event()
        meta.terminal_state = terminal or TerminalState.MISSING

    def _tool_event(self, record: RawRecord, data: dict[str, Any]) -> Event | None:
        tool_call = data.get("tool_call")
        if not isinstance(tool_call, dict) or len(tool_call) != 1:
            return None
        ((key, body),) = tool_call.items()
        if not isinstance(body, dict):
            return None
        if key == "function":
            name = body.get("name")
            tool_name = name if isinstance(name, str) else None
            category, arguments = ToolCategory.UNKNOWN, body.get("arguments")
        else:
            tool_name, category, arguments = key, self.tool_category(key), body.get("args")
        call_id = native_id(data.get("call_id"))
        source = _ref(record, "tool_call")
        subtype = data.get("subtype")
        if subtype == "started":
            return ToolCall(source=source, call_id=call_id, tool_name=tool_name, category=category, arguments=arguments)
        if subtype == "completed":
            result = body.get("result")
            if isinstance(result, dict) and "success" in result:
                status, basis, output = ResultStatus.SUCCESS, StatusBasis.EXPLICIT, result["success"]
            else:
                status, basis, output = ResultStatus.UNKNOWN, StatusBasis.UNDETERMINED, result
            return ToolResult(
                source=source,
                call_id=call_id,
                status=status,
                status_basis=basis,
                tool_name=tool_name,
                category=category,
                output=output,
            )
        return unrecognized(record, self, UnrecognizedReason.UNKNOWN_SUBTYPE)


def _ref(record: RawRecord, record_type: str) -> SourceRef:
    return SourceRef(line=record.line, record_type=record_type)


def _message_text(data: Mapping[str, Any]) -> str | None:
    """Concatenated text of a documented message object, or None if it is not interpretable."""
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for block in content:
        if not (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)):
            return None
        parts.append(block["text"])
    return "".join(parts)


def _result_status(data: Mapping[str, Any]) -> tuple[ResultStatus, TerminalState]:
    if data.get("is_error") is True:
        return ResultStatus.ERROR, TerminalState.FAILED
    if data.get("subtype") == "success" and data.get("is_error") in (False, None):
        return ResultStatus.SUCCESS, TerminalState.COMPLETED
    return ResultStatus.UNKNOWN, TerminalState.UNCLEAR
