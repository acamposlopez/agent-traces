"""Adapter for Claude Code session transcripts (``<session-id>.jsonl``).

Anthropic states that this format is internal to Claude Code and changes between versions
(https://code.claude.com/docs/en/sessions). This adapter follows the official documentation where it
exists and public community analyses otherwise (sources and provenance of every name in
``claude_code_vocabulary``). It was built with synthetic fixtures and checked, by structure only, against
one real session of this project.

Main facts it relies on:
- each line has a top-level ``type``; ``user``, ``assistant`` and ``system`` carry the conversation,
  linked by ``uuid`` / ``parentUuid``; many other types are operational metadata;
- one assistant response with several content blocks is written as several lines sharing ``message.id``;
- tool results travel in ``user`` lines as ``tool_result`` blocks whose ``tool_use_id`` matches the
  ``tool_use`` block ``id``; ``toolUseResult`` holds a structured copy;
- subagent lines carry ``isSidechain: true`` and ``agentId``; recent versions store them in separate files;
- some records use snake_case keys (such as ``session_id``) next to camelCase ones: no single key decides
  the format.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from agent_traces.adapters.base import (
    MAX_ID_LENGTH,
    Signal,
    SourceAdapter,
    copy_attributes,
    native_id,
    remember_context,
    text_value,
    unrecognized,
)
from agent_traces.adapters.claude_code_vocabulary import (
    KNOWN_TOOLS,
    METADATA_SPECS,
    ORIGIN_MARKERS,
    SYSTEM_SPECS,
    VOCABULARY,
    MetadataSpec,
)
from agent_traces.model import (
    ContentKind,
    ErrorEvent,
    ErrorScope,
    Event,
    Message,
    MessageOrigin,
    MetadataStructure,
    Notice,
    Reasoning,
    ResultStatus,
    Role,
    SourceRef,
    StatusBasis,
    ToolCall,
    ToolCategory,
    ToolResult,
    TraceMetadata,
    Unparseable,
    Unrecognized,
    UnrecognizedReason,
    parse_iso_timestamp,
)
from agent_traces.reader import RawRecord

_CONVERSATION_TYPES = frozenset({"user", "assistant", "system"})
_SYNTHETIC_MODEL = "<synthetic>"
# Technical attributes kept verbatim when present (source key -> canonical name, see KNOWN_ATTRIBUTES).
_RECORD_ATTRIBUTES = (
    ("logicalParentUuid", "logical_parent_id"),
    ("sourceToolUseID", "source_tool_use_id"),
    ("sourceToolAssistantUUID", "source_tool_assistant_id"),
    ("isMeta", "is_meta"),
    ("isCompactSummary", "is_compact_summary"),
    ("requestId", "request_id"),
)
_MESSAGE_ATTRIBUTES = (("stop_reason", "stop_reason"), ("usage", "usage"))
_MAX_PENDING_CALLS = 50_000  # calls still waiting for their result; the oldest are forgotten beyond this


class ClaudeCodeAdapter(SourceAdapter):
    name = "claude-code"
    display_name = "Claude Code session transcript (JSONL)"
    version = "0.1.0"
    has_terminal_event = False
    known_record_types = _CONVERSATION_TYPES | frozenset(METADATA_SPECS)
    known_tools: Mapping[str, ToolCategory] = KNOWN_TOOLS
    vocabulary = VOCABULARY
    limitations = (
        "claude-code: the transcript format is internal to Claude Code and changes between versions; this "
        "adapter follows public documentation and community analyses and was checked, by structure only, "
        "against one real session.",
        "claude-code: metadata records (snapshots, titles, queue operations, attachments, progress, modes...) "
        "are identified but their content is not interpreted; attachment, last-prompt and similar records may "
        "hold prompts, injected context or paths and are not exportable by default.",
        "claude-code: recent versions store subagent transcripts in separate files; each file is inspected on "
        "its own and links to subagents are reported as references only.",
        "claude-code: tool results without is_error are counted as successful by protocol default (Anthropic "
        "Messages API), never as confirmed successes.",
        "claude-code: assistant records flagged isApiErrorMessage are counted as API errors (interpretation "
        "based on the field name).",
        "claude-code: the origin of user-role messages (prompt, command, command output, interruption, "
        "injected) comes from source flags and harness markers; 'prompt' means no marker was found, not that a "
        "person typed it.",
    )

    def sniff(self, record: Mapping[str, Any]) -> Signal:
        # Positive signatures only (several keys and types together). Extra or snake_case keys are neutral.
        rtype = text_value(record.get("type"))
        if (
            rtype in _CONVERSATION_TYPES
            and isinstance(record.get("uuid"), str)
            and "parentUuid" in record
            and isinstance(record.get("sessionId"), str)
            and isinstance(record.get("timestamp"), str)
        ):
            return Signal.STRONG
        if (
            rtype == "file-history-snapshot"
            and isinstance(record.get("messageId"), str)
            and isinstance(record.get("snapshot"), dict)
        ):
            return Signal.STRONG
        if rtype in METADATA_SPECS or (rtype in _CONVERSATION_TYPES and isinstance(record.get("sessionId"), str)):
            return Signal.WEAK
        return Signal.NONE

    def tool_category(self, name: Any) -> ToolCategory:
        if isinstance(name, str) and name.startswith("mcp__"):
            return ToolCategory.MCP
        return super().tool_category(name)

    def parse(self, records: Iterable[RawRecord], meta: TraceMetadata) -> Iterator[Event]:
        pending: dict[str, tuple[str | None, ToolCategory]] = {}  # call_id -> (tool name, category)
        for record in records:
            data = record.data
            if data is None:
                yield unrecognized(record, self)
                continue
            rtype = text_value(data.get("type"))
            common = _common(data)
            spec = METADATA_SPECS.get(rtype) if rtype is not None else None
            if spec is not None:
                yield self._notice(record, rtype, spec, data, common)
                continue
            if rtype not in _CONVERSATION_TYPES:
                yield unrecognized(record, self, UnrecognizedReason.UNKNOWN_TYPE, **common)
                continue
            meta.session_ids.note(data.get("sessionId"))
            meta.agent_versions.note(data.get("version"))
            remember_context(meta, "cwd", data.get("cwd"))
            remember_context(meta, "git_branch", data.get("gitBranch"))
            if rtype == "user":
                events = self._user(record, data, common, pending)
            elif rtype == "assistant":
                events = self._assistant(record, data, common, pending, meta)
            else:
                events = iter((self._system(record, data, common),))
            attributes = _record_attributes(rtype, data)
            for event in events:
                if attributes and not isinstance(event, Unrecognized):
                    event.attributes.update(attributes)  # each event gets its own dict
                yield event

    def _notice(
        self, record: RawRecord, rtype: str, spec: MetadataSpec, data: dict[str, Any], common: dict[str, Any]
    ) -> Notice:
        notice_type, verified = _classify(rtype, spec, data)
        return Notice(
            source=SourceRef(line=record.line, record_type=rtype),
            notice_type=notice_type,
            metadata_class=spec.metadata_class,
            structure=MetadataStructure.VERIFIED if verified else MetadataStructure.NAME_ONLY,
            **common,
        )

    def _user(
        self,
        record: RawRecord,
        data: dict[str, Any],
        common: dict[str, Any],
        pending: dict[str, tuple[str | None, ToolCategory]],
    ) -> Iterator[Event]:
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            origin = _user_origin(data, content, carries_tool_result=False)
            yield Message(source=_ref(record, "user"), role=Role.USER, content=content, origin=origin, **common)
            return
        if not isinstance(content, list) or not content:
            yield unrecognized(record, self, UnrecognizedReason.INCOMPLETE, **common)
            return
        result_blocks = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_result")
        details = data.get("toolUseResult") if result_blocks == 1 else None
        for index, block in enumerate(content):
            source = _ref(record, "user", index)
            btype = text_value(block.get("type")) if isinstance(block, dict) else None
            if btype == "text":
                text = block.get("text")
                origin = _user_origin(data, text, carries_tool_result=result_blocks > 0)
                text = text if isinstance(text, str) else Unparseable("invalid_text", text)
                yield Message(source=source, role=Role.USER, content=text, origin=origin, **common)
            elif btype in ("image", "document"):
                kind = ContentKind.IMAGE if btype == "image" else ContentKind.DOCUMENT
                origin = _user_origin(data, None, carries_tool_result=result_blocks > 0)
                yield Message(
                    source=source, role=Role.USER, content_kind=kind, content=block, origin=origin, **common
                )
            elif btype == "tool_result":
                call_id = native_id(block.get("tool_use_id"))
                unknown_tool: tuple[str | None, ToolCategory] = (None, ToolCategory.UNKNOWN)
                tool_name, category = pending.pop(call_id, unknown_tool) if isinstance(call_id, str) else unknown_tool
                status, basis = _result_status(block.get("is_error"))
                yield ToolResult(
                    source=source,
                    call_id=call_id,
                    status=status,
                    status_basis=basis,
                    tool_name=tool_name,
                    category=category,
                    output=block.get("content"),
                    details=details,
                    subagent_ref=native_id(details.get("agentId")) if isinstance(details, dict) else None,
                    **common,
                )
            else:
                yield _unrecognized_block(source, block, common)

    def _assistant(
        self,
        record: RawRecord,
        data: dict[str, Any],
        common: dict[str, Any],
        pending: dict[str, tuple[str | None, ToolCategory]],
        meta: TraceMetadata,
    ) -> Iterator[Event]:
        message = data.get("message")
        if not isinstance(message, dict):
            yield unrecognized(record, self, UnrecognizedReason.INCOMPLETE, **common)
            return
        message_id = native_id(message.get("id"))
        model = message.get("model") if isinstance(message.get("model"), str) else None
        if model == _SYNTHETIC_MODEL:
            model = None
        else:
            meta.models.note(message.get("model"))
        content = message.get("content")
        if data.get("isApiErrorMessage") is True:
            yield ErrorEvent(
                source=_ref(record, "assistant"),
                scope=ErrorScope.API,
                code="api_error_message",
                detail=content,
                message_id=message_id,
                **common,
            )
            return
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            yield unrecognized(record, self, UnrecognizedReason.INCOMPLETE, message_id=message_id, **common)
            return
        for index, block in enumerate(content):
            source = _ref(record, "assistant", index)
            btype = text_value(block.get("type")) if isinstance(block, dict) else None
            ids = dict(message_id=message_id, **common)
            if btype == "text":
                text = block.get("text")
                text = text if isinstance(text, str) else Unparseable("invalid_text", text)
                yield Message(source=source, role=Role.ASSISTANT, content=text, model=model, **ids)
            elif btype in self.reasoning_block_types:
                # The reasoning content is dropped here, on purpose: it must never reach reports or exports.
                yield Reasoning(source=source, redacted=btype == "redacted_thinking", **ids)
            elif btype == "tool_use":
                call_id = native_id(block.get("id"))
                name = block.get("name") if isinstance(block.get("name"), str) else None
                category = self.tool_category(name)
                if isinstance(call_id, str):
                    short_name = name if name is not None and len(name) <= MAX_ID_LENGTH else None
                    pending[call_id] = (short_name, category)
                    if len(pending) > _MAX_PENDING_CALLS:
                        del pending[next(iter(pending))]
                        meta.normalization["claude_pending_call_forgotten"] += 1
                yield ToolCall(
                    source=source,
                    call_id=call_id,
                    tool_name=name,
                    category=category,
                    arguments=block.get("input"),
                    **ids,
                )
            else:
                yield _unrecognized_block(source, block, ids)

    def _system(self, record: RawRecord, data: dict[str, Any], common: dict[str, Any]) -> Event:
        subtype = text_value(data.get("subtype"))
        if subtype == "api_error":
            return ErrorEvent(
                source=_ref(record, "system"), scope=ErrorScope.API, code="api_error", detail=data, **common
            )
        spec = SYSTEM_SPECS.get(subtype) if subtype is not None else None
        if spec is None:
            return unrecognized(record, self, UnrecognizedReason.UNKNOWN_SUBTYPE, **common)
        verified = bool(spec.expected) and spec.expected <= data.keys()
        payload = {key: data.get(key) for key in ("content", "compactMetadata", "level") if key in data}
        return Notice(
            source=_ref(record, "system"),
            notice_type=f"system/{subtype}",
            metadata_class=spec.metadata_class,
            structure=MetadataStructure.VERIFIED if verified else MetadataStructure.NAME_ONLY,
            payload=payload,
            **common,
        )


def _classify(rtype: str, spec: MetadataSpec, data: Mapping[str, Any]) -> tuple[str, bool]:
    """Notice type (refined by a vocabulary-gated discriminator value) and whether the structure is verified."""
    notice_type = rtype
    discriminator_ok = True
    if spec.discriminator is not None:
        value: Any = data
        for key in spec.discriminator:
            value = value.get(key) if isinstance(value, dict) else None
        known = isinstance(value, str) and value in VOCABULARY.discriminators
        if isinstance(value, str):
            notice_type = f"{rtype}/{value}" if known else f"{rtype}/other"
        # Fields differ by subtype: the structure is only verified for subtypes in the vocabulary.
        discriminator_ok = known
    verified = bool(spec.expected) and spec.expected <= data.keys() and discriminator_ok
    return notice_type, verified


def _user_origin(data: Mapping[str, Any], text: Any, *, carries_tool_result: bool) -> MessageOrigin:
    """Never assume a person: source flags, tool links and harness markers decide; 'prompt' means none."""
    if data.get("isMeta") is True or data.get("isCompactSummary") is True:
        return MessageOrigin.INJECTED
    if carries_tool_result or "toolUseResult" in data or "sourceToolUseID" in data:
        return MessageOrigin.INJECTED  # text that travels with, or comes from, a tool result
    if isinstance(text, str):
        stripped = text.lstrip()
        for marker, origin in ORIGIN_MARKERS:
            if stripped.startswith(marker):
                return origin
    return MessageOrigin.PROMPT


def _common(data: Mapping[str, Any]) -> dict[str, Any]:
    sidechain = data.get("isSidechain")
    return {
        "timestamp": parse_iso_timestamp(data.get("timestamp")),
        "event_id": native_id(data.get("uuid")),
        "parent_id": native_id(data.get("parentUuid")),
        "agent_id": native_id(data.get("agentId")),
        "is_sidechain": sidechain if isinstance(sidechain, bool) else None,
    }


def _record_attributes(rtype: str, data: Mapping[str, Any]) -> dict[str, Any]:
    attributes = copy_attributes(data, _RECORD_ATTRIBUTES)
    message = data.get("message")
    if rtype == "assistant" and isinstance(message, dict):
        attributes.update(copy_attributes(message, _MESSAGE_ATTRIBUTES))
    return attributes


def _result_status(is_error: Any) -> tuple[ResultStatus, StatusBasis]:
    if is_error is True:
        return ResultStatus.ERROR, StatusBasis.EXPLICIT
    if is_error is False:
        return ResultStatus.SUCCESS, StatusBasis.EXPLICIT
    if is_error is None:  # absent: the Anthropic Messages API default is false (reported as such)
        return ResultStatus.SUCCESS, StatusBasis.PROTOCOL_DEFAULT
    return ResultStatus.UNKNOWN, StatusBasis.UNDETERMINED


def _ref(record: RawRecord, record_type: str, block: int | None = None) -> SourceRef:
    return SourceRef(line=record.line, record_type=record_type, block=block)


def _unrecognized_block(source: SourceRef, block: Any, common: dict[str, Any]) -> Event:
    return Unrecognized(source=source, reason=UnrecognizedReason.UNKNOWN_BLOCK, raw=block, **common)
