"""Canonical trace model: the common, versioned representation every source adapter produces.

Design rules:
- Nothing is invented: a field missing from the source is ``None``; a field present but not interpretable
  is an ``Unparseable`` wrapper (``state_of`` tells the three cases apart). When a value comes from a
  documented protocol default instead of the record itself, the event says so (``StatusBasis``).
- Order is preserved: events are emitted in source order and numbered by the engine (``Event.seq``).
- Relations are kept as native ids (call ids, parent ids, message ids, agent ids), never guessed.
- Technical metadata without conversational content (stop reason, token usage, request ids, durations,
  logical parents) is kept in ``Event.attributes`` under the names listed in ``KNOWN_ATTRIBUTES``.
- Privacy: every field that may hold original content or native identifiers is declared with ``repr=False``
  so it never shows up in reprs, logs or tracebacks. Reasoning/thinking content is never stored: adapters
  drop it and the engine removes reasoning blocks nested in any raw payload (``omit_reasoning``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "0.1.0"

# Technical metadata an adapter may keep in ``Event.attributes``. Only these names are allowed, and only
# when the source record carries the value (nothing is filled in).
KNOWN_ATTRIBUTES = frozenset(
    {
        "stop_reason",  # why the model stopped (e.g. end_turn, tool_use)
        "usage",  # token accounting as reported by the source
        "request_id",  # provider or agent request id
        "duration_ms",  # run duration reported by the source
        "duration_api_ms",
        "logical_parent_id",  # logical parent across a compaction boundary
        "source_tool_use_id",  # tool call that produced this record
        "source_tool_assistant_id",  # assistant record that issued that tool call
        "is_meta",  # flags kept verbatim; Message.injected summarises them
        "is_compact_summary",
    }
)

# Event fields that may hold raw nested data from the source; the engine cleans them (omit_reasoning).
RAW_FIELDS = ("content", "arguments", "output", "details", "detail", "payload", "raw", "attributes")

MAX_TRACKED_VALUES = 32  # distinct values kept per session-level field
MAX_VALUE_LENGTH = 256  # longer session ids, versions or model names are treated as unparseable


class FieldState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True, slots=True)
class Unparseable:
    """A field present in the source whose value could not be interpreted.

    ``reason`` is a fixed code chosen by the adapter (safe to report); ``raw`` keeps the original value in
    memory only; it is never reported.
    """

    reason: str
    raw: Any = field(default=None, repr=False, compare=False)


def state_of(value: Any) -> FieldState:
    if value is None:
        return FieldState.ABSENT
    if isinstance(value, Unparseable):
        return FieldState.UNPARSEABLE
    return FieldState.PRESENT


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    OTHER = "other"


class MessageOrigin(StrEnum):
    """Who produced a message, as far as the source shows it. Decided by flags and markers, never guessed:
    a user-role message is not treated as human input when it carries harness output or injected context."""

    PROMPT = "prompt"  # user-role text without harness markers (typed input; not proof of a human)
    COMMAND = "command"  # command invocation recorded by the harness (e.g. a slash command)
    COMMAND_OUTPUT = "command_output"  # output of a local command, produced by the harness
    INTERRUPTION = "interruption"  # harness notice that the run was interrupted
    INJECTED = "injected"  # flagged by the source as meta or summary, or wrapped in harness markup
    MODEL = "model"  # assistant output
    UNKNOWN = "unknown"


class MetadataStructure(StrEnum):
    VERIFIED = "verified"  # type known and the record has the structural fields documented or observed for it
    NAME_ONLY = "name_only"  # type known by name only: its structure is not documented or not present


class ExportPolicy(StrEnum):
    """Default export treatment of an event. No export exists; this fixes a conservative default policy."""

    AFTER_SCRUBBING = "after_scrubbing"  # conversational events (default label; nothing is exported)
    NOT_BY_DEFAULT = "not_by_default"  # metadata and injected context: excluded by default
    NEVER = "never"  # unrecognised data and reasoning


class RecordCategory(StrEnum):
    """What the parser made of one source record."""

    SEMANTIC = "semantic"  # produced at least one interpreted conversational event
    METADATA = "metadata"  # identified operational record whose content is not interpreted
    UNKNOWN = "unknown"  # not interpretable: unknown type or subtype, incomplete or malformed
    SKIPPED = "skipped"  # deliberately skipped by a documented normalisation (duplicate flushes)


class ToolCategory(StrEnum):
    SHELL = "shell"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SEARCH = "search"
    WEB = "web"
    SUBAGENT = "subagent"
    PLANNING = "planning"
    USER_INTERACTION = "user_interaction"
    MCP = "mcp"
    OTHER = "other"  # recognised tool without a more specific category
    UNKNOWN = "unknown"  # tool name/type not in the adapter's allowlist


class ResultStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    UNKNOWN = "unknown"  # the source does not say, or uses an undocumented shape


class StatusBasis(StrEnum):
    """Where a tool result status comes from."""

    EXPLICIT = "explicit"  # the record states it (e.g. is_error: true / false, result.success)
    PROTOCOL_DEFAULT = "protocol_default"  # the field is absent and the documented default applies
    UNDETERMINED = "undetermined"  # status is unknown


class ErrorScope(StrEnum):
    API = "api"  # model/provider request failed
    RUN = "run"  # the agent run itself failed
    SYSTEM = "system"  # other error reported by the agent harness


class SessionPhase(StrEnum):
    START = "start"
    END = "end"


class UnrecognizedReason(StrEnum):
    MALFORMED_JSON = "malformed_json"
    INVALID_ENCODING = "invalid_encoding"
    OVERSIZED = "oversized"
    NOT_AN_OBJECT = "not_an_object"
    UNKNOWN_TYPE = "unknown_type"
    UNKNOWN_SUBTYPE = "unknown_subtype"
    UNKNOWN_BLOCK = "unknown_block"
    INCOMPLETE = "incomplete"  # known record type with required fields missing or invalid


class TerminalState(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    UNCLEAR = "unclear"  # the terminal event is there but its outcome is not interpretable
    MISSING = "missing"  # the format defines a terminal event and it is not there
    NOT_APPLICABLE = "not_applicable"  # the format has no terminal event


@dataclass(slots=True, kw_only=True)
class SourceRef:
    """Where an event comes from. ``record_type`` is only set when the adapter allowlists it."""

    line: int
    line_end: int | None = None  # last line when the event aggregates several records (streamed deltas)
    record_type: str | None = None
    block: int | None = None  # index of the content block inside the record


@dataclass(slots=True, kw_only=True)
class Event:
    source: SourceRef
    seq: int = -1  # position in the canonical sequence, assigned by the engine
    timestamp: datetime | Unparseable | None = None
    event_id: str | Unparseable | None = field(default=None, repr=False)  # native id of the source record
    parent_id: str | Unparseable | None = field(default=None, repr=False)  # native id of the parent record
    message_id: str | Unparseable | None = field(default=None, repr=False)  # groups one logical message
    agent_id: str | Unparseable | None = field(default=None, repr=False)  # subagent id; None = main/unknown
    is_sidechain: bool | None = None  # True when the source marks the event as part of a subagent run
    attributes: dict[str, Any] = field(default_factory=dict, repr=False)  # see KNOWN_ATTRIBUTES


@dataclass(slots=True, kw_only=True)
class Message(Event):
    role: Role
    content_kind: ContentKind = ContentKind.TEXT
    content: Any = field(default=None, repr=False)  # text for TEXT, raw block otherwise
    model: str | None = field(default=None, repr=False)
    origin: MessageOrigin | None = None  # defaults to MODEL for assistant messages, UNKNOWN otherwise

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = MessageOrigin.MODEL if self.role is Role.ASSISTANT else MessageOrigin.UNKNOWN

    @property
    def injected(self) -> bool:
        """Produced by the agent harness rather than typed: command output, interruptions, injected context."""
        return self.origin in (MessageOrigin.COMMAND_OUTPUT, MessageOrigin.INTERRUPTION, MessageOrigin.INJECTED)


@dataclass(slots=True, kw_only=True)
class Reasoning(Event):
    """A reasoning/thinking block existed here. Its content is deliberately never stored."""

    redacted: bool = False


@dataclass(slots=True, kw_only=True)
class ToolCall(Event):
    call_id: str | Unparseable | None = field(repr=False)
    tool_name: str | None = field(default=None, repr=False)
    category: ToolCategory = ToolCategory.UNKNOWN
    arguments: Any = field(default=None, repr=False)


@dataclass(slots=True, kw_only=True)
class ToolResult(Event):
    call_id: str | Unparseable | None = field(repr=False)
    status: ResultStatus = ResultStatus.UNKNOWN
    status_basis: StatusBasis = StatusBasis.UNDETERMINED
    tool_name: str | None = field(default=None, repr=False)
    category: ToolCategory = ToolCategory.UNKNOWN
    output: Any = field(default=None, repr=False)
    details: Any = field(default=None, repr=False)  # source-specific structured result
    subagent_ref: str | Unparseable | None = field(default=None, repr=False)  # subagent it reports on


@dataclass(slots=True, kw_only=True)
class ErrorEvent(Event):
    scope: ErrorScope
    code: str | None = None  # fixed, allowlisted code; never a free-text message
    detail: Any = field(default=None, repr=False)


@dataclass(slots=True, kw_only=True)
class SessionMarker(Event):
    phase: SessionPhase
    status: ResultStatus | None = None
    payload: Any = field(default=None, repr=False)


@dataclass(slots=True, kw_only=True)
class Notice(Event):
    """Operational record identified by the adapter whose content is not interpreted (compaction boundary,
    file snapshot, queue operation, attachment...). It may still hold prompts, paths or other free text, so
    it is never exportable by default."""

    notice_type: str  # allowlisted by the adapter, safe to report (e.g. "progress/hook_progress")
    metadata_class: str = "other"  # allowlisted class (title, config, attachment, prompt, progress...)
    structure: MetadataStructure = MetadataStructure.NAME_ONLY
    payload: Any = field(default=None, repr=False)


@dataclass(slots=True, kw_only=True)
class Unrecognized(Event):
    """A record or content block the adapter could not interpret.

    Never considered safe: it is only counted by reason, its native type is only kept when allowlisted, and
    its parsed value stays in memory (``raw``, reasoning removed) for later stages that must treat it as
    sensitive.
    """

    reason: UnrecognizedReason
    raw: Any = field(default=None, repr=False)


@dataclass(slots=True)
class ValueTracker:
    """Distinct values of a session-level field plus how often it was present, absent or unparseable.

    Bounded: at most MAX_TRACKED_VALUES distinct values are kept (``capped`` is set beyond that), and values
    longer than MAX_VALUE_LENGTH count as unparseable, so a hostile file cannot grow memory through them.
    """

    values: list[str] = field(default_factory=list, repr=False)
    states: Counter[FieldState] = field(default_factory=Counter)
    capped: bool = False

    def note(self, value: Any) -> None:
        if value is None:
            self.states[FieldState.ABSENT] += 1
        elif isinstance(value, str) and 0 < len(value) <= MAX_VALUE_LENGTH:
            self.states[FieldState.PRESENT] += 1
            if value not in self.values:
                if len(self.values) < MAX_TRACKED_VALUES:
                    self.values.append(value)
                else:
                    self.capped = True
        else:
            self.states[FieldState.UNPARSEABLE] += 1

    @property
    def state(self) -> FieldState:
        if self.states[FieldState.PRESENT]:
            return FieldState.PRESENT
        if self.states[FieldState.UNPARSEABLE]:
            return FieldState.UNPARSEABLE
        return FieldState.ABSENT


@dataclass(slots=True)
class TraceMetadata:
    """Session-level facts and provenance. Filled by the adapter while it parses."""

    source_format: str
    adapter_version: str
    schema_version: str = SCHEMA_VERSION
    session_ids: ValueTracker = field(default_factory=ValueTracker)
    agent_versions: ValueTracker = field(default_factory=ValueTracker)
    models: ValueTracker = field(default_factory=ValueTracker)
    terminal_state: TerminalState = TerminalState.NOT_APPLICABLE
    # Distinct working directories, branches...: kept in memory, never reported, bounded.
    context: dict[str, list[str]] = field(default_factory=dict, repr=False)
    # Normalisation actions with fixed codes (duplicate flushes skipped, oversized segments split...).
    normalization: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class Trace:
    """A whole trace in memory. Never written to disk."""

    metadata: TraceMetadata
    events: list[Event]


def parse_iso_timestamp(value: Any) -> datetime | Unparseable | None:
    if value is None:
        return None
    if isinstance(value, str) and len(value) <= 64:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return Unparseable("invalid_timestamp", value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return Unparseable("invalid_timestamp", value)


def parse_epoch_ms(value: Any) -> datetime | Unparseable | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value / 1000, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return Unparseable("invalid_timestamp", value)
    return Unparseable("invalid_timestamp", value)


def omit_reasoning(value: Any, block_types: Iterable[str]) -> Any:
    """Replace every dict whose ``type`` is a reasoning block type with a stub that keeps only the type.

    Works in place on nested dicts and lists, iteratively (deep JSON cannot exhaust the stack) and returns
    the possibly replaced root. The stub records that a reasoning block existed without its content.
    """
    types = frozenset(block_types)
    if not types:
        return value
    if _is_reasoning(value, types):
        return _reasoning_stub(value)
    stack, seen = [value], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, item in list(items):
            if _is_reasoning(item, types):
                node[key] = _reasoning_stub(item)
            elif isinstance(item, (dict, list)):
                stack.append(item)
    return value


def _is_reasoning(value: Any, types: frozenset[str]) -> bool:
    return isinstance(value, dict) and isinstance(value.get("type"), str) and value["type"] in types


def _reasoning_stub(value: dict[str, Any]) -> dict[str, Any]:
    return {"type": value["type"], "omitted": "reasoning"}


def export_policy(event: Event) -> ExportPolicy:
    """Default export treatment: a conservative default."""
    if isinstance(event, (Unrecognized, Reasoning)):
        return ExportPolicy.NEVER
    if isinstance(event, Notice):
        return ExportPolicy.NOT_BY_DEFAULT
    if isinstance(event, Message) and event.origin in (MessageOrigin.INJECTED, MessageOrigin.COMMAND_OUTPUT):
        return ExportPolicy.NOT_BY_DEFAULT
    return ExportPolicy.AFTER_SCRUBBING


_CATEGORY_RANK = {RecordCategory.SEMANTIC: 3, RecordCategory.METADATA: 2, RecordCategory.UNKNOWN: 1}


def event_category(event: Event) -> RecordCategory:
    if isinstance(event, Unrecognized):
        return RecordCategory.UNKNOWN
    if isinstance(event, Notice):
        return RecordCategory.METADATA
    return RecordCategory.SEMANTIC


def combine_categories(current: RecordCategory | None, new: RecordCategory) -> RecordCategory:
    """Category of a record from several events: an interpreted event wins over metadata over unknown."""
    if current is None or _CATEGORY_RANK.get(new, 0) > _CATEGORY_RANK.get(current, 0):
        return new
    return current


SEMANTIC_EVENT_TYPES = (Message, ToolCall, ToolResult, ErrorEvent)


def semantic_view(events: list[Event]) -> list[tuple[Any, ...]]:
    """Format-independent summary of a trace, for comparing equivalent traces from different agents.

    Keeps messages (one entry per logical message; user messages that are not plain prompts carry their
    origin), tool calls, results and errors; drops session markers, notices, reasoning (some agents never
    export it) and unrecognised records. Call ids are replaced by their order of appearance so that native id
    formats do not matter.
    """
    view: list[tuple[Any, ...]] = []
    call_numbers: dict[str, int] = {}
    last_message_key: tuple[Any, ...] | None = None
    for event in events:
        if not isinstance(event, SEMANTIC_EVENT_TYPES):
            continue
        if isinstance(event, Message):
            grouping = event.message_id if isinstance(event.message_id, str) else ("line", event.source.line)
            key = (event.role, grouping)
            if key == last_message_key:
                continue
            last_message_key = key
            entry: tuple[Any, ...] = ("message", event.role.value, event.content_kind.value)
            plain = event.origin in (MessageOrigin.PROMPT, MessageOrigin.MODEL)
            view.append(entry if plain else entry + (event.origin.value,))
            continue
        last_message_key = None
        if isinstance(event, ToolCall):
            number = call_numbers.setdefault(event.call_id, len(call_numbers) + 1) if isinstance(
                event.call_id, str
            ) else 0
            view.append(("tool_call", number, event.category.value))
        elif isinstance(event, ToolResult):
            number = call_numbers.get(event.call_id, 0) if isinstance(event.call_id, str) else 0
            view.append(("tool_result", number, event.category.value, event.status.value))
        elif isinstance(event, ErrorEvent):
            view.append(("error", event.scope.value))
    return view
