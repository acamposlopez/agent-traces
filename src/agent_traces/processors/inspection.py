"""Inspection processor: a structural report of a trace that never contains original content.

Privacy by construction: the report only holds numbers, enum values, fixed codes and texts written in this
code base, and native names that the adapter allowlists (record types, tool names). Model identifiers, agent
versions and the file hash are only included on explicit request, and even then unsafe-looking values stay
hidden.

It separates structural recognition from semantic interpretation: every record is classified as semantic
(interpreted conversational events), metadata (identified, content not interpreted; verified structure or
name only), unknown, or skipped (documented duplicates).

Bounded memory: identifiers are remembered as keyed 64-bit digests (never the original strings) and every
tracking structure has a fixed maximum size, so a hostile file cannot exhaust memory through them.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from agent_traces.engine import RunContext
from agent_traces.model import (
    SCHEMA_VERSION,
    ErrorEvent,
    Event,
    ExportPolicy,
    FieldState,
    Message,
    MessageOrigin,
    MetadataStructure,
    Notice,
    Reasoning,
    RecordCategory,
    ResultStatus,
    Role,
    SessionMarker,
    SessionPhase,
    StatusBasis,
    TerminalState,
    ToolCall,
    ToolCategory,
    ToolResult,
    Unparseable,
    Unrecognized,
    UnrecognizedReason,
    ValueTracker,
    export_policy,
    state_of,
)
from agent_traces.processors.classification import RecordClassifier
from agent_traces.reader import RawRecord

REPORT_VERSION = "2"
MAX_TRACKED_IDS = 500_000  # per relation check (record ids, message ids, call ids); ~70 bytes each
MAX_TRACKED_NAMES = 10_000  # distinct agent ids, subagent refs, hidden tool names, timestamp scopes

_VERSION_RE = re.compile(r"^\d{1,4}(\.\d{1,6}){0,3}(-(alpha|beta|rc|dev|pre|preview)(\.?\d{1,6})?)?$")
_MODEL_CHARSET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:@+-]{0,63}$")
_MODEL_FAMILY_RE = re.compile(
    r"^(?:anthropic\.|openai\.|google\.)?"
    r"(claude|gpt|o\d|gemini|grok|composer|cursor|sonnet|opus|haiku|auto|deepseek|kimi|qwen|llama|"
    r"mistral|codestral|codex)(?![A-Za-z])",
    re.IGNORECASE,
)
_MODEL_TOKEN_SPLIT_RE = re.compile(r"[ ._:@+-]")
_MAX_MODEL_TOKEN = 16  # longer runs look like keys or random ids, never like model names
_NOTICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/-]{0,64}$")
_ORIGIN_PRIORITY = {
    MessageOrigin.PROMPT: 6,
    MessageOrigin.COMMAND: 5,
    MessageOrigin.INTERRUPTION: 4,
    MessageOrigin.COMMAND_OUTPUT: 3,
    MessageOrigin.INJECTED: 2,
    MessageOrigin.UNKNOWN: 1,
}


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class IntegrityStatus(StrEnum):
    OK = "ok"
    WARNINGS = "warnings"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"  # records exist but none could be interpreted
    EMPTY = "empty"


ISSUE_SEVERITY: dict[str, Severity] = {
    UnrecognizedReason.MALFORMED_JSON.value: Severity.ERROR,
    UnrecognizedReason.INVALID_ENCODING.value: Severity.ERROR,
    UnrecognizedReason.OVERSIZED.value: Severity.ERROR,
    UnrecognizedReason.NOT_AN_OBJECT.value: Severity.ERROR,
    UnrecognizedReason.INCOMPLETE.value: Severity.WARNING,
    UnrecognizedReason.UNKNOWN_TYPE.value: Severity.WARNING,
    UnrecognizedReason.UNKNOWN_SUBTYPE.value: Severity.WARNING,
    UnrecognizedReason.UNKNOWN_BLOCK.value: Severity.WARNING,
    "input_shrank_during_read": Severity.ERROR,
    "incomplete_final_record": Severity.WARNING,
    "missing_terminal_event": Severity.ERROR,
    "unclear_terminal_event": Severity.WARNING,
    "calls_without_result": Severity.WARNING,
    "results_without_call": Severity.WARNING,
    "duplicate_call_ids": Severity.WARNING,
    "duplicate_results": Severity.WARNING,
    "unknown_result_outcomes": Severity.WARNING,
    "missing_call_ids": Severity.WARNING,
    "unresolved_parent_refs": Severity.WARNING,
    "unparseable_ids": Severity.WARNING,
    "conversation_out_of_order": Severity.WARNING,
    "unparseable_timestamps": Severity.WARNING,
    "unparseable_session_ids": Severity.WARNING,
    "multiple_sessions": Severity.WARNING,
    "relation_tracking_truncated": Severity.WARNING,
}

# Canonical fields whose presence is measured (present / absent / unparseable).
TRACKED_FIELDS = (
    "message.content",
    "tool_call.call_id",
    "tool_call.arguments",
    "tool_result.call_id",
    "tool_result.output",
)

GENERAL_LIMITATIONS = (
    "Content is never shown: prompts, responses, tool arguments and outputs, paths, identifiers and "
    "unknown field or tool names are reduced to counts.",
    "Model identifiers, agent versions and the file hash are omitted unless explicitly requested.",
    "Metadata records are identified, not interpreted: 'verified' means their documented or observed "
    "structural fields are present, not that their content is understood.",
)


@dataclass(frozen=True, slots=True)
class InspectionReport:
    report_version: str
    schema_version: str
    source: dict[str, Any]
    input: dict[str, Any]
    session: dict[str, Any]
    coverage: dict[str, Any]
    counts: dict[str, Any]
    tools: dict[str, Any]
    relations: dict[str, Any]
    records: dict[str, Any]
    fields: dict[str, Any]
    integrity: dict[str, Any]
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True)


class _DigestSet:
    """Remembers strings as keyed 64-bit BLAKE2b digests (never the strings themselves), up to ``limit``."""

    __slots__ = ("_items", "_key", "_limit", "saturated")

    def __init__(self, key: bytes, limit: int) -> None:
        self._key = key
        self._items: set[bytes] = set()
        self._limit = limit
        self.saturated = False

    def __len__(self) -> int:
        return len(self._items)

    def digest(self, value: str) -> bytes:
        return hashlib.blake2b(value.encode("utf-8", "surrogatepass"), digest_size=8, key=self._key).digest()

    def seen(self, value: str) -> bool | None:
        """True if remembered, False if certainly not, None if unknown because the set is full."""
        if self.digest(value) in self._items:
            return True
        return None if self.saturated else False

    def add(self, value: str) -> bool | None:
        """Remember ``value``: True if it was new, False if already known, None if the set is full."""
        item = self.digest(value)
        if item in self._items:
            return False
        if len(self._items) >= self._limit:
            self.saturated = True
            return None
        self._items.add(item)
        return True


class InspectionProcessor:
    name = "inspection"

    def __init__(
        self, *, include_file_hash: bool = False, include_models: bool = False, include_agent_version: bool = False
    ) -> None:
        self.include_file_hash = include_file_hash
        self.include_models = include_models
        self.include_agent_version = include_agent_version
        self._reset(frozenset())

    def _reset(self, known_tools: frozenset[str]) -> None:
        key = secrets.token_bytes(16)  # per run: digests cannot be precomputed or compared across runs
        self.known_tools = known_tools
        self.events = 0
        self.interpreted = 0
        self.record_categories: Counter[str] = Counter()
        self.classifier = RecordClassifier(self._record_classified)
        self.unknown_blocks = 0
        self.messages: Counter[str] = Counter()
        self.message_ids = _DigestSet(key, MAX_TRACKED_IDS)
        self.last_line_message: tuple[str, int] | None = None
        self.user_origins: Counter[str] = Counter()
        self.current_user: list[Any] | None = None  # [line, origin] of the user record being read
        self.content_kinds: Counter[str] = Counter()
        self.reasoning = 0
        self.reasoning_redacted = 0
        self.tool_calls = 0
        self.tool_results = 0
        self.categories: Counter[str] = Counter()
        self.known_tool_counts: Counter[str] = Counter()  # allowlisted names only
        self.hidden_tool_names = _DigestSet(key, MAX_TRACKED_NAMES)
        self.calls_without_name = 0
        self.outcomes: Counter[str] = Counter()
        self.error_scopes: Counter[str] = Counter()
        self.run_failures = 0
        self.calls = _DigestSet(key, MAX_TRACKED_IDS)
        self.answered = _DigestSet(key, MAX_TRACKED_IDS)
        self.pairing: Counter[str] = Counter()
        self.notices: Counter[str] = Counter()
        self.metadata_classes: Counter[str] = Counter()
        self.metadata_verified = 0
        self.unrecognized: Counter[str] = Counter()
        self.session_markers: Counter[str] = Counter()
        self.export: Counter[str] = Counter()
        self.record_ids = _DigestSet(key, MAX_TRACKED_IDS)
        # Events of one record arrive consecutively: per-record checks only need the last line seen.
        self.last_id_line = 0
        self.parent_refs: Counter[str] = Counter()
        self.unparseable_ids = 0
        self.last_timestamp_line = 0
        self.timestamps: Counter[str] = Counter()
        self.out_of_order: Counter[str] = Counter()
        self.last_timestamp: dict[tuple[str, bytes], datetime] = {}
        self.agent_ids = _DigestSet(key, MAX_TRACKED_NAMES)
        self.subagent_refs = _DigestSet(key, MAX_TRACKED_NAMES)
        self.sidechain_events = 0
        self.fields: dict[str, Counter[str]] = {}

    # Processor and RecordObserver protocols -----------------------------------------------------------------

    def start(self, context: RunContext) -> None:
        self._reset(frozenset(context.adapter.known_tools) if context.adapter else frozenset())

    def observe(self, record: RawRecord) -> None:
        self.classifier.observe(record.line)

    def handle(self, event: Event) -> None:
        self.events += 1
        self.classifier.event(event)
        self.export[export_policy(event).value] += 1
        if event.source.line != self.last_id_line:
            self.last_id_line = event.source.line
            self._record_ids(event)
        if isinstance(event, Unrecognized):
            self.unrecognized[event.reason.value] += 1
            self.unknown_blocks += event.source.block is not None
            return
        self.interpreted += 1
        self._track_timestamp(event)
        if event.is_sidechain:
            self.sidechain_events += 1

        if isinstance(event, Message):
            self._message(event)
        elif isinstance(event, Reasoning):
            self.reasoning += 1
            self.reasoning_redacted += event.redacted
        elif isinstance(event, ToolCall):
            self._tool_call(event)
        elif isinstance(event, ToolResult):
            self._tool_result(event)
        elif isinstance(event, ErrorEvent):
            self.error_scopes[event.scope.value] += 1
        elif isinstance(event, SessionMarker):
            self.session_markers[event.phase.value] += 1
            if event.phase is SessionPhase.END and event.status is ResultStatus.ERROR:
                self.run_failures += 1
        elif isinstance(event, Notice):
            self.notices[event.notice_type if _NOTICE_RE.match(event.notice_type) else "other"] += 1
            self.metadata_classes[event.metadata_class if _NOTICE_RE.match(event.metadata_class) else "other"] += 1
            self.metadata_verified += event.structure is MetadataStructure.VERIFIED

    def finish(self, context: RunContext) -> InspectionReport:
        self.classifier.finish()
        self._close_user_message()
        return self._build(context)

    # Tracking helpers -----------------------------------------------------------------------------------

    def _record_classified(self, line: int, payload: Any, category: RecordCategory) -> None:
        self.record_categories[category.value] += 1

    def _field(self, name: str, value: Any) -> None:
        self.fields.setdefault(name, Counter())[state_of(value).value] += 1

    def _record_ids(self, event: Event) -> None:
        parent, own, agent = event.parent_id, event.event_id, event.agent_id
        self.unparseable_ids += sum(
            isinstance(value, Unparseable) for value in (own, parent, event.message_id, agent)
        )
        if isinstance(parent, str):
            known = self.record_ids.seen(parent)
            self.parent_refs["resolved" if known else "unverified" if known is None else "unresolved"] += 1
        if isinstance(own, str):
            self.record_ids.add(own)
        if isinstance(agent, str):
            self.agent_ids.add(agent)

    def _track_timestamp(self, event: Event) -> None:
        line = event.source.line
        if line == self.last_timestamp_line:
            return
        self.last_timestamp_line = line
        value = event.timestamp
        self.timestamps[state_of(value).value] += 1
        if isinstance(value, datetime):
            kind = "metadata" if isinstance(event, Notice) else "conversation"
            agent = event.agent_id if isinstance(event.agent_id, str) else ""
            scope = (kind, self.agent_ids.digest(agent) + (b"s" if event.is_sidechain else b"m"))
            previous = self.last_timestamp.get(scope)
            if previous is not None and value < previous:
                self.out_of_order[kind] += 1
            if previous is not None or len(self.last_timestamp) < MAX_TRACKED_NAMES:
                self.last_timestamp[scope] = value

    def _message(self, event: Message) -> None:
        role = event.role.value
        if isinstance(event.message_id, str):
            if self.message_ids.add(f"{role}\x00{event.message_id}") is not False:
                self.messages[role] += 1  # new id, or set full (counted as new: approximate)
        else:
            key = (role, event.source.line)
            if key != self.last_line_message:
                self.last_line_message = key
                self.messages[role] += 1
        if event.role is Role.USER:
            origin = event.origin or MessageOrigin.UNKNOWN
            if self.current_user is not None and self.current_user[0] == event.source.line:
                if _ORIGIN_PRIORITY[origin] > _ORIGIN_PRIORITY[self.current_user[1]]:
                    self.current_user[1] = origin  # the most human origin among the blocks of one record
            else:
                self._close_user_message()
                self.current_user = [event.source.line, origin]
        self.content_kinds[event.content_kind.value] += 1
        self._field(TRACKED_FIELDS[0], event.content)

    def _close_user_message(self) -> None:
        if self.current_user is not None:
            self.user_origins[self.current_user[1].value] += 1
            self.current_user = None

    def _tool_call(self, event: ToolCall) -> None:
        self.tool_calls += 1
        self.categories[event.category.value] += 1
        name = event.tool_name
        if name is None:
            self.calls_without_name += 1
        elif name in self.known_tools:
            self.known_tool_counts[name] += 1
        else:
            self.hidden_tool_names.add(name)
        self._field(TRACKED_FIELDS[1], event.call_id)
        self._field(TRACKED_FIELDS[2], event.arguments)
        if event.category is ToolCategory.SUBAGENT:
            self.pairing["subagent_calls"] += 1
        if isinstance(event.call_id, str):
            if self.calls.add(event.call_id) is False:
                self.pairing["duplicate_call_ids"] += 1
        else:
            self.pairing["missing_call_ids"] += 1

    def _tool_result(self, event: ToolResult) -> None:
        self.tool_results += 1
        self.outcomes[_outcome(event)] += 1
        self._field(TRACKED_FIELDS[3], event.call_id)
        self._field(TRACKED_FIELDS[4], event.output)
        if isinstance(event.subagent_ref, str):
            self.subagent_refs.add(event.subagent_ref)
        elif isinstance(event.subagent_ref, Unparseable):
            self.unparseable_ids += 1
        call_id = event.call_id
        if not isinstance(call_id, str):
            self.pairing["missing_call_ids"] += 1
            return
        known = self.calls.seen(call_id)
        if known is None:
            self.pairing["unverified_results"] += 1
        elif not known:
            self.pairing["results_without_call"] += 1
        elif self.answered.add(call_id) is False:
            self.pairing["duplicate_results"] += 1
        else:
            self.pairing["paired"] += 1

    # Report -------------------------------------------------------------------------------------------

    def _build(self, context: RunContext) -> InspectionReport:
        adapter, meta, stats, detection = context.adapter, context.metadata, context.stats, context.detection
        calls_without_result = len(self.calls) - len(self.answered)
        hidden_tool_names = len(self.hidden_tool_names)
        versions_tracker = meta.agent_versions if meta else ValueTracker()
        models_tracker = meta.models if meta else ValueTracker()
        versions, hidden_versions = _visible(versions_tracker, _is_safe_version, self.include_agent_version)
        models, hidden_models = _visible(models_tracker, _is_safe_model, self.include_models)
        terminal = meta.terminal_state if meta else TerminalState.NOT_APPLICABLE
        session_ids = meta.session_ids if meta else ValueTracker()
        tool_errors = self.outcomes["error_confirmed"]
        messages = dict(sorted(self.messages.items()))
        truncated = any(
            tracker.saturated
            for tracker in (self.message_ids, self.calls, self.answered, self.record_ids, self.hidden_tool_names,
                            self.agent_ids, self.subagent_refs)
        )

        issues: Counter[str] = Counter()
        for reason, count in self.unrecognized.items():
            issues[reason] += count
        issues["input_shrank_during_read"] += stats.shrank_during_read
        issues["incomplete_final_record"] += stats.incomplete_tail_bytes > 0
        if adapter and adapter.has_terminal_event and stats.records:
            if terminal is TerminalState.MISSING:
                issues["missing_terminal_event"] += 1
            elif terminal is TerminalState.UNCLEAR:
                issues["unclear_terminal_event"] += 1
        issues["calls_without_result"] += calls_without_result
        for key in ("results_without_call", "duplicate_call_ids", "duplicate_results", "missing_call_ids"):
            issues[key] += self.pairing[key]
        issues["unknown_result_outcomes"] += self.outcomes["unknown"]
        issues["unresolved_parent_refs"] += self.parent_refs["unresolved"]
        issues["unparseable_ids"] += self.unparseable_ids
        issues["conversation_out_of_order"] += self.out_of_order["conversation"]
        issues["unparseable_timestamps"] += self.timestamps[FieldState.UNPARSEABLE.value]
        issues["unparseable_session_ids"] += session_ids.states[FieldState.UNPARSEABLE]
        issues["multiple_sessions"] += len(session_ids.values) > 1
        issues["relation_tracking_truncated"] += truncated
        issue_list = sorted(
            (
                {"code": code, "severity": ISSUE_SEVERITY[code].value, "count": count}
                for code, count in issues.items()
                if count
            ),
            key=lambda item: (item["severity"] != Severity.ERROR.value, item["code"]),
        )
        if stats.records == 0:
            status = IntegrityStatus.EMPTY
        elif self.interpreted == 0:
            status = IntegrityStatus.UNUSABLE
        elif any(item["severity"] == Severity.ERROR.value for item in issue_list):
            status = IntegrityStatus.DEGRADED
        elif issue_list:
            status = IntegrityStatus.WARNINGS
        else:
            status = IntegrityStatus.OK

        coverage = self._coverage(stats.records)
        limitations = self._limitations(adapter, stats, hidden_tool_names, hidden_models, hidden_versions, truncated)

        input_section: dict[str, Any] = {
            "bytes": stats.bytes_read,
            "records": stats.records,
            "blank_lines": stats.blank_lines,
            "ends_with_newline": stats.ends_with_newline if stats.bytes_read else None,
            "snapshot_bytes": stats.snapshot_bytes,
            "grew_during_read": stats.grew_during_read,
            "incomplete_tail_bytes": stats.incomplete_tail_bytes,
        }
        if self.include_file_hash:
            input_section["sha256"] = stats.sha256 or None

        return InspectionReport(
            report_version=REPORT_VERSION,
            schema_version=SCHEMA_VERSION,
            source={
                "format": adapter.name if adapter else None,
                "format_name": adapter.display_name if adapter else None,
                "adapter_version": adapter.version if adapter else None,
                "detection": detection.status.value,
                "records_examined": detection.records_examined,
            },
            input=input_section,
            session={
                "session_id": session_ids.state.value,
                "distinct_sessions": len(session_ids.values),
                "distinct_sessions_capped": session_ids.capped,
                "agent": adapter.name if adapter else None,
                "agent_versions_state": versions_tracker.state.value,
                "agent_versions_visible": self.include_agent_version,
                "agent_versions": versions,
                "agent_versions_not_shown": hidden_versions,
                "models_state": models_tracker.state.value,
                "models_visible": self.include_models,
                "models": models,
                "models_not_shown": hidden_models,
                "terminal_state": terminal.value,
            },
            coverage=coverage,
            counts={
                "events": self.events,
                "messages": {"total": sum(messages.values()), **messages},
                "user_messages_by_origin": {
                    origin.value: self.user_origins[origin.value]
                    for origin in MessageOrigin
                    if origin is not MessageOrigin.MODEL
                },
                "content_kinds": dict(sorted(self.content_kinds.items())),
                "reasoning_blocks": self.reasoning,
                "reasoning_redacted": self.reasoning_redacted,
                "tool_calls": self.tool_calls,
                "tool_results": self.tool_results,
                "result_outcomes": {key: self.outcomes[key] for key in _OUTCOMES},
                "errors": {
                    "total": tool_errors + sum(self.error_scopes.values()) + self.run_failures,
                    "tool": tool_errors,
                    **{scope: n for scope, n in sorted(self.error_scopes.items())},
                    "run_failed": self.run_failures,
                },
            },
            tools={
                "categories": dict(sorted(self.categories.items())),
                "known_tools": dict(sorted(self.known_tool_counts.items())),
                "names_not_shown": hidden_tool_names,
                "calls_without_name": self.calls_without_name,
            },
            relations={
                "paired_calls": self.pairing["paired"],
                "calls_without_result": calls_without_result,
                "results_without_call": self.pairing["results_without_call"],
                "duplicate_call_ids": self.pairing["duplicate_call_ids"],
                "duplicate_results": self.pairing["duplicate_results"],
                "missing_call_ids": self.pairing["missing_call_ids"],
                "unverified_results": self.pairing["unverified_results"],
                "parent_refs_resolved": self.parent_refs["resolved"],
                "parent_refs_unresolved": self.parent_refs["unresolved"],
                "parent_refs_unverified": self.parent_refs["unverified"],
                "subagent_calls": self.pairing["subagent_calls"],
                "subagent_refs": len(self.subagent_refs),
                "agent_ids": len(self.agent_ids),
                "sidechain_events": self.sidechain_events,
                "tracking_truncated": truncated,
            },
            records={
                "interpreted_events": self.interpreted,
                "unrecognized": dict(sorted(self.unrecognized.items())),
                "notices": dict(sorted(self.notices.items())),
                "metadata_classes": dict(sorted(self.metadata_classes.items())),
                "session_markers": dict(sorted(self.session_markers.items())),
                "normalization": dict(sorted(meta.normalization.items())) if meta else {},
                "export_policy": {policy.value: self.export[policy.value] for policy in ExportPolicy},
            },
            fields={
                "timestamp": _states(self.timestamps),
                "out_of_order": {"conversation": self.out_of_order["conversation"],
                                 "metadata": self.out_of_order["metadata"]},
                **{name: _states(counter) for name, counter in sorted(self.fields.items())},
            },
            integrity={"status": status.value, "issues": issue_list},
            limitations=limitations,
        )

    def _coverage(self, records: int) -> dict[str, Any]:
        semantic = self.record_categories[RecordCategory.SEMANTIC.value]
        metadata = self.record_categories[RecordCategory.METADATA.value]
        verified = min(self.metadata_verified, metadata)

        def share(count: int) -> dict[str, Any]:
            return {"records": count, "percent": _percent(count, records)}

        return {
            "records": records,
            "semantic": share(semantic),
            "metadata_verified": share(verified),
            "metadata_name_only": share(metadata - verified),
            "unknown": share(self.record_categories[RecordCategory.UNKNOWN.value]),
            "skipped": share(self.record_categories[RecordCategory.SKIPPED.value]),
            "semantically_interpreted_percent": _percent(semantic, records),
            "structurally_recognized_percent": _percent(semantic + metadata, records),
            "unknown_blocks_in_interpreted_records": self.unknown_blocks,
        }

    def _limitations(self, adapter, stats, hidden_tool_names, hidden_models, hidden_versions, truncated) -> list[str]:
        limitations = list(GENERAL_LIMITATIONS)
        if adapter:
            limitations.extend(adapter.limitations)
        if self.reasoning:
            limitations.append(
                f"{self.reasoning} reasoning block(s) found: their content is discarded while parsing and never "
                "reported."
            )
        unrecognized_total = sum(self.unrecognized.values())
        if unrecognized_total:
            limitations.append(
                f"{unrecognized_total} record(s) or content block(s) could not be interpreted; they are treated as "
                "potentially sensitive and only counted."
            )
        if hidden_tool_names:
            limitations.append(
                f"{hidden_tool_names} distinct tool name(s) are not in the adapter allowlist and are not shown."
            )
        unsafe_models = hidden_models if self.include_models else 0
        unsafe_versions = hidden_versions if self.include_agent_version else 0
        if unsafe_models or unsafe_versions:
            limitations.append(
                f"{unsafe_models} model and {unsafe_versions} agent version value(s) are not shown because they "
                "do not match a known safe pattern."
            )
        if stats.grew_during_read:
            limitations.append(
                f"{stats.snapshot_bytes} bytes inspected: the file grew while it was read (active session) and "
                "later bytes were not read."
            )
        if stats.incomplete_tail_bytes:
            limitations.append(
                f"{stats.incomplete_tail_bytes} bytes at the end form an unterminated record that was not counted: "
                "the file may still be being written."
            )
        if truncated:
            limitations.append(
                f"{MAX_TRACKED_IDS} tracked identifiers reached: relation checks beyond that point are partial."
            )
        return limitations


_OUTCOMES = ("success_confirmed", "success_protocol_default", "error_confirmed", "unknown")


def _outcome(event: ToolResult) -> str:
    """Never turns an unknown or defaulted state into a confirmed one."""
    if event.status is ResultStatus.ERROR and event.status_basis is StatusBasis.EXPLICIT:
        return "error_confirmed"
    if event.status is ResultStatus.SUCCESS and event.status_basis is StatusBasis.EXPLICIT:
        return "success_confirmed"
    if event.status is ResultStatus.SUCCESS and event.status_basis is StatusBasis.PROTOCOL_DEFAULT:
        return "success_protocol_default"
    return "unknown"


def _percent(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def _states(counter: Counter[str]) -> dict[str, int]:
    return {state.value: counter[state.value] for state in FieldState}


def _is_safe_version(value: str) -> bool:
    return bool(_VERSION_RE.match(value))


def _is_safe_model(value: str) -> bool:
    return bool(
        _MODEL_CHARSET_RE.match(value)
        and _MODEL_FAMILY_RE.match(value)
        and all(len(token) <= _MAX_MODEL_TOKEN for token in _MODEL_TOKEN_SPLIT_RE.split(value))
    )


def _visible(tracker: ValueTracker, is_safe, requested: bool) -> tuple[list[str], int]:
    """Values shown (only on request and only when safe) and how many distinct values stay hidden."""
    if not requested:
        return [], len(tracker.values)
    shown = [value for value in tracker.values if is_safe(value)]
    return shown, len(tracker.values) - len(shown)


# Text rendering (ASCII only, so that no terminal encoding can make it fail) ---------------------------------


def render_text(report: InspectionReport) -> str:
    r = report
    src, inp, ses, cov, cnt, tools, rel, rec, fld = (
        r.source, r.input, r.session, r.coverage, r.counts, r.tools, r.relations, r.records, r.fields
    )
    fmt = src["format"] or "undetermined"
    detection = {"explicit": "set with --format", "detected": "auto-detected", "empty": "nothing to detect"}.get(
        src["detection"], src["detection"]
    )
    lines = ["agent-traces inspect: structural report (trace content is never shown)", ""]

    def section(title: str, rows: list[tuple[str, str]]) -> None:
        lines.append(title)
        width = max(len(label) for label, _ in rows)
        lines.extend(f"  {label.ljust(width)}  {value}" for label, value in rows)
        lines.append("")

    examined = f", {src['records_examined']} records examined" if src["detection"] == "detected" else ""
    input_text = f"{inp['bytes']} bytes, {inp['records']} records, {inp['blank_lines']} blank lines"
    if inp["grew_during_read"]:
        input_text += " (file grew while read: snapshot inspected)"
    if inp["incomplete_tail_bytes"]:
        input_text += f", {inp['incomplete_tail_bytes']} bytes of unterminated tail"
    source_rows = [
        ("format", f"{fmt} ({detection}{examined})"),
        ("adapter", f"{src['format_name'] or '-'}, adapter {src['adapter_version'] or '-'}, "
                    f"canonical schema {r.schema_version}"),
        ("input", input_text),
    ]
    if "sha256" in inp:
        source_rows.append(("sha256", inp["sha256"] or "-"))
    section("Source", source_rows)

    distinct = f"{ses['distinct_sessions']}{'+' if ses['distinct_sessions_capped'] else ''}"
    section(
        "Session",
        [
            ("session id", f"{ses['session_id']} ({distinct} distinct, values not shown)"),
            ("agent version", _hidden_text(ses["agent_versions_visible"], ses["agent_versions"],
                                           ses["agent_versions_not_shown"], ses["agent_versions_state"],
                                           "--include-agent-version")),
            ("model", _hidden_text(ses["models_visible"], ses["models"], ses["models_not_shown"],
                                   ses["models_state"], "--include-models")),
            ("terminal event", ses["terminal_state"].replace("_", " ")),
        ],
    )

    def pct(key: str) -> str:
        return f"{cov[key]['records']} ({cov[key]['percent']}%)"

    section(
        "Coverage (per record)",
        [
            ("semantic", f"{pct('semantic')} interpreted conversational records"),
            ("metadata", f"{pct('metadata_verified')} verified structure, {pct('metadata_name_only')} name only "
                         "(content not interpreted)"),
            ("unknown", f"{pct('unknown')} not interpretable"),
            ("skipped", f"{pct('skipped')} documented duplicates"),
            ("recognized", f"{cov['structurally_recognized_percent']}% structurally, "
                           f"{cov['semantically_interpreted_percent']}% semantically"),
        ],
    )
    msgs = cnt["messages"]
    by_role = ", ".join(f"{role} {n}" for role, n in msgs.items() if role != "total") or "none"
    errors = cnt["errors"]
    error_parts = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in errors.items() if k != "total")
    section(
        "Counts",
        [
            ("records", str(inp["records"])),
            ("events", str(cnt["events"])),
            ("messages", f"{msgs['total']} ({by_role})"),
            ("user messages", _counter(cnt["user_messages_by_origin"]) + " (by origin)"),
            ("content kinds", _counter(cnt["content_kinds"])),
            ("reasoning", f"{cnt['reasoning_blocks']} blocks (content never stored)"),
            ("tool calls", str(cnt["tool_calls"])),
            ("tool results", f"{cnt['tool_results']} ({_counter(cnt['result_outcomes'])})"),
            ("errors", f"{errors['total']} ({error_parts})"),
        ],
    )
    section(
        "Tools",
        [
            ("categories", _counter(tools["categories"])),
            ("known tools", _counter(tools["known_tools"])),
            ("names not shown", f"{tools['names_not_shown']} distinct (not in the allowlist)"),
        ],
    )
    section(
        "Relations",
        [
            ("call <-> result", f"{rel['paired_calls']} paired, {rel['calls_without_result']} calls without "
                                f"result, {rel['results_without_call']} results without call"),
            ("parent links", f"{rel['parent_refs_resolved']} resolved, {rel['parent_refs_unresolved']} unresolved"),
            ("subagents", f"{rel['subagent_calls']} subagent calls, {rel['subagent_refs']} subagent refs, "
                          f"{rel['agent_ids']} agent ids, {rel['sidechain_events']} sidechain events"),
        ],
    )
    ts = fld["timestamp"]
    order = fld["out_of_order"]
    section(
        "Records",
        [
            ("interpreted events", str(rec["interpreted_events"])),
            ("unrecognized", _counter(rec["unrecognized"])),
            ("metadata types", _counter(rec["notices"])),
            ("metadata classes", _counter(rec["metadata_classes"])),
            ("session markers", _counter(rec["session_markers"])),
            ("normalization", _counter(rec["normalization"])),
            ("export policy", _counter(rec["export_policy"]) + " (events; nothing is exported yet)"),
            ("timestamps", f"{ts['present']} present, {ts['absent']} absent, {ts['unparseable']} unparseable"),
            ("out of order", f"{order['conversation']} conversation, {order['metadata']} metadata"),
        ],
    )
    lines.append(f"Integrity: {r.integrity['status']}")
    for issue in r.integrity["issues"]:
        lines.append(f"  [{issue['severity']}] {issue['code']}: {issue['count']}")
    lines.append("")
    lines.append("Limitations")
    lines.extend(f"  - {text}" for text in r.limitations)
    return "\n".join(lines) + "\n"


def _counter(values: dict[str, int]) -> str:
    items = [f"{key} {count}" for key, count in values.items() if count]
    return ", ".join(items) if items else "none"


def _hidden_text(visible: bool, values: list[str], hidden: int, state: str, flag: str) -> str:
    if visible:
        parts = [", ".join(values)] if values else []
        if hidden:
            parts.append(f"{hidden} not shown")
        return "; ".join(parts) if parts else state
    if hidden:
        return f"present ({hidden} distinct, not shown; use {flag})"
    return state
