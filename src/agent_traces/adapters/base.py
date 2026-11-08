"""Contract every source adapter implements. The engine and processors only depend on this module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, ClassVar

from agent_traces.model import (
    Event,
    SourceRef,
    ToolCategory,
    TraceMetadata,
    Unparseable,
    Unrecognized,
    UnrecognizedReason,
)
from agent_traces.reader import RawRecord

MAX_ID_LENGTH = 256  # real ids (uuids, toolu_..., call ids) are far shorter; longer values are not ids
MAX_CONTEXT_VALUES = 32  # distinct cwd / branch values kept per trace
MAX_CONTEXT_LENGTH = 4096


class Signal(IntEnum):
    """How strongly one record matches a format. Signals are positive only: a record that does not match is
    NONE, and contradictions are derived by detection when records match several formats strongly."""

    NONE = 0
    WEAK = 1  # consistent with the format, but other formats look alike
    STRONG = 2  # matches a full structural signature (several keys and types) only this format produces


class Source(StrEnum):
    """Provenance of a vocabulary name."""

    DOCUMENTED = "documented"  # official documentation of the agent or API
    OBSERVED = "observed"  # public third-party analyses of real files (experimental observation)


@dataclass(frozen=True)
class Vocabulary:
    """Explicit, reviewed structural names of a format. Only these may appear in schema discovery; any other
    name is replaced by an opaque local code, whatever it looks like."""

    record_types: Mapping[str, Source] = field(default_factory=dict)  # values of the top-level "type"
    discriminators: Mapping[str, Source] = field(default_factory=dict)  # subtype-like values (see paths)
    discriminator_paths: tuple[tuple[str, ...], ...] = (("subtype",),)
    fields: Mapping[str, Source] = field(default_factory=dict)  # field names at any described level
    block_types: Mapping[str, Source] = field(default_factory=dict)  # "type" of content blocks
    containers: frozenset[str] = frozenset()  # fields whose object value is described one level deep

    def source_of(self, name: str) -> Source | None:
        for table in (self.record_types, self.discriminators, self.fields, self.block_types):
            if name in table:
                return table[name]
        return None


class FormatRecognizer(ABC):
    """Something detection can recognise. Supported formats are ``SourceAdapter`` subclasses; known but
    unsupported formats only recognise their signatures, so that they are reported instead of mistaken."""

    name: ClassVar[str]  # kebab-case
    display_name: ClassVar[str]
    supported: ClassVar[bool] = False
    vocabulary: ClassVar[Vocabulary] = Vocabulary()

    @abstractmethod
    def sniff(self, record: Mapping[str, Any]) -> Signal:
        """Classify one parsed record for format detection. Must not keep any reference to it."""


class SourceAdapter(FormatRecognizer):
    name: ClassVar[str]  # value of --format, kebab-case
    display_name: ClassVar[str]
    supported: ClassVar[bool] = True
    version: ClassVar[str]  # adapter version, bumped when its interpretation changes
    has_terminal_event: ClassVar[bool] = False
    # Native names that are safe to show in reports (record types, notice types, tool names).
    known_record_types: ClassVar[frozenset[str]] = frozenset()
    known_tools: ClassVar[Mapping[str, ToolCategory]] = {}
    # Values of "type" that mark reasoning content in this format. The engine replaces any such object
    # found inside raw payloads with a content-free stub, so reasoning never reaches processors.
    reasoning_block_types: ClassVar[frozenset[str]] = frozenset({"thinking", "redacted_thinking"})
    # Fixed, human-readable notes about what this adapter cannot interpret.
    limitations: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def parse(self, records: Iterable[RawRecord], meta: TraceMetadata) -> Iterator[Event]:
        """Convert native records to canonical events, in source order.

        Must yield an ``Unrecognized`` event for every record it cannot interpret (see ``unrecognized``),
        record session-level facts in ``meta``, keep its own state bounded and never raise on malformed
        input.
        """

    def new_metadata(self) -> TraceMetadata:
        return TraceMetadata(source_format=self.name, adapter_version=self.version)

    def safe_record_type(self, value: Any) -> str | None:
        return value if isinstance(value, str) and value in self.known_record_types else None

    def tool_category(self, name: Any) -> ToolCategory:
        if isinstance(name, str):
            return self.known_tools.get(name, ToolCategory.UNKNOWN)
        return ToolCategory.UNKNOWN


def unrecognized(
    record: RawRecord,
    adapter: SourceAdapter,
    reason: UnrecognizedReason | None = None,
    **kwargs: Any,
) -> Unrecognized:
    """Build the ``Unrecognized`` event for a record (keeps the parsed object in memory as ``raw``)."""
    data = record.data
    record_type = adapter.safe_record_type(data.get("type")) if data is not None else None
    return Unrecognized(
        source=SourceRef(line=record.line, record_type=record_type),
        reason=reason or record.problem or UnrecognizedReason.INCOMPLETE,
        raw=data,
        **kwargs,
    )


def native_id(value: Any) -> str | Unparseable | None:
    """Normalise a native identifier: absent, a non-empty string of sane length, or not interpretable."""
    if value is None:
        return None
    if isinstance(value, str) and 0 < len(value) <= MAX_ID_LENGTH:
        return value
    return Unparseable("invalid_id", value)


def text_value(value: Any) -> str | None:
    """The value if it is a string, else None. Use it before comparing native fields with sets of names:
    JSON lists or objects are unhashable and must never crash an adapter."""
    return value if isinstance(value, str) else None


def remember_context(meta: TraceMetadata, key: str, value: Any) -> None:
    """Keep a distinct context value (cwd, branch...) in memory, within fixed bounds; never reported."""
    if not isinstance(value, str) or not 0 < len(value) <= MAX_CONTEXT_LENGTH:
        return
    values = meta.context.setdefault(key, [])
    if value not in values and len(values) < MAX_CONTEXT_VALUES:
        values.append(value)


def copy_attributes(source: Mapping[str, Any], mapping: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Technical attributes present in ``source`` (source key -> canonical name). Absent keys stay absent."""
    return {name: source[key] for key, name in mapping if key in source}
