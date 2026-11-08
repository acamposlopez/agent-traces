"""Processing engine: resolve the format, stream canonical events from the adapter into processors.

The engine knows nothing about specific formats: adapters come from a registry (or are passed in) and
processors implement the small ``Processor`` protocol. Inspection and schema discovery are the only processors;
other stages would plug in here.

Guarantees enforced here for every adapter, present or future:
- events flow one at a time and nothing is written to disk;
- reasoning blocks (``SourceAdapter.reasoning_block_types``) nested in any raw payload are replaced by a
  content-free stub before a processor sees the event;
- the input file is closed even if an adapter or processor fails.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Protocol

from agent_traces.adapters import DEFAULT_ADAPTERS, KNOWN_UNSUPPORTED, FormatRecognizer, SourceAdapter, get_adapter
from agent_traces.detection import REFUSED, Detection, detect, explicit
from agent_traces.model import RAW_FIELDS, Event, Trace, TraceMetadata, Unparseable, omit_reasoning
from agent_traces.reader import Limits, RawRecord, ReadStats, iter_records

AUTO = "auto"


class FormatError(Exception):
    """The format could not be determined without guessing. Carries the detection, never content."""

    def __init__(self, detection: Detection) -> None:
        super().__init__(f"format {detection.status.value}")
        self.detection = detection


@dataclass(slots=True)
class RunContext:
    detection: Detection
    adapter: SourceAdapter | None  # None only for an empty file in auto mode
    metadata: TraceMetadata | None
    stats: ReadStats
    limits: Limits


class Processor(Protocol):
    name: str

    def start(self, context: RunContext) -> None: ...

    def handle(self, event: Event) -> None: ...

    def finish(self, context: RunContext) -> Any: ...


class RecordObserver(Protocol):
    """Optional processor capability: see each source record before the adapter does.

    Used to describe structure (schema discovery) and to classify records. ``observe`` must not keep the
    record or any value from it.
    """

    def observe(self, record: RawRecord) -> None: ...


def resolve_format(
    path: str | os.PathLike[str],
    fmt: str = AUTO,
    adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS,
    limits: Limits = Limits(),
    unsupported: Sequence[FormatRecognizer] = KNOWN_UNSUPPORTED,
) -> Detection:
    if fmt != AUTO:
        return explicit(get_adapter(fmt, adapters))
    detection = detect(path, adapters, limits, unsupported)
    if detection.status in REFUSED:
        raise FormatError(detection)
    return detection


def run(
    path: str | os.PathLike[str],
    processors: Sequence[Processor],
    *,
    fmt: str = AUTO,
    adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS,
    limits: Limits = Limits(),
    compute_hash: bool = False,
    unsupported: Sequence[FormatRecognizer] = KNOWN_UNSUPPORTED,
) -> dict[str, Any]:
    """Read a file incrementally (the ``inspect`` path) and feed its records to the processors."""
    detection = resolve_format(path, fmt, adapters, limits, unsupported)
    stats = ReadStats()
    with closing(iter_records(path, limits, stats, compute_hash=compute_hash)) as records:
        return run_records(records, processors, detection=detection, limits=limits, stats=stats)


def run_records(
    records: Iterable[RawRecord],
    processors: Sequence[Processor],
    *,
    detection: Detection,
    limits: Limits = Limits(),
    stats: ReadStats | None = None,
) -> dict[str, Any]:
    """Feed already available records (incremental or captured in memory) to the adapter and processors.

    The same guarantees as ``run`` apply: reasoning nested in raw payloads is removed before any processor
    sees an event, observers see each record before the adapter does, and nothing is written.
    """
    adapter = detection.adapter
    context = RunContext(
        detection=detection,
        adapter=adapter,
        metadata=adapter.new_metadata() if adapter else None,
        stats=stats if stats is not None else ReadStats(),
        limits=limits,
    )
    for processor in processors:
        processor.start(context)
    observers = [processor for processor in processors if hasattr(processor, "observe")]
    if adapter is None or context.metadata is None:
        for _ in records:  # empty file: still read it so that the statistics are complete
            pass
    else:
        reasoning_types = adapter.reasoning_block_types
        source = _observed(iter(records), observers) if observers else records
        for seq, event in enumerate(adapter.parse(source, context.metadata)):
            event.seq = seq
            _omit_reasoning(event, reasoning_types)
            for processor in processors:
                processor.handle(event)
    return {processor.name: processor.finish(context) for processor in processors}


def _observed(records: Iterator[RawRecord], observers: list[Any]) -> Iterator[RawRecord]:
    for record in records:
        for observer in observers:
            observer.observe(record)
        yield record


def _omit_reasoning(event: Event, block_types: frozenset[str]) -> None:
    """Remove reasoning from every raw payload of the event, including the raw value kept by an
    ``Unparseable`` field (an id, a timestamp or a text that the adapter could not interpret)."""
    for item in dataclasses.fields(event):
        value = getattr(event, item.name, None)
        if isinstance(value, Unparseable):
            if isinstance(value.raw, (dict, list)):
                setattr(event, item.name, Unparseable(value.reason, omit_reasoning(value.raw, block_types)))
        elif item.name in RAW_FIELDS and isinstance(value, (dict, list)):
            setattr(event, item.name, omit_reasoning(value, block_types))


class EventCollector:
    """Keeps the canonical events in memory (for tests and future in-memory stages). Never persists them.

    Memory grows with the trace: use it for traces that fit in memory, not for streaming inspection.
    """

    name = "events"

    def __init__(self) -> None:
        self.events: list[Event] = []

    def start(self, context: RunContext) -> None:
        self.events = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def finish(self, context: RunContext) -> Trace | None:
        if context.metadata is None:
            return None
        return Trace(metadata=context.metadata, events=self.events)


def load_trace(
    path: str | os.PathLike[str],
    *,
    fmt: str = AUTO,
    adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS,
    limits: Limits = Limits(),
) -> Trace | None:
    return run(path, [EventCollector()], fmt=fmt, adapters=adapters, limits=limits)["events"]
