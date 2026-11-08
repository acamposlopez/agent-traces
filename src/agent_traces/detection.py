"""Deterministic format detection from the whole set of structural signals, never from a single key.

Every recognizer (supported adapters and known unsupported formats) scores each examined record with a
positive signal: STRONG when the record matches a full signature of that format (several keys and value
types together), WEAK when it is merely consistent with it. A record strong for exactly one format is
exclusive evidence for it; a record strong for several formats proves none of them (shared evidence).

- exclusive evidence for more than one format          -> contradictory evidence (mixed formats: refuse);
- exclusive evidence for one format, and every shared
  record also matches that format                      -> detected (or "unsupported" for a known,
                                                          unsupported format);
- shared evidence only, or shared records that do not
  match the format with exclusive evidence             -> contradictory evidence (ambiguous: refuse);
- no strong evidence at all                             -> insufficient evidence (refuse, maybe hint).

Records that match nothing (unknown or newer record types, extra fields such as a snake_case session id)
are neutral: they neither prove nor contradict a format.

The decision works on parsed records: ``detect`` reads them incrementally from a path (the ``inspect``
path) and ``detect_records`` receives records that were already parsed in memory.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum

from agent_traces.adapters.base import FormatRecognizer, Signal, SourceAdapter
from agent_traces.reader import Limits, RawRecord, ReadStats, iter_records


class DetectionStatus(StrEnum):
    EXPLICIT = "explicit"  # chosen with --format
    DETECTED = "detected"
    EMPTY = "empty"  # the file has no records at all
    INSUFFICIENT = "insufficient_evidence"  # no format has a strong structural signature
    CONTRADICTORY = "contradictory_evidence"  # several formats have strong signatures (mixed or ambiguous)
    UNSUPPORTED = "unsupported_format"  # the only strong signature belongs to a known, unsupported format


REFUSED = (DetectionStatus.INSUFFICIENT, DetectionStatus.CONTRADICTORY, DetectionStatus.UNSUPPORTED)


@dataclass(slots=True)
class Evidence:
    exclusive: int = 0  # records strong for this format only
    shared: int = 0  # records strong for this format and at least one other
    weak: int = 0

    @property
    def strong(self) -> int:
        return self.exclusive + self.shared


@dataclass(slots=True)
class Detection:
    status: DetectionStatus
    adapter: SourceAdapter | None = None
    records_examined: int = 0
    evidence: dict[str, Evidence] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)  # recognizers with strong evidence
    unsupported: FormatRecognizer | None = None  # set when the status is UNSUPPORTED
    supported_names: frozenset[str] = frozenset()

    @property
    def likely(self) -> list[str]:
        """Supported formats with only weak evidence (a hint when evidence is insufficient, never a decision)."""
        if self.status is not DetectionStatus.INSUFFICIENT:
            return []
        return [name for name, ev in self.evidence.items() if ev.weak and name in self.supported_names]


def explicit(adapter: SourceAdapter) -> Detection:
    return Detection(DetectionStatus.EXPLICIT, adapter)


def detect(
    path: str | os.PathLike[str],
    adapters: Sequence[SourceAdapter],
    limits: Limits,
    unsupported: Sequence[FormatRecognizer] = (),
) -> Detection:
    """Detect the format of a file, reading only as many records as the sniff limits allow."""
    stats = ReadStats()
    with closing(iter_records(path, limits, stats)) as records:
        return _detect(records, adapters, limits, unsupported, lambda: stats.records)


def detect_records(
    records: Iterable[RawRecord],
    adapters: Sequence[SourceAdapter],
    limits: Limits,
    unsupported: Sequence[FormatRecognizer] = (),
) -> Detection:
    """Detect the format of records that were already parsed (same decision rules and sniff limits)."""
    seen = 0

    def counted() -> Iterator[RawRecord]:
        nonlocal seen
        for record in records:
            seen += 1
            yield record

    return _detect(counted(), adapters, limits, unsupported, lambda: seen)


def _detect(
    records: Iterator[RawRecord],
    adapters: Sequence[SourceAdapter],
    limits: Limits,
    unsupported: Sequence[FormatRecognizer],
    records_seen: Callable[[], int],
) -> Detection:
    recognizers: list[FormatRecognizer] = [*adapters, *unsupported]
    evidence = {recognizer.name: Evidence() for recognizer in recognizers}
    shared_sets: set[frozenset[str]] = set()  # distinct combinations of formats matched by one record
    examined = 0
    for record in records:
        if record.data is not None:
            examined += 1
            strong_here = []
            for recognizer in recognizers:
                signal = recognizer.sniff(record.data)
                if signal is Signal.STRONG:
                    strong_here.append(recognizer.name)
                elif signal is Signal.WEAK:
                    evidence[recognizer.name].weak += 1
            if len(strong_here) == 1:
                evidence[strong_here[0]].exclusive += 1
            elif strong_here:
                shared_sets.add(frozenset(strong_here))
                for name in strong_here:
                    evidence[name].shared += 1
        has_strong = any(ev.strong for ev in evidence.values())
        # Bounded work: a normal sample once some format is proven, a larger one while nothing is,
        # and a hard stop on floods of unparseable lines.
        if (has_strong and examined >= limits.sniff_records) or examined >= limits.sniff_max_records:
            break
        if records_seen() >= limits.sniff_max_records * 10:
            break
    total = records_seen()
    detection = Detection(
        DetectionStatus.EMPTY if total == 0 else DetectionStatus.INSUFFICIENT,
        records_examined=examined,
        evidence=evidence,
        supported_names=frozenset(adapter.name for adapter in adapters),
    )
    if total == 0:
        return detection
    detection.matched = [recognizer.name for recognizer in recognizers if evidence[recognizer.name].strong]
    exclusive = [recognizer for recognizer in recognizers if evidence[recognizer.name].exclusive]
    if len(exclusive) > 1 or (not exclusive and shared_sets):
        detection.status = DetectionStatus.CONTRADICTORY
    elif exclusive:
        winner = exclusive[0]
        if any(winner.name not in combination for combination in shared_sets):
            detection.status = DetectionStatus.CONTRADICTORY
        elif isinstance(winner, SourceAdapter) and winner.supported:
            detection.status, detection.adapter = DetectionStatus.DETECTED, winner
        else:
            detection.status, detection.unsupported = DetectionStatus.UNSUPPORTED, winner
    return detection
