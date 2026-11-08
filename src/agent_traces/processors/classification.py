"""Per-record classification from the events each record produced.

Processors that observe records (``engine.RecordObserver``) feed every record here before the adapter sees
it, then every event. Adapters emit events in source order, so a record is final as soon as an event for a
later line arrives. A record that produced no event was skipped by a documented normalisation.

Memory is bounded: only records still waiting for their events are kept (normally one; a streamed message
spans as many records as its deltas, which adapters cap), with a hard limit beyond which the oldest records
are closed as unknown and counted in ``overflowed``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from agent_traces.model import Event, RecordCategory, combine_categories, event_category

MAX_PENDING_RECORDS = 50_000


class RecordClassifier:
    def __init__(self, on_record: Callable[[int, Any, RecordCategory], None], max_pending: int = MAX_PENDING_RECORDS):
        self._on_record = on_record
        self._pending: deque[list[Any]] = deque()  # [line, payload, category or None]
        self._max_pending = max_pending
        self.overflowed = 0

    def observe(self, line: int, payload: Any = None) -> None:
        self._pending.append([line, payload, None])
        if len(self._pending) > self._max_pending:
            entry = self._pending.popleft()
            self.overflowed += 1
            self._on_record(entry[0], entry[1], entry[2] or RecordCategory.UNKNOWN)

    def event(self, event: Event) -> None:
        first = event.source.line
        last = event.source.line_end or first
        while self._pending and self._pending[0][0] < first:
            self._close(self._pending.popleft())
        category = event_category(event)
        for entry in self._pending:
            if entry[0] > last:
                break
            entry[2] = combine_categories(entry[2], category)

    def finish(self) -> None:
        while self._pending:
            self._close(self._pending.popleft())

    def _close(self, entry: list[Any]) -> None:
        self._on_record(entry[0], entry[1], entry[2] or RecordCategory.SKIPPED)
