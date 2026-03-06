"""Lightweight categorical wall-clock timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Lightweight categorical wall-clock timing."""

    _totals: dict[str, float] = field(default_factory=dict)
    _start_time: float | None = None
    _end_time: float | None = None

    def start(self) -> None:
        """Begin timing. Call before any tracked regions."""
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def stop(self) -> None:
        """End timing. Call after all tracked regions."""
        if self._start_time is not None and self._end_time is None:
            self._end_time = time.perf_counter()

    @property
    def total_secs(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time
