"""Lightweight categorical wall-clock timing."""

from __future__ import annotations

import time
from collections.abc import Generator, Iterable
from contextlib import AbstractContextManager, contextmanager
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

    def __getitem__(self, tags: str | Iterable[str]) -> AbstractContextManager[None]:
        """Context manager for timing region(s)."""
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, Iterable):
            raise TypeError(f"tag must be str, got {type(tags).__name__}: {tags!r}")
        validated = []
        for tag in tags:
            if not isinstance(tag, str):
                raise TypeError(f"tag must be str, got {type(tag).__name__}: {tag!r}")
            validated.append(tag)
        return self._track(tuple(validated))

    @contextmanager
    def _track(self, tags: tuple[str, ...]) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            for tag in tags:
                self._totals[tag] = self._totals.get(tag, 0.0) + elapsed

    def get_timing_secs(self) -> dict[str, float]:
        """Return timing dict with category breakdowns."""
        metrics: dict[str, float] = {"total": self.total_secs}
        for cat, secs in self._totals.items():
            metrics[cat] = secs
        if self._totals:
            metrics["overhead"] = self.total_secs - sum(self._totals.values())
        return metrics
