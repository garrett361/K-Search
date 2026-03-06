# Span Design

Execution context for Node lifecycle with categorical timing.

## Overview

**Span** wraps a Node's execution lifecycle. Created when executor selects a node, completed when execution finishes. Passive container — executor manages lifecycle.

**Timer** provides lightweight categorical wall-clock timing. Explicit start/stop, context manager for regions.

```
Executor                     Span                      Timer
   │
   ├─► creates ───────────────► owns ─────────────────► (idle)
   │
   ├─► timer.start() ─────────────────────────────────► starts
   │
   ├─► with timer["llm"]: ────────────────────────────► accumulates
   ├─► with timer["eval"]: ───────────────────────────► accumulates
   │
   ├─► timer.stop() ──────────────────────────────────► end_time set
   │
   └─► tracker.log(span.get_metrics()) ───► get_timing_secs()
```

## Timer

`k_search/modular/timer.py`

```python
from __future__ import annotations
from collections.abc import Iterable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import time

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

    def __getitem__(self, categories: str | Iterable[str]) -> AbstractContextManager[None]:
        """Context manager for timing region(s).

        Usage:
            with timer["llm"]: ...
            with timer["llm", "codegen"]: ...

        Assignment (timer["x"] = ...) raises TypeError — only __getitem__ is implemented.
        """
        if isinstance(categories, str):
            categories = [categories]
        return self._track(tuple(categories))

    @contextmanager
    def _track(self, categories: tuple[str, ...]) -> AbstractContextManager[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            for cat in categories:
                self._totals[cat] = self._totals.get(cat, 0.0) + elapsed

    @property
    def total_secs(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    def get_timing_secs(self) -> dict[str, float]:
        """Return timing dict with category breakdowns.

        Keys: "total", category names, "overhead" (if categories tracked).
        """
        metrics: dict[str, float] = {"total": self.total_secs}
        for cat, secs in self._totals.items():
            metrics[cat] = secs
        if self._totals:
            metrics["overhead"] = self.total_secs - sum(self._totals.values())
        return metrics
```

Features:
- Explicit `start()` / `stop()` lifecycle
- `timer["cat"]` or `timer["cat1", "cat2"]` for overlapping categories
- Accepts any iterable of strings
- `AbstractContextManager[None]` return type makes usage clear
- Assignment attempts raise `TypeError` (only `__getitem__` implemented)
- `get_timing_secs()` returns plain keys (no prefix)

## Span

`k_search/modular/span.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from k_search.modular.timer import Timer
from k_search.modular.world.node import Node

@dataclass
class Span:
    """Execution context for a Node's lifecycle.

    Wraps a Node, owns timing via Timer, extensible for future attributes.
    Passive container — executor manages lifecycle.
    """

    node: Node
    timer: Timer = field(default_factory=Timer)
    annotations: dict[str, Any] = field(default_factory=dict)

    def get_metrics(self) -> dict[str, float]:
        """Return metrics dict suitable for MetricsTracker.log().

        Currently delegates to timer. Extend for non-timing metrics.
        """
        return self.timer.get_timing_secs()
```

Fields:
- `node` — the Node being executed
- `timer` — owned Timer instance
- `annotations` — extensibility hook for future data

## Executor Integration (Schematic)

Illustrative usage — executor design is not yet finalized.

```python
async def _execute_node(self, node: Node) -> None:
    span = Span(node=node)
    span.timer.start()

    try:
        prompt = build_prompt(self._tree, node)

        with span.timer["llm", "codegen"]:
            code = await self._llm(prompt)

        impl = self._task.create_implementation(code)

        with span.timer["eval"]:
            result = await asyncio.to_thread(self._evaluator.evaluate, impl)

        round_ = Round(impl=impl, result=result, ...)
        node.cycle = Cycle(rounds=[round_])
        node.status = "closed"

    finally:
        span.timer.stop()
        for tracker in self._metrics_trackers:
            tracker.log(span.get_metrics())
```

## File Structure

```
k_search/modular/
├── timer.py          # Timer class
├── span.py           # Span class (imports Timer)
├── world/
│   ├── node.py
│   ├── cycle.py
│   └── round.py
└── ...
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Name | Span (not Trajectory) | Avoids RL terminology conflict, matches observability |
| Span vs Node extension | Separate class | Single responsibility, transient vs persistent |
| Timer ownership | Span owns Timer | Natural scope, executor passes span around |
| Timer start/stop | Explicit methods | No magic, clear lifecycle |
| Category syntax | `timer["cat"]` | Concise, read-only via `__getitem__` |
| `__getitem__` return type | `AbstractContextManager[None]` | Explicit that it returns a context manager |
| Multi-category | Iterable support | Tag regions with overlapping categories |
| Metrics keys | Plain (no prefix) | Consumer adds prefixes if needed |

## Future Considerations

- Span ID if needed for correlation
- Non-timing metrics in `get_metrics()`
- Multiple spans per node (retries, parallel attempts)
- Event log for replay/debugging (deferred)
