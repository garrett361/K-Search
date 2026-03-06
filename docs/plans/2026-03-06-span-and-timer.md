# Span and Timer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Timer and Span classes for categorical timing of Node execution.

**Architecture:** Timer is a standalone utility for wall-clock timing with explicit lifecycle and context manager API. Span wraps a Node and owns a Timer, providing execution context for metrics.

**Tech Stack:** Python dataclasses, contextlib, time.perf_counter

---

### Task 1: Timer Basic Lifecycle

**Files:**
- Create: `k_search/modular/timer.py`
- Test: `tests/modular/test_timer.py`

**Step 1: Write the failing test**

```python
"""Tests for Timer class."""

import time

from k_search.modular.timer import Timer


def test_timer_total_secs_before_start_is_zero():
    timer = Timer()
    assert timer.total_secs == 0.0


def test_timer_start_stop_captures_elapsed_time():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    assert timer.total_secs >= 0.01
    assert timer.total_secs < 0.1  # sanity bound
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/modular/test_timer.py -v --tb=short`
Expected: FAIL with "ModuleNotFoundError: No module named 'k_search.modular.timer'"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/modular/test_timer.py -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add k_search/modular/timer.py tests/modular/test_timer.py
git commit -m "feat(timer): add Timer with start/stop lifecycle"
```

---

### Task 2: Timer Single Category Tracking

**Files:**
- Modify: `k_search/modular/timer.py`
- Modify: `tests/modular/test_timer.py`

**Step 1: Write the failing test**

Add to `tests/modular/test_timer.py`:

```python
def test_timer_single_category_tracking():
    timer = Timer()
    timer.start()

    with timer["llm"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "llm" in timing
    assert timing["llm"] >= 0.01
    assert timing["total"] >= timing["llm"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/modular/test_timer.py::test_timer_single_category_tracking -v --tb=short`
Expected: FAIL with "TypeError: 'Timer' object is not subscriptable"

**Step 3: Write minimal implementation**

Add to `Timer` class in `k_search/modular/timer.py`:

```python
from collections.abc import Iterable
from contextlib import AbstractContextManager, contextmanager
```

Add methods:

```python
    def __getitem__(self, categories: str | Iterable[str]) -> AbstractContextManager[None]:
        """Context manager for timing region(s)."""
        if isinstance(categories, str):
            categories = [categories]
        validated = []
        for cat in categories:
            if not isinstance(cat, str):
                raise TypeError(f"category must be str, got {type(cat).__name__}: {cat!r}")
            validated.append(cat)
        return self._track(tuple(validated))

    @contextmanager
    def _track(self, categories: tuple[str, ...]) -> AbstractContextManager[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            for cat in categories:
                self._totals[cat] = self._totals.get(cat, 0.0) + elapsed

    def get_timing_secs(self) -> dict[str, float]:
        """Return timing dict with category breakdowns."""
        metrics: dict[str, float] = {"total": self.total_secs}
        for cat, secs in self._totals.items():
            metrics[cat] = secs
        if self._totals:
            metrics["overhead"] = self.total_secs - sum(self._totals.values())
        return metrics
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/modular/test_timer.py -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add k_search/modular/timer.py tests/modular/test_timer.py
git commit -m "feat(timer): add category tracking with context manager"
```

---

### Task 3: Timer Multi-Category and Overhead

**Files:**
- Modify: `tests/modular/test_timer.py`

**Step 1: Write the failing tests**

Add to `tests/modular/test_timer.py`:

```python
def test_timer_multi_category_tracking():
    timer = Timer()
    timer.start()

    with timer["llm", "codegen"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert timing["llm"] >= 0.01
    assert timing["codegen"] >= 0.01
    assert timing["llm"] == timing["codegen"]  # same elapsed time for both


def test_timer_overhead_calculation():
    timer = Timer()
    timer.start()

    time.sleep(0.01)  # overhead
    with timer["work"]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "overhead" in timing
    assert timing["overhead"] >= 0.01
    assert timing["total"] >= timing["work"] + timing["overhead"]


def test_timer_no_overhead_when_no_categories():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    timing = timer.get_timing_secs()
    assert "overhead" not in timing
    assert timing["total"] >= 0.01


def test_timer_rejects_non_string_category():
    import pytest

    timer = Timer()
    with pytest.raises(TypeError, match="category must be str, got int: 123"):
        with timer[123]:
            pass
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/modular/test_timer.py -v --tb=short`
Expected: PASS (implementation already handles these cases)

**Step 3: Commit**

```bash
git add tests/modular/test_timer.py
git commit -m "test(timer): add multi-category and overhead tests"
```

---

### Task 4: Timer Iterable Support

**Files:**
- Modify: `tests/modular/test_timer.py`

**Step 1: Write the failing test**

Add to `tests/modular/test_timer.py`:

```python
def test_timer_accepts_list_of_categories():
    timer = Timer()
    timer.start()

    categories = ["llm", "world_model"]
    with timer[categories]:
        time.sleep(0.01)

    timer.stop()

    timing = timer.get_timing_secs()
    assert "llm" in timing
    assert "world_model" in timing
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/modular/test_timer.py::test_timer_accepts_list_of_categories -v --tb=short`
Expected: PASS (implementation already handles iterables)

**Step 3: Commit**

```bash
git add tests/modular/test_timer.py
git commit -m "test(timer): verify iterable category support"
```

---

### Task 5: Span Class

**Files:**
- Create: `k_search/modular/span.py`
- Create: `tests/modular/test_span.py`

**Step 1: Write the failing test**

```python
"""Tests for Span class."""

import time
from unittest.mock import MagicMock

from k_search.modular.span import Span
from k_search.modular.timer import Timer


def test_span_wraps_node_with_timer():
    node = MagicMock()
    span = Span(node=node)

    assert span.node is node
    assert isinstance(span.timer, Timer)
    assert span.annotations == {}


def test_span_get_metrics_delegates_to_timer():
    node = MagicMock()
    span = Span(node=node)

    span.timer.start()
    with span.timer["llm"]:
        time.sleep(0.01)
    span.timer.stop()

    metrics = span.get_metrics()
    assert "total" in metrics
    assert "llm" in metrics
    assert metrics["llm"] >= 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/modular/test_span.py -v --tb=short`
Expected: FAIL with "ModuleNotFoundError: No module named 'k_search.modular.span'"

**Step 3: Write minimal implementation**

```python
"""Execution context for Node lifecycle."""

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
        """Return metrics dict suitable for MetricsTracker.log()."""
        return self.timer.get_timing_secs()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/modular/test_span.py -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add k_search/modular/span.py tests/modular/test_span.py
git commit -m "feat(span): add Span class wrapping Node with Timer"
```

---

### Task 6: Export from modular package

**Files:**
- Modify: `k_search/modular/__init__.py`

**Step 1: Check current exports**

Read `k_search/modular/__init__.py` to see current export pattern.

**Step 2: Add exports**

Add to exports:

```python
from k_search.modular.timer import Timer
from k_search.modular.span import Span
```

**Step 3: Verify imports work**

Run: `python -c "from k_search.modular import Timer, Span; print('OK')"`
Expected: "OK"

**Step 4: Commit**

```bash
git add k_search/modular/__init__.py
git commit -m "feat(modular): export Timer and Span"
```

---

### Task 7: Run full test suite and lint

**Step 1: Run all tests**

Run: `pytest tests/modular/test_timer.py tests/modular/test_span.py -v --tb=short`
Expected: All PASS

**Step 2: Run linter**

Run: `ruff check k_search/modular/timer.py k_search/modular/span.py`
Expected: No errors

**Step 3: Run type checker**

Run: `ty check k_search/modular/timer.py k_search/modular/span.py`
Expected: No errors (or acceptable warnings)

**Step 4: Format**

Run: `ruff format k_search/modular/timer.py k_search/modular/span.py`

**Step 5: Final commit if any formatting changes**

```bash
git add -u && git commit -m "style: format timer and span" || echo "No changes"
```
