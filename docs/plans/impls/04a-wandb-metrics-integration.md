# Wandb Metrics Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scalar metrics tracking to V2 search loop via MetricsTracker protocol.

**Architecture:** Protocol injection with no-op default. MetricsTracker logs per-round scalars to wandb or discards them.

**Tech Stack:** Python dataclasses, typing.Protocol, wandb (optional).

**Design doc:** `docs/plans/2026-03-04-wandb-integration-design.md`

---

## Task 1: MetricsConfig and MetricsTracker protocol

**Files:**
- Modify: `k_search/modular/config.py`
- Create: `k_search/modular/metrics/__init__.py`
- Create: `k_search/modular/metrics/protocol.py`
- Test: `tests/modular/metrics/__init__.py`
- Test: `tests/modular/metrics/test_protocol.py`

**Step 1: Write the failing test**

Create `tests/modular/metrics/__init__.py` (empty).

Create `tests/modular/metrics/test_protocol.py`:

```python
"""Tests for MetricsTracker protocol."""

from k_search.modular.metrics.protocol import MetricsTracker


def test_custom_class_implements_protocol():
    class MyTracker:
        def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
            pass

    assert isinstance(MyTracker(), MetricsTracker)
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_protocol.py -v`
Expected: FAIL with "No module named 'k_search.modular.metrics'"

**Step 3: Write implementation**

Add to `k_search/modular/config.py`:

```python
@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""

    chars_per_token: int = 4
    wandb: bool = False
```

Create `k_search/modular/metrics/protocol.py`:

```python
"""MetricsTracker protocol definition."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsTracker(Protocol):
    """Protocol for logging scalar metrics during search."""

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        ...
```

Create `k_search/modular/metrics/__init__.py`:

```python
"""Metrics tracking for modular."""

from k_search.modular.metrics.protocol import MetricsTracker

__all__ = ["MetricsTracker"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/ tests/modular/
git commit -m "feat(modular): add MetricsConfig and MetricsTracker protocol"
```

---

## Task 2: NoOpMetricsTracker and WandbMetricsTracker

**Files:**
- Create: `k_search/modular/metrics/noop.py`
- Create: `k_search/modular/metrics/wandb.py`
- Modify: `k_search/modular/metrics/__init__.py`
- Test: `tests/modular/metrics/test_trackers.py`

**Step 1: Write the failing test**

Create `tests/modular/metrics/test_trackers.py`:

```python
"""Tests for MetricsTracker implementations."""

from unittest.mock import MagicMock, patch

import pytest

from k_search.modular.config import MetricsConfig
from k_search.modular.metrics.protocol import MetricsTracker


class TestNoOpMetricsTracker:
    def test_implements_protocol(self):
        from k_search.modular.metrics.noop import NoOpMetricsTracker

        assert isinstance(NoOpMetricsTracker(), MetricsTracker)

    def test_log_does_not_raise(self):
        from k_search.modular.metrics.noop import NoOpMetricsTracker

        tracker = NoOpMetricsTracker()
        tracker.log({"score": 0.5}, step=0)
        tracker.log({})


class TestWandbMetricsTracker:
    def test_implements_protocol(self):
        from k_search.modular.metrics.wandb import WandbMetricsTracker

        with patch("k_search.modular.metrics.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            assert isinstance(WandbMetricsTracker(MetricsConfig(wandb=True)), MetricsTracker)

    def test_raises_if_no_active_run(self):
        from k_search.modular.metrics.wandb import WandbMetricsTracker

        with patch("k_search.modular.metrics.wandb.wandb") as mock_wandb:
            mock_wandb.run = None
            with pytest.raises(RuntimeError, match="no active run"):
                WandbMetricsTracker(MetricsConfig(wandb=True))

    def test_log_calls_wandb_log(self):
        from k_search.modular.metrics.wandb import WandbMetricsTracker

        with patch("k_search.modular.metrics.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            tracker = WandbMetricsTracker(MetricsConfig(wandb=True))

            tracker.log({"score": 0.5, "latency_ms": 10.0}, step=5)

            mock_wandb.log.assert_called_once_with({"score": 0.5, "latency_ms": 10.0}, step=5)
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_trackers.py -v`
Expected: FAIL

**Step 3: Write implementation**

Create `k_search/modular/metrics/noop.py`:

```python
"""No-op metrics tracker implementation."""


class NoOpMetricsTracker:
    """Metrics tracker that does nothing."""

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        pass
```

Create `k_search/modular/metrics/wandb.py`:

```python
"""Wandb metrics tracker implementation."""

from k_search.modular.config import MetricsConfig


class WandbMetricsTracker:
    """Metrics tracker that logs to Weights & Biases."""

    def __init__(self, config: MetricsConfig) -> None:
        try:
            import wandb as _wandb
        except ImportError:
            raise RuntimeError("wandb configured but not installed")

        if _wandb.run is None:
            raise RuntimeError(
                "wandb configured but no active run (call wandb.init() first)"
            )

        self._wandb = _wandb

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)
```

Update `k_search/modular/metrics/__init__.py`:

```python
"""Metrics tracking for modular."""

from k_search.modular.metrics.noop import NoOpMetricsTracker
from k_search.modular.metrics.protocol import MetricsTracker
from k_search.modular.metrics.wandb import WandbMetricsTracker

__all__ = ["MetricsTracker", "NoOpMetricsTracker", "WandbMetricsTracker"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_trackers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/metrics/ tests/modular/metrics/
git commit -m "feat(modular): add NoOpMetricsTracker and WandbMetricsTracker"
```

---

## Task 3: create_metrics_trackers factory

**Files:**
- Modify: `k_search/modular/metrics/__init__.py`
- Test: `tests/modular/metrics/test_trackers.py`

**Step 1: Write the failing test**

Add to `tests/modular/metrics/test_trackers.py`:

```python
class TestCreateMetricsTrackers:
    def test_returns_noop_by_default(self):
        from k_search.modular.metrics import create_metrics_trackers
        from k_search.modular.metrics.noop import NoOpMetricsTracker

        trackers = create_metrics_trackers()
        assert len(trackers) == 1
        assert isinstance(trackers[0], NoOpMetricsTracker)

    def test_returns_wandb_when_enabled(self):
        from k_search.modular.metrics import create_metrics_trackers
        from k_search.modular.metrics.wandb import WandbMetricsTracker

        with patch("k_search.modular.metrics.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            trackers = create_metrics_trackers(MetricsConfig(wandb=True))
            assert len(trackers) == 1
            assert isinstance(trackers[0], WandbMetricsTracker)
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_trackers.py::TestCreateMetricsTrackers -v`
Expected: FAIL with "cannot import name 'create_metrics_trackers'"

**Step 3: Write implementation**

Update `k_search/modular/metrics/__init__.py`:

```python
"""Metrics tracking for modular."""

from k_search.modular.config import MetricsConfig
from k_search.modular.metrics.noop import NoOpMetricsTracker
from k_search.modular.metrics.protocol import MetricsTracker
from k_search.modular.metrics.wandb import WandbMetricsTracker


def create_metrics_trackers(config: MetricsConfig | None = None) -> list[MetricsTracker]:
    config = config or MetricsConfig()
    if config.wandb:
        return [WandbMetricsTracker(config)]
    return [NoOpMetricsTracker()]


__all__ = [
    "MetricsTracker",
    "NoOpMetricsTracker",
    "WandbMetricsTracker",
    "create_metrics_trackers",
]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/metrics/test_trackers.py::TestCreateMetricsTrackers -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/metrics/
git commit -m "feat(modular): add create_metrics_trackers factory"
```

---

## Task 4: _build_round_metrics helper

**Files:**
- Modify: `k_search/modular/loop.py`
- Test: `tests/modular/test_loop.py`

**Step 1: Write the failing test**

Add to `tests/modular/test_loop.py`:

```python
class TestBuildRoundMetrics:
    def test_basic_metrics(self):
        from k_search.modular.loop import _build_round_metrics

        result = make_eval_result_mock(
            succeeded=True, metrics={"latency_ms": 5.0, "speedup_factor": 2.0}
        )

        metrics = _build_round_metrics(
            round_time_secs=1.5,
            score=0.8,
            result=result,
            best_score=0.8,
            cumulative_prompt_tokens=100,
            cumulative_completion_tokens=50,
        )

        assert metrics["round_time_secs"] == 1.5
        assert metrics["score"] == 0.8
        assert metrics["succeeded"] == 1
        assert metrics["best_score"] == 0.8
        assert metrics["prompt_tokens_est"] == 100
        assert metrics["completion_tokens_est"] == 50
        assert metrics["total_tokens_est"] == 150
        assert metrics["latency_ms"] == 5.0
        assert metrics["speedup_factor"] == 2.0

    def test_filters_non_numeric_and_bool(self):
        from k_search.modular.loop import _build_round_metrics

        result = make_eval_result_mock(
            succeeded=True,
            metrics={"latency_ms": 5.0, "status": "passed", "passed": True},
        )

        metrics = _build_round_metrics(
            round_time_secs=1.0, score=0.5, result=result, best_score=0.5,
            cumulative_prompt_tokens=0, cumulative_completion_tokens=0,
        )

        assert metrics["latency_ms"] == 5.0
        assert "status" not in metrics
        assert "passed" not in metrics
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/test_loop.py::TestBuildRoundMetrics -v`
Expected: FAIL with "cannot import name '_build_round_metrics'"

**Step 3: Write implementation**

Add to `k_search/modular/loop.py` after imports:

```python
from k_search.modular.protocols.results import EvaluationResult


def _build_round_metrics(
    round_time_secs: float,
    score: float,
    result: EvaluationResult,
    best_score: float,
    cumulative_prompt_tokens: int,
    cumulative_completion_tokens: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "round_time_secs": round_time_secs,
        "score": score,
        "succeeded": int(result.succeeded()),
        "best_score": best_score,
        "prompt_tokens_est": cumulative_prompt_tokens,
        "completion_tokens_est": cumulative_completion_tokens,
        "total_tokens_est": cumulative_prompt_tokens + cumulative_completion_tokens,
    }

    for key, val in result.get_metrics().items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics[key] = val

    return metrics
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/test_loop.py::TestBuildRoundMetrics -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/loop.py tests/modular/test_loop.py
git commit -m "feat(modular): add _build_round_metrics helper"
```

---

## Task 5: Integrate metrics_trackers into run_search

**Files:**
- Modify: `k_search/modular/loop.py`
- Modify: `k_search/modular/__init__.py`
- Test: `tests/modular/test_loop.py`

**Step 1: Write the failing test**

Add to `tests/modular/test_loop.py`:

```python
class TestRunSearchWithMetrics:
    def test_calls_tracker_log_each_round(self):
        from unittest.mock import Mock

        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)
        tracker = Mock()

        run_search(task, evaluator, stub_llm, config, metrics_trackers=tracker)

        assert tracker.log.call_count == 3
        calls = tracker.log.call_args_list
        assert calls[0][1]["step"] == 0
        assert calls[1][1]["step"] == 1
        assert calls[2][1]["step"] == 2

    def test_logs_expected_metrics(self):
        from unittest.mock import Mock

        task = make_task_mock()
        task.scorer.score.return_value = 0.75
        result = make_eval_result_mock(
            succeeded=True, metrics={"latency_ms": 10.0, "speedup_factor": 1.5}
        )
        evaluator = make_evaluator_mock(result)
        config = SearchConfig(max_rounds=1)
        tracker = Mock()

        run_search(task, evaluator, stub_llm, config, metrics_trackers=tracker)

        logged = tracker.log.call_args[0][0]
        assert logged["score"] == 0.75
        assert logged["succeeded"] == 1
        assert logged["latency_ms"] == 10.0
        assert logged["speedup_factor"] == 1.5

    def test_filters_non_numeric_metrics(self):
        from unittest.mock import Mock

        task = make_task_mock()
        result = make_eval_result_mock(
            metrics={"latency_ms": 5.0, "status": "passed", "passed": True}
        )
        evaluator = make_evaluator_mock(result)
        config = SearchConfig(max_rounds=1)
        tracker = Mock()

        run_search(task, evaluator, stub_llm, config, metrics_trackers=tracker)

        logged = tracker.log.call_args[0][0]
        assert logged["latency_ms"] == 5.0
        assert "status" not in logged
        assert "passed" not in logged

    def test_tracks_cumulative_tokens(self):
        from unittest.mock import Mock

        task = make_task_mock()
        task.get_prompt_text.return_value = "a" * 40  # 10 tokens
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)
        tracker = Mock()

        def llm_fixed(prompt: str) -> str:
            return "b" * 20  # 5 tokens

        run_search(task, evaluator, llm_fixed, config, metrics_trackers=tracker)

        calls = tracker.log.call_args_list
        assert calls[2][0][0]["completion_tokens_est"] == 15  # 3 rounds * 5

    def test_accepts_list_of_trackers(self):
        from unittest.mock import Mock

        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)
        t1, t2 = Mock(), Mock()

        run_search(task, evaluator, stub_llm, config, metrics_trackers=[t1, t2])

        assert t1.log.call_count == 1
        assert t2.log.call_count == 1
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/test_loop.py::TestRunSearchWithMetrics -v`
Expected: FAIL with "unexpected keyword argument 'metrics_trackers'"

**Step 3: Write implementation**

Update `k_search/modular/loop.py` - add `metrics_config` param and use `chars_per_token`:

```python
"""Core search loop implementation."""

import logging
import time
from typing import Callable

from k_search.modular.config import MetricsConfig, SearchConfig, SearchResult
from k_search.modular.metrics import MetricsTracker, NoOpMetricsTracker
from k_search.modular.prompts import build_prompt, create_impl
from k_search.modular.protocols.evaluator import Evaluator
from k_search.modular.protocols.results import EvaluationResult
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.types import Round

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], str]


def _build_round_metrics(
    round_time_secs: float,
    score: float,
    result: EvaluationResult,
    best_score: float,
    cumulative_prompt_tokens: int,
    cumulative_completion_tokens: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "round_time_secs": round_time_secs,
        "score": score,
        "succeeded": int(result.succeeded()),
        "best_score": best_score,
        "prompt_tokens_est": cumulative_prompt_tokens,
        "completion_tokens_est": cumulative_completion_tokens,
        "total_tokens_est": cumulative_prompt_tokens + cumulative_completion_tokens,
    }

    for key, val in result.get_metrics().items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics[key] = val

    return metrics


def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
    metrics_config: MetricsConfig | None = None,
    metrics_trackers: MetricsTracker | list[MetricsTracker] | None = None,
) -> SearchResult:
    metrics_config = metrics_config or MetricsConfig()

    if metrics_trackers is None:
        metrics_trackers = [NoOpMetricsTracker()]
    elif not isinstance(metrics_trackers, list):
        metrics_trackers = [metrics_trackers]

    best_impl = None
    best_score = float("-inf")
    best_result = None

    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0

    for round_idx in range(config.max_rounds):
        if best_result:
            metrics = best_result.get_metrics()
            speedup = metrics.get("speedup_factor", "N/A")
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | "
                f"Best: {best_score:.4f} (speedup: {speedup})"
            )
        else:
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | No solution found yet"
            )

        round_start = time.perf_counter()

        best_outcome = None
        if best_impl and best_result:
            best_outcome = Round(impl=best_impl, result=best_result)

        prompt = build_prompt(task, best_outcome)
        code = llm(prompt)
        impl = create_impl(code, round_idx, task_name=task.name)
        result = evaluator.evaluate(impl)
        score = task.scorer.score(result)

        cumulative_prompt_tokens += len(prompt) // metrics_config.chars_per_token
        cumulative_completion_tokens += len(code) // metrics_config.chars_per_token

        if score > best_score:
            best_impl = impl
            best_score = score
            best_result = result

        round_elapsed = time.perf_counter() - round_start

        round_metrics = _build_round_metrics(
            round_time_secs=round_elapsed,
            score=score,
            result=result,
            best_score=best_score,
            cumulative_prompt_tokens=cumulative_prompt_tokens,
            cumulative_completion_tokens=cumulative_completion_tokens,
        )
        for tracker in metrics_trackers:
            tracker.log(round_metrics, step=round_idx)

        logger.info(
            f"Round {round_idx + 1} complete | Score: {score:.4f} | Time: {round_elapsed:.1f}s"
        )

    return SearchResult(
        impl=best_impl,
        score=best_score,
        result=best_result,
        rounds_completed=config.max_rounds,
    )
```

Update `k_search/modular/__init__.py`:

```python
"""Search V2 module."""

from k_search.modular.config import MetricsConfig, SearchConfig, SearchResult
from k_search.modular.loop import run_search

__all__ = ["run_search", "SearchConfig", "SearchResult", "MetricsConfig"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/test_loop.py::TestRunSearchWithMetrics -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd K-Search && python -m pytest tests/modular/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
cd K-Search && git add k_search/modular/ tests/modular/
git commit -m "feat(modular): integrate metrics_trackers into run_search"
```

---

## Validation

```bash
cd K-Search && python -m pytest tests/modular/ -v
cd K-Search && ruff check k_search/modular/
```
