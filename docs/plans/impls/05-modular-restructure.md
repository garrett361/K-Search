# Modular Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate `modular/` and `modular/` into a unified `modular/` module with consistent naming.

**Architecture:** Merge two artificially separated modules into one cohesive unit. All protocols live in `protocols/`. Implementation-specific directories (`metrics/`, `artifacts/`, `adapters/`) hold only concrete implementations. Rename `Round` → `Round` with expanded fields.

**Tech Stack:** Python 3.12, dataclasses, typing.Protocol

**Note:** All commands assume you're in the `K-Search/` directory. Run `cd K-Search` first if needed.

---

## Target Structure

```
k_search/modular/
├── __init__.py
├── round.py                    # Round (expanded container)
├── results.py                  # CheckResult, AnalysisResult
├── llm_utils.py
├── loop.py
├── config.py
├── prompts.py
├── protocols/
│   ├── __init__.py
│   ├── eval_result.py          # EvaluationResult protocol
│   ├── impl.py                 # Implementation protocol
│   ├── metrics_tracker.py      # MetricsTracker protocol (moved)
│   ├── artifact_store.py       # ArtifactStore protocol (moved)
│   ├── input_generator.py
│   ├── reference_impl.py
│   ├── correctness.py
│   ├── scorer.py
│   ├── feedback_provider.py
│   ├── evaluator.py
│   ├── analyzer.py
│   └── task_definition.py
├── adapters/
│   ├── __init__.py
│   └── gpu_mode/
│       ├── __init__.py
│       ├── wrappers.py         # GpuModeEvaluationResult, GpuModeImplementation
│       ├── task_definition.py
│       └── evaluator.py
├── metrics/
│   ├── __init__.py
│   ├── noop.py                 # NoOpMetricsTracker
│   └── wandb.py                # WandbMetricsTracker
└── artifacts/
    ├── __init__.py
    ├── noop.py                 # NoOpArtifactStore
    ├── local.py                # LocalArtifactStore
    └── wandb.py                # WandbArtifactStore
```

---

## Phase 1: Create Directory Structure

### Task 1.1: Create modular directory skeleton

**Files:**
- Create: `k_search/modular/__init__.py`
- Create: `k_search/modular/protocols/__init__.py`
- Create: `k_search/modular/adapters/__init__.py`
- Create: `k_search/modular/adapters/gpu_mode/__init__.py`
- Create: `k_search/modular/metrics/__init__.py`
- Create: `k_search/modular/artifacts/__init__.py`

**Step 1: Create directories with empty inits**

```bash
cd K-Search
mkdir -p k_search/modular/protocols
mkdir -p k_search/modular/adapters/gpu_mode
mkdir -p k_search/modular/metrics
mkdir -p k_search/modular/artifacts
touch k_search/modular/__init__.py
touch k_search/modular/protocols/__init__.py
touch k_search/modular/adapters/__init__.py
touch k_search/modular/adapters/gpu_mode/__init__.py
touch k_search/modular/metrics/__init__.py
touch k_search/modular/artifacts/__init__.py
```

**Step 2: Commit**

```bash
git add k_search/modular/
git commit -m "feat(modular): create directory skeleton"
```

---

## Phase 2: Move Core Types

### Task 2.1: Create results.py

**Files:**
- Create: `k_search/modular/results.py`

**Step 1: Create results.py**

```python
"""Result types for evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Result of correctness check."""

    passed: bool
    message: str = ""
    criteria: dict[str, Any] | None = None


@dataclass
class AnalysisResult:
    """Result of post-evaluation analysis."""

    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_artifact: bytes | None = None
    strategic_guidance: str | None = None
```

**Step 2: Commit**

```bash
git add k_search/modular/results.py
git commit -m "feat(modular): add CheckResult and AnalysisResult"
```

---

### Task 2.2: Create round.py with expanded Round

**Files:**
- Create: `k_search/modular/round.py`

**Step 1: Create round.py**

```python
"""Round container for complete iteration context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.protocols.eval_result import EvaluationResult
    from k_search.modular.protocols.impl import Implementation
    from k_search.modular.results import AnalysisResult


@dataclass
class Round:
    """Complete result of a search iteration.

    Contains all context needed by downstream consumers (FeedbackProvider,
    ArtifactStore, metrics) to understand and report on the round.
    """

    impl: Implementation
    result: EvaluationResult
    prompt: str
    llm_response: str
    prompt_tokens: int
    completion_tokens: int
    duration_secs: float
    score: float
    analysis: AnalysisResult | None = None
```

**Step 2: Commit**

```bash
git add k_search/modular/round.py
git commit -m "feat(modular): add Round dataclass with expanded fields"
```

---

## Phase 3: Move Protocol Files

### Task 3.1: Create eval_result.py and impl.py protocols

**Files:**
- Create: `k_search/modular/protocols/eval_result.py`
- Create: `k_search/modular/protocols/impl.py`

**Step 1: Create eval_result.py**

```python
"""EvaluationResult protocol definition."""

from typing import Any, Protocol


class EvaluationResult(Protocol):
    """Generic evaluation result."""

    def is_success(self) -> bool:
        """Return True if evaluation passed."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return evaluation metrics as dict."""
        ...

    def get_log(self) -> str:
        """Return evaluation log/output."""
        ...
```

**Step 2: Create impl.py**

```python
"""Implementation protocol definition."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class Implementation(Protocol):
    """Data container for code to be evaluated.

    Format is task-specific:
    - str: single source file
    - dict[str, str]: multiple files {filename: content}
    - Path: reference to file on disk
    """

    name: str
    """Identifier for this implementation."""

    content: Any
    """The implementation data."""

    @contextmanager
    def artifact_dir(self) -> Iterator[Path | None]:
        """Yield directory containing files for artifact storage, or None."""
        yield None
```

**Step 3: Commit**

```bash
git add k_search/modular/protocols/eval_result.py k_search/modular/protocols/impl.py
git commit -m "feat(modular): add EvaluationResult and Implementation protocols"
```

---

### Task 3.2: Create metrics_tracker.py protocol

**Files:**
- Create: `k_search/modular/protocols/metrics_tracker.py`

**Step 1: Create metrics_tracker.py**

```python
"""MetricsTracker protocol definition."""

from typing import Protocol


class MetricsTracker(Protocol):
    """Protocol for metrics tracking implementations."""

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log metrics to the tracking backend."""
        ...
```

**Step 2: Commit**

```bash
git add k_search/modular/protocols/metrics_tracker.py
git commit -m "feat(modular): add MetricsTracker protocol"
```

---

### Task 3.3: Create artifact_store.py protocol

**Files:**
- Create: `k_search/modular/protocols/artifact_store.py`

**Step 1: Create artifact_store.py**

```python
"""ArtifactStore protocol definition."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from k_search.modular.round import Round


class ArtifactStore(Protocol):
    """Protocol for storing artifacts during search."""

    def store(self, round_: Round, round_idx: int) -> None:
        """Store artifacts from a search round."""
        ...
```

**Step 2: Commit**

```bash
git add k_search/modular/protocols/artifact_store.py
git commit -m "feat(modular): add ArtifactStore protocol"
```

---

### Task 3.4: Copy remaining protocol files

**Files:**
- Copy: `modular/protocols/*.py` → `modular/protocols/`

**Step 1: Copy files**

```bash
cd K-Search
cp k_search/modular/protocols/input_generator.py k_search/modular/protocols/
cp k_search/modular/protocols/reference_impl.py k_search/modular/protocols/
cp k_search/modular/protocols/correctness.py k_search/modular/protocols/
cp k_search/modular/protocols/scorer.py k_search/modular/protocols/
cp k_search/modular/protocols/feedback_provider.py k_search/modular/protocols/
cp k_search/modular/protocols/evaluator.py k_search/modular/protocols/
cp k_search/modular/protocols/analyzer.py k_search/modular/protocols/
cp k_search/modular/protocols/task_definition.py k_search/modular/protocols/
```

**Step 2: Update imports in copied files**

In each file, replace:
- `from k_search.modular.types import` → `from k_search.modular.results import` (for CheckResult, AnalysisResult)
- `from k_search.modular.types import Round` → `from k_search.modular.round import Round`
- `from k_search.modular.protocols.results import` → split into `from k_search.modular.protocols.eval_result import` and `from k_search.modular.protocols.impl import`

**Step 3: Commit**

```bash
git add k_search/modular/protocols/
git commit -m "feat(modular): copy remaining protocol files"
```

---

### Task 3.5: Write protocols __init__.py

**Files:**
- Modify: `k_search/modular/protocols/__init__.py`

**Step 1: Write exports**

```python
"""Protocol definitions for the modular framework."""

from k_search.modular.protocols.analyzer import Analyzer
from k_search.modular.protocols.artifact_store import ArtifactStore
from k_search.modular.protocols.correctness import CorrectnessChecker
from k_search.modular.protocols.eval_result import EvaluationResult
from k_search.modular.protocols.evaluator import Evaluator
from k_search.modular.protocols.feedback_provider import FeedbackProvider
from k_search.modular.protocols.impl import Implementation
from k_search.modular.protocols.input_generator import InputGenerator
from k_search.modular.protocols.metrics_tracker import MetricsTracker
from k_search.modular.protocols.reference_impl import ReferenceImpl
from k_search.modular.protocols.scorer import Scorer
from k_search.modular.protocols.task_definition import TaskDefinition

__all__ = [
    "Analyzer",
    "ArtifactStore",
    "CorrectnessChecker",
    "EvaluationResult",
    "Evaluator",
    "FeedbackProvider",
    "Implementation",
    "InputGenerator",
    "MetricsTracker",
    "ReferenceImpl",
    "Scorer",
    "TaskDefinition",
]
```

**Step 2: Verify imports**

```bash
cd K-Search && python -c "from k_search.modular.protocols import EvaluationResult, MetricsTracker, ArtifactStore"
```

**Step 3: Commit**

```bash
git add k_search/modular/protocols/__init__.py
git commit -m "feat(modular): add protocols __init__ exports"
```

---

## Phase 4: Move Implementations

### Task 4.1: Copy metrics implementations

**Files:**
- Copy: `modular/metrics/noop.py` → `modular/metrics/noop.py`
- Copy: `modular/metrics/wandb.py` → `modular/metrics/wandb.py`

**Step 1: Copy files**

```bash
cp k_search/modular/metrics/noop.py k_search/modular/metrics/
cp k_search/modular/metrics/wandb.py k_search/modular/metrics/
```

**Step 2: Update imports**

In both files, remove protocol imports (they don't need to import the protocol they implement).

**Step 3: Write metrics __init__.py**

```python
"""Metrics tracking implementations."""

from k_search.modular.metrics.noop import NoOpMetricsTracker
from k_search.modular.metrics.wandb import WandbMetricsTracker, create_metrics_trackers

__all__ = [
    "NoOpMetricsTracker",
    "WandbMetricsTracker",
    "create_metrics_trackers",
]
```

**Step 4: Commit**

```bash
git add k_search/modular/metrics/
git commit -m "feat(modular): add metrics implementations"
```

---

### Task 4.2: Copy artifacts implementations

**Files:**
- Copy: `modular/artifacts/noop.py` → `modular/artifacts/noop.py`
- Copy: `modular/artifacts/local.py` → `modular/artifacts/local.py`
- Copy: `modular/artifacts/wandb.py` → `modular/artifacts/wandb.py`

**Step 1: Copy files**

```bash
cp k_search/modular/artifacts/noop.py k_search/modular/artifacts/
cp k_search/modular/artifacts/local.py k_search/modular/artifacts/
cp k_search/modular/artifacts/wandb.py k_search/modular/artifacts/
```

**Step 2: Update imports**

In each file:
- Remove protocol import
- `from k_search.modular.types import Round` → `from k_search.modular.round import Round`
- Update method signatures: `outcome: Round` → `round_: Round`
- Update attribute access: `outcome.impl` → `round_.impl`, etc.

**Step 3: Write artifacts __init__.py**

```python
"""Artifact storage implementations."""

from k_search.modular.artifacts.local import LocalArtifactStore
from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular.artifacts.wandb import WandbArtifactStore, create_artifact_stores

__all__ = [
    "LocalArtifactStore",
    "NoOpArtifactStore",
    "WandbArtifactStore",
    "create_artifact_stores",
]
```

**Step 4: Commit**

```bash
git add k_search/modular/artifacts/
git commit -m "feat(modular): add artifacts implementations"
```

---

### Task 4.3: Copy gpu_mode adapter

**Files:**
- Copy: `modular/adapters/gpu_mode/types.py` → `modular/adapters/gpu_mode/wrappers.py`
- Copy: `modular/adapters/gpu_mode/task_definition.py` → `modular/adapters/gpu_mode/task_definition.py`
- Copy: `modular/adapters/gpu_mode/evaluator.py` → `modular/adapters/gpu_mode/evaluator.py`

**Step 1: Copy and rename files**

```bash
cp k_search/modular/adapters/gpu_mode/types.py k_search/modular/adapters/gpu_mode/wrappers.py
cp k_search/modular/adapters/gpu_mode/task_definition.py k_search/modular/adapters/gpu_mode/
cp k_search/modular/adapters/gpu_mode/evaluator.py k_search/modular/adapters/gpu_mode/
```

**Step 2: Update imports in copied files**

In `task_definition.py`:
- `from k_search.modular.adapters.gpu_mode.types import` → `from k_search.modular.adapters.gpu_mode.wrappers import`
- `from k_search.modular.types import` → `from k_search.modular.results import`

In `evaluator.py`:
- `from k_search.modular.adapters.gpu_mode.types import` → `from k_search.modular.adapters.gpu_mode.wrappers import`

**Step 3: Write gpu_mode __init__.py**

```python
"""GPU Mode adapter implementations."""

from k_search.modular.adapters.gpu_mode.evaluator import GpuModeEvaluator
from k_search.modular.adapters.gpu_mode.task_definition import GpuModeTriMulTaskDefinition
from k_search.modular.adapters.gpu_mode.wrappers import (
    GpuModeEvaluationResult,
    GpuModeImplementation,
)

__all__ = [
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
    "GpuModeTriMulTaskDefinition",
]
```

**Step 4: Write adapters __init__.py**

```python
"""Adapter implementations for specific task backends."""

from k_search.modular.adapters.gpu_mode import (
    GpuModeEvaluationResult,
    GpuModeEvaluator,
    GpuModeImplementation,
    GpuModeTriMulTaskDefinition,
)

__all__ = [
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
    "GpuModeTriMulTaskDefinition",
]
```

**Step 5: Commit**

```bash
git add k_search/modular/adapters/
git commit -m "feat(modular): add gpu_mode adapter"
```

---

## Phase 5: Move Loop Components

### Task 5.1: Copy llm_utils.py

**Files:**
- Copy: `modular/llm_utils.py` → `modular/llm_utils.py`

**Step 1: Copy file**

```bash
cp k_search/modular/llm_utils.py k_search/modular/
```

**Step 2: Commit**

```bash
git add k_search/modular/llm_utils.py
git commit -m "feat(modular): add llm_utils"
```

---

### Task 5.2: Copy config.py

**Files:**
- Copy: `modular/config.py` → `modular/config.py`

**Step 1: Copy file**

```bash
cp k_search/modular/config.py k_search/modular/
```

**Step 2: Commit**

```bash
git add k_search/modular/config.py
git commit -m "feat(modular): add config"
```

---

### Task 5.3: Copy prompts.py

**Files:**
- Copy: `modular/prompts.py` → `modular/prompts.py`

**Step 1: Copy file**

```bash
cp k_search/modular/prompts.py k_search/modular/
```

**Step 2: Update imports**

- `from k_search.modular.types import Round` → `from k_search.modular.round import Round`
- Update function signatures and docstrings: `Round` → `Round`

**Step 3: Commit**

```bash
git add k_search/modular/prompts.py
git commit -m "feat(modular): add prompts"
```

---

### Task 5.4: Copy and update loop.py

**Files:**
- Copy: `modular/loop.py` → `modular/loop.py`

**Step 1: Copy file**

```bash
cp k_search/modular/loop.py k_search/modular/
```

**Step 2: Update imports**

```python
from k_search.modular.artifacts import NoOpArtifactStore
from k_search.modular.config import MetricsConfig, SearchConfig, SearchResult
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.prompts import build_prompt
from k_search.modular.protocols import ArtifactStore, Evaluator, EvaluationResult, MetricsTracker
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.round import Round
```

**Step 3: Update Round construction**

Replace `Round(impl=impl, result=result)` with:

```python
round_ = Round(
    impl=impl,
    result=result,
    prompt=prompt,
    llm_response=code,
    prompt_tokens=prompt_toks,
    completion_tokens=completion_toks,
    duration_secs=round_elapsed,
    score=score,
)
```

**Step 4: Update best tracking and artifact store calls**

- `best_outcome: Round | None = None` → `best_round: Round | None = None`
- `store.store(outcome, round_idx)` → `store.store(round_, round_idx)`

**Step 5: Commit**

```bash
git add k_search/modular/loop.py
git commit -m "feat(modular): add loop with Round construction"
```

---

### Task 5.5: Write modular __init__.py

**Files:**
- Modify: `k_search/modular/__init__.py`

**Step 1: Write complete exports**

```python
"""Modular framework: protocol-based abstractions for code optimization tasks."""

from k_search.modular.config import ArtifactConfig, MetricsConfig, SearchConfig, SearchResult
from k_search.modular.loop import LLMCall, run_search
from k_search.modular.results import AnalysisResult, CheckResult
from k_search.modular.round import Round

from k_search.modular.protocols import (
    Analyzer,
    ArtifactStore,
    CorrectnessChecker,
    EvaluationResult,
    Evaluator,
    FeedbackProvider,
    Implementation,
    InputGenerator,
    MetricsTracker,
    ReferenceImpl,
    Scorer,
    TaskDefinition,
)
from k_search.modular.adapters import (
    GpuModeEvaluationResult,
    GpuModeEvaluator,
    GpuModeImplementation,
    GpuModeTriMulTaskDefinition,
)
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.artifacts import NoOpArtifactStore

__all__ = [
    # Config
    "ArtifactConfig",
    "MetricsConfig",
    "SearchConfig",
    "SearchResult",
    # Loop
    "LLMCall",
    "run_search",
    # Types
    "AnalysisResult",
    "CheckResult",
    "Round",
    # Protocols
    "Analyzer",
    "ArtifactStore",
    "CorrectnessChecker",
    "EvaluationResult",
    "Evaluator",
    "FeedbackProvider",
    "Implementation",
    "InputGenerator",
    "MetricsTracker",
    "ReferenceImpl",
    "Scorer",
    "TaskDefinition",
    # Adapters
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
    "GpuModeTriMulTaskDefinition",
    # Default implementations
    "NoOpMetricsTracker",
    "NoOpArtifactStore",
]
```

**Step 2: Verify imports**

```bash
cd K-Search && python -c "from k_search.modular import run_search, Round, GpuModeTriMulTaskDefinition, MetricsTracker, ArtifactStore"
```

**Step 3: Commit**

```bash
git add k_search/modular/__init__.py
git commit -m "feat(modular): add complete module exports"
```

---

## Phase 6: Update Entry Point

### Task 6.1: Update run_modular.py

**Files:**
- Modify: `run_modular.py`

**Step 1: Update imports**

Replace:
```python
from k_search.modular import run_search, SearchConfig, ArtifactConfig
from k_search.modular.artifacts import create_artifact_stores
from k_search.modular.config import MetricsConfig
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
```

With:
```python
from k_search.modular import run_search, SearchConfig, ArtifactConfig
from k_search.modular.artifacts import create_artifact_stores
from k_search.modular.config import MetricsConfig
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
```

**Step 2: Verify**

```bash
cd K-Search && python run_modular.py --help
```

**Step 3: Commit**

```bash
git add run_modular.py
git commit -m "refactor: update run_modular.py to use modular imports"
```

---

## Phase 7: Move Tests

### Task 7.1: Create modular test directory and move tests

**Files:**
- Create: `tests/modular/__init__.py`
- Move: `tests/modular/test_wrappers.py` → `tests/modular/test_gpu_mode_wrappers.py`
- Move: `tests/modular/test_gpu_mode_task_definition.py` → `tests/modular/test_gpu_mode_task_definition.py`
- Move: `tests/modular/test_e2e_causal_conv1d.py` → `tests/modular/test_e2e_causal_conv1d.py`
- Move: `tests/modular/test_loop.py` → `tests/modular/test_loop.py`
- Move: `tests/modular/test_e2e_search.py` → `tests/modular/test_e2e_search.py`
- Move: `tests/modular/test_metrics.py` → `tests/modular/test_metrics.py`
- Move: `tests/modular/artifacts/test_stores.py` → `tests/modular/test_artifacts.py`

**Step 1: Create directory and copy files**

```bash
mkdir -p tests/modular
touch tests/modular/__init__.py
cp tests/modular/test_wrappers.py tests/modular/test_gpu_mode_wrappers.py
cp tests/modular/test_gpu_mode_task_definition.py tests/modular/
cp tests/modular/test_e2e_causal_conv1d.py tests/modular/
cp tests/modular/test_loop.py tests/modular/
cp tests/modular/test_e2e_search.py tests/modular/
cp tests/modular/test_metrics.py tests/modular/
cp tests/modular/artifacts/test_stores.py tests/modular/test_artifacts.py
```

**Step 2: Update imports in all test files**

- `from k_search.modular` → `from k_search.modular`
- `from k_search.modular` → `from k_search.modular`
- `Round` → `Round`
- `from k_search.modular.adapters.gpu_mode` → `from k_search.modular.adapters.gpu_mode`

**Step 3: Verify tests pass**

```bash
cd K-Search && pytest tests/modular/ -v
```

**Step 4: Commit**

```bash
git add tests/modular/
git commit -m "refactor: move tests to modular directory"
```

---

### Task 7.2: Organize misplaced root tests

**Files:**
- Move: `tests/test_gpu_mode_causal_conv1d.py` → `tests/tasks/gpu_mode/test_causal_conv1d.py`
- Move: `tests/test_gpu_mode_moe.py` → `tests/tasks/gpu_mode/test_moe.py`
- Move: `tests/test_gpu_mode_reference_benchmark.py` → `tests/tasks/gpu_mode/test_reference_benchmark.py`
- Move: `tests/test_kernel_generator.py` → `tests/kernel_generators/test_generator.py`
- Move: `tests/test_world_model_llm_integration.py` → `tests/kernel_generators/test_world_model.py`

**Step 1: Create directories and move**

```bash
mkdir -p tests/tasks/gpu_mode tests/kernel_generators
touch tests/tasks/__init__.py tests/tasks/gpu_mode/__init__.py tests/kernel_generators/__init__.py
git mv tests/test_gpu_mode_causal_conv1d.py tests/tasks/gpu_mode/test_causal_conv1d.py
git mv tests/test_gpu_mode_moe.py tests/tasks/gpu_mode/test_moe.py
git mv tests/test_gpu_mode_reference_benchmark.py tests/tasks/gpu_mode/test_reference_benchmark.py
git mv tests/test_kernel_generator.py tests/kernel_generators/test_generator.py
git mv tests/test_world_model_llm_integration.py tests/kernel_generators/test_world_model.py
```

**Step 2: Commit**

```bash
git add tests/tasks/ tests/kernel_generators/
git commit -m "refactor: organize tests into proper directory structure"
```

---

## Phase 8: Cleanup

### Task 8.1: Delete old modular directory

**Step 1: Verify no imports remain**

```bash
cd K-Search && grep -r "from k_search.modular" --include="*.py" k_search/ tests/ run_modular.py
```

Expected: No matches

**Step 2: Delete**

```bash
git rm -r k_search/modular/
git commit -m "refactor: remove old modular directory"
```

---

### Task 8.2: Delete old modular directory

**Step 1: Verify no imports remain**

```bash
cd K-Search && grep -r "from k_search.modular" --include="*.py" k_search/ tests/ run_modular.py
```

Expected: No matches

**Step 2: Delete**

```bash
git rm -r k_search/modular/
git commit -m "refactor: remove old modular directory"
```

---

### Task 8.3: Delete old test directories

**Step 1: Delete**

```bash
git rm -r tests/modular/ tests/modular/
git commit -m "refactor: remove old test directories"
```

---

## Phase 9: Update Documentation

### Task 9.1: Update CLAUDE.md

**Step 1: Update architecture section**

Replace references to `modular` and `modular` with `modular`. Add `Round` to key types.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for modular restructure"
```

---

### Task 9.2: Update doc files

**Step 1: Batch find/replace in docs**

```bash
cd K-Search
find docs/ -name "*.md" -exec sed -i 's/modular/modular/g' {} \;
find docs/ -name "*.md" -exec sed -i 's/modular/modular/g' {} \;
find docs/ -name "*.md" -exec sed -i 's/Round/Round/g' {} \;
```

**Step 2: Commit**

```bash
git add docs/
git commit -m "docs: update import paths for modular restructure"
```

---

## Phase 10: Final Verification

### Task 10.1: Run full test suite

```bash
pytest tests/ -v
```

Expected: All tests pass

### Task 10.2: Run type checker

```bash
ty check k_search/modular/
```

Expected: No errors

### Task 10.3: Run linter

```bash
ruff check k_search/modular/
```

Expected: No errors

### Task 10.4: Run e2e search test

Run the entry point with a quick dry-run to verify the full pipeline works:

```bash
python run_modular.py \
    --task causal_conv1d \
    --model-name gpt-4o \
    --max-rounds 1 \
    --base-url https://api.openai.com/v1
```

Expected: Completes one round without import errors. (May fail on API key / GPU but should get past imports and config.)

Alternatively, run the e2e test if available:

```bash
pytest tests/modular/test_e2e_search.py -v
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1.1 | Create directory skeleton |
| 2 | 2.1-2.2 | Move core types (CheckResult, AnalysisResult, Round) |
| 3 | 3.1-3.5 | Move all protocols to protocols/ |
| 4 | 4.1-4.3 | Move implementations (metrics, artifacts, adapters) |
| 5 | 5.1-5.5 | Move loop components |
| 6 | 6.1 | Update entry point |
| 7 | 7.1-7.2 | Reorganize tests |
| 8 | 8.1-8.3 | Delete old directories |
| 9 | 9.1-9.2 | Update documentation |
| 10 | 10.1-10.4 | Final verification (tests, types, lint, e2e) |

Total: ~21 tasks, ~20 commits
