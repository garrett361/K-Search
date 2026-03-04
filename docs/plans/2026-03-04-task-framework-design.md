# Task Framework Design

Core abstraction layer for code optimization tasks. Provides composable protocols that wrap existing implementations without modification. Designed to be generic — not coupled to GPU kernels, specific languages, or evaluation infrastructure.

## Goals

1. **Task format flexibility**: Support different task types (GPU Mode, FlashInfer, SQL optimization, config tuning) with varying specs, input generators, and scoring
2. **Clean interfaces**: Separate concerns (input gen, correctness, scoring, feedback routing) into composable protocols
3. **Zero disruption**: Wrap existing `GpuModeTask` via adapters — no changes to working code
4. **Infrastructure agnostic**: Framework protocols don't assume subprocess evaluation, GPUs, or specific languages
5. **Future-ready**: Design accommodates parallelization, profiling, and configurable info routing (see `2026-03-04-task-framework-extensions.md`)

## Module Structure

```
k_search/task_framework/           # NEW: isolated submodule
├── __init__.py                    # Public exports
│
├── protocols/
│   ├── __init__.py
│   ├── results.py                 # EvaluationResult, SolutionArtifact
│   ├── input_generator.py
│   ├── reference_impl.py
│   ├── correctness.py
│   ├── scorer.py
│   ├── feedback_provider.py
│   ├── evaluator.py
│   └── task_definition.py         # Composite protocol
│
├── adapters/
│   ├── __init__.py
│   └── gpu_mode.py                # Wraps GpuModeTask
│
├── execution/
│   ├── __init__.py
│   └── sequential.py              # Current behavior
│
├── config.py                      # ExecutionConfig
├── loader.py                      # Directory-based discovery
└── types.py                       # EvalOutcome, CheckResult
```

## Core Protocols

### EvaluationResult and SolutionArtifact

Framework-owned protocols — decoupled from any concrete implementation:

```python
from typing import Any, Protocol

class EvaluationResult(Protocol):
    """Generic evaluation result — framework doesn't prescribe fields."""

    def is_success(self) -> bool: ...
    def get_metrics(self) -> dict[str, Any]: ...
    def get_log(self) -> str: ...

class SolutionArtifact(Protocol):
    """Generic solution container."""

    @property
    def name(self) -> str: ...

    @property
    def content(self) -> Any: ...  # str, dict, bytes — task-specific
```

Adapters convert between these protocols and concrete types (e.g., `EvalResult`, `Solution`).

### InputGenerator

```python
from typing import Any, Protocol

class InputGenerator(Protocol):
    """Generates task inputs from parameters."""

    def generate(self, params: dict[str, Any], seed: int) -> Any:
        """
        Args:
            params: Task-specific parameters (e.g., {"B": 2, "T": 4096})
            seed: Random seed for reproducibility

        Returns:
            Task-specific input data
        """
        ...
```

### ReferenceImpl

```python
from typing import Any, Protocol

class ReferenceImpl(Protocol):
    """Reference implementation for generating ground truth."""

    def run(self, input_data: Any) -> Any:
        """Execute reference implementation on input data."""
        ...
```

### CorrectnessChecker

```python
from typing import Any, Protocol

class CorrectnessChecker(Protocol):
    """Compares submission output against reference (if available)."""

    def check(self, output: Any, reference_output: Any) -> CheckResult:
        """
        reference_output may be None if task has no reference_impl.
        Checker decides how to handle (e.g., just verify output exists).
        """
        ...
```

### Scorer

```python
from typing import Protocol

class Scorer(Protocol):
    """Converts evaluation results to comparable scalar."""

    def score(self, result: EvaluationResult) -> float:
        """Higher is better. Return negative for failed evaluations."""
        ...
```

### FeedbackProvider

```python
from typing import Any, Protocol

class FeedbackProvider(Protocol):
    """Routes evaluation feedback to different LLM consumers."""

    def for_codegen(
        self,
        outcomes: EvalOutcome | list[EvalOutcome],
    ) -> str:
        """
        Aggregate outcomes into feedback for codegen LLM.
        Includes logs, errors, analysis data.
        """
        ...

    def for_world_model(
        self,
        outcomes: EvalOutcome | list[EvalOutcome],
    ) -> list[dict[str, Any]]:
        """
        Format outcomes for world model tree.
        Numeric metrics only — no verbose logs.
        Returns one dict per outcome.
        """
        ...
```

### Evaluator

```python
from typing import Any, Protocol

class Evaluator(Protocol):
    """Executes a solution and produces evaluation results."""

    def evaluate(
        self,
        solution: SolutionArtifact,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,  # Task-specific (device_id, etc.)
    ) -> EvaluationResult:
        ...
```

Note: Async evaluation (`evaluate_async`) is covered in the extensions doc for pipelining support.

### TaskDefinition (Composite)

```python
from typing import Any, Protocol

class TaskDefinition(Protocol):
    """Complete task definition."""

    # Identity
    name: str

    # Required components
    input_generator: InputGenerator
    correctness_checker: CorrectnessChecker
    scorer: Scorer
    feedback_provider: FeedbackProvider

    # Optional components
    reference_impl: ReferenceImpl | None  # None if correctness doesn't need reference
    # analyzer: Analyzer | None  # See extensions doc

    # Prompt generation
    def get_prompt_text(self, context: dict[str, Any] | None = None) -> str:
        """
        Task description for LLM.
        Context is task-specific (e.g., {"language": "triton"} for GPU tasks).
        """
        ...

    def get_test_cases(self) -> list[dict[str, Any]]:
        """Parameter sets for evaluation."""
        ...
```

## Types

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CheckResult:
    """Result of correctness check."""
    passed: bool
    message: str = ""
    criteria: dict[str, Any] | None = None  # rtol, atol, etc. if relevant

@dataclass
class EvalOutcome:
    """Complete result of evaluating a solution."""
    solution: SolutionArtifact
    result: EvaluationResult
    # analysis: AnalysisResult | None = None  # See extensions doc
```

## Configuration

```python
from dataclasses import dataclass, field

@dataclass
class ExecutionConfig:
    """How to run evaluations."""
    timeout_secs: int = 60
    # Parallel execution config in extensions doc
```

Note: Implementation-specific config (subprocess limits, etc.) belongs in adapters, not the core framework.

## GPU Mode Adapter

Wraps existing `GpuModeTask` to implement `TaskDefinition`. Adapter handles conversion between framework protocols and concrete types:

```python
from k_search.tasks.gpu_mode_task import GpuModeTask
from k_search.tasks.task_base import EvalResult, Solution

class GpuModeAdapter:
    """Adapts GpuModeTask to TaskDefinition protocol."""

    def __init__(self, task: GpuModeTask):
        self._task = task
        self.name = task.name
        self.input_generator = _GpuModeInputGenerator(task)
        self.reference_impl = _GpuModeReferenceImpl(task)
        self.correctness_checker = _GpuModeCorrectnessChecker(task)
        self.scorer = _GpuModeScorer()
        self.feedback_provider = _GpuModeFeedbackProvider()

    def get_prompt_text(self, context: dict[str, Any] | None = None) -> str:
        language = (context or {}).get("language", "triton")
        return self._task.get_definition_text(language)

    def get_test_cases(self) -> list[dict[str, Any]]:
        # Extract from spec or hardcode per task
        ...


class _GpuModeInputGenerator:
    """Delegates to task's reference.py generate_input()."""

    def __init__(self, task: GpuModeTask):
        self._task = task
        self._generate_fn = self._load_generate_input()

    def generate(self, params: dict[str, Any], seed: int) -> Any:
        return self._generate_fn(**params, seed=seed)


class _GpuModeCorrectnessChecker:
    """Delegates to task's reference.py check_implementation()."""

    def check(self, output: Any, reference_output: Any) -> CheckResult:
        # Wrap existing check_implementation
        # Put rtol/atol in criteria dict
        ...


class _GpuModeScorer:
    """Uses inverse latency as score (current behavior)."""

    def score(self, result: EvaluationResult) -> float:
        if not result.is_success():
            return -1.0
        metrics = result.get_metrics()
        latency = metrics.get("latency_ms")
        if latency and latency > 0:
            return 1.0 / latency
        return -1.0


class _GpuModeFeedbackProvider:
    """Routes feedback per docs/llm-info-routing.md."""

    def for_codegen(
        self,
        outcomes: EvalOutcome | list[EvalOutcome],
    ) -> str:
        if isinstance(outcomes, EvalOutcome):
            outcomes = [outcomes]
        # Include full logs for debugging
        return "\n\n".join(o.result.get_log() for o in outcomes)

    def for_world_model(
        self,
        outcomes: EvalOutcome | list[EvalOutcome],
    ) -> list[dict[str, Any]]:
        if isinstance(outcomes, EvalOutcome):
            outcomes = [outcomes]
        # Numeric metrics only
        return [o.result.get_metrics() for o in outcomes]


class _GpuModeEvaluationResult:
    """Wraps EvalResult to implement EvaluationResult protocol."""

    def __init__(self, eval_result: EvalResult):
        self._inner = eval_result

    def is_success(self) -> bool:
        return self._inner.is_passed()

    def get_metrics(self) -> dict[str, Any]:
        return self._inner.to_dict(include_log_excerpt=False)

    def get_log(self) -> str:
        return self._inner.log_excerpt


class _GpuModeSolutionArtifact:
    """Wraps Solution to implement SolutionArtifact protocol."""

    def __init__(self, solution: Solution):
        self._inner = solution

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def content(self) -> Any:
        entry = self._inner.get_entry_source()
        return entry.content if entry else ""
```

## Directory-Based Loader

```python
from pathlib import Path

def load_task_definition(task_dir: Path) -> TaskDefinition:
    """
    Load a TaskDefinition from a directory.

    Expected structure (GPU Mode):
        task_dir/
        ├── spec.py        # SPEC_TEXT_TRITON, SPEC_TEXT_CUDA
        ├── reference.py   # generate_input(), ref_kernel(), check_implementation()
        ├── submission.py  # Baseline custom_kernel()
        └── task.py        # Type definitions (input_t, output_t)

    Other task types may have different structures — use appropriate adapter.
    """
    from k_search.tasks.gpu_mode_task import GpuModeTask

    task = GpuModeTask(task_dir=task_dir)
    return GpuModeAdapter(task)
```

## Integration Points

When ready to integrate with generators, these files change:

| File | Change | Complexity |
|------|--------|------------|
| `kernel_generator_world_model.py` | Accept `TaskDefinition` instead of `Task` | Low |
| `world_model_manager.py` | Use `feedback_provider.for_world_model()` | Low |
| `gpu_mode_task.py` | No changes (wrapped by adapter) | None |

## Design Decisions

1. **Protocols over ABCs**: Structural typing allows duck-typed compatibility without inheritance
2. **Framework-owned result protocols**: `EvaluationResult` and `SolutionArtifact` decouple framework from concrete types
3. **Adapter pattern**: Wraps existing code without modification; handles type conversion
4. **Evaluation external to TaskDefinition**: TaskDefinition defines *what*, execution layer defines *how*
5. **FeedbackProvider handles routing**: Single place for codegen vs world model info routing logic
6. **Generic context dicts**: `get_prompt_text(context)` and `evaluate(..., context)` allow task-specific params without framework changes
7. **Criteria dict on CheckResult**: Keeps correctness checking generic (rtol/atol for tensors, other criteria for other tasks)

## Validation Checklist

Before implementation, verify:

- [ ] `GpuModeAdapter` can implement all `TaskDefinition` methods using existing `GpuModeTask`
- [ ] `_GpuModeEvaluationResult` correctly wraps `EvalResult`
- [ ] `FeedbackProvider.for_world_model()` output matches current behavior
- [ ] `FeedbackProvider.for_codegen()` output matches current `_last_round_trace_logs_for_prompt`
- [ ] Existing tests pass when using adapter-wrapped tasks

## References

- Current Task protocol: `k_search/tasks/task_base.py:289-322`
- GPU Mode task: `k_search/tasks/gpu_mode_task.py`
- Info routing: `docs/llm-info-routing.md`
- Per-task structure: `k_search/tasks/gpu_mode/causal_conv1d/`
- Future extensions: `docs/plans/2026-03-04-task-framework-extensions.md`
