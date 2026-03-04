# Task Framework Extensions

Future capabilities building on the core task framework. Each section can be implemented independently after the foundation is stable.

## Prerequisites

- Core task framework implemented (`2026-03-04-task-framework-design.md`)
- `GpuModeAdapter` validated against existing GPU Mode tasks
- Existing tests passing with adapter-wrapped tasks

## 0. Executor Protocol (Foundation for Parallel/Async)

Before implementing parallel or async evaluation, add the Executor abstraction.

### Separation of Concerns

| Component | Responsibility |
|-----------|---------------|
| **Evaluator** | WHAT: evaluation logic (input gen, run solution, check correctness, score) |
| **Executor** | HOW: execution strategy (sequential, parallel, pipelined, retry) |

### Protocol

```python
from typing import Any, Protocol

class Executor(Protocol):
    """Executes evaluations with configurable strategy."""

    def execute(
        self,
        solution: Solution,
        evaluator: Evaluator,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvalResult:
        """
        Execute evaluation of solution.

        Args:
            solution: The Solution to evaluate
            evaluator: The Evaluator that knows how to evaluate this task
            context: Execution context (worker_id, timeout, etc.)

        Returns:
            EvalResult with status, metrics, log_excerpt
        """
        ...
```

Uses existing types only:
- `Solution` from `task_base.py`
- `EvalResult` from `task_base.py`
- `Evaluator` from task_framework protocols

### Sequential Implementation

```python
class SequentialExecutor:
    """Single-threaded sequential execution."""

    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()

    def execute(
        self,
        solution: Solution,
        evaluator: Evaluator,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvalResult:
        return evaluator.evaluate(solution, context=context)
```

### Configuration

```python
@dataclass
class ExecutorConfig:
    timeout_secs: int = 60
    # Extended by ParallelConfig, PipelineConfig
```

### Integration

**V1 migration:**
```python
# Before (V1)
result = task.run_benchmark(solution=solution)

# After (with Executor)
result = executor.execute(solution, evaluator)
```

**V2 SearchOrchestrator:**
```python
class SearchOrchestrator:
    def __init__(self, evaluator: Evaluator, executor: Executor, ...):
        self.evaluator = evaluator
        self.executor = executor

    def _execute_action(self, action: ActionNode) -> EvalResult:
        solution = self._generate_code(action)
        return self.executor.execute(solution, self.evaluator)
```

### Parallel Extension

```python
class ParallelExecutor:
    def __init__(self, config: ParallelConfig):
        self.config = config  # includes worker_ids

    def execute_batch(
        self,
        solutions: list[Solution],
        evaluator: Evaluator,
    ) -> list[EvalResult]:
        # Distribute across workers
        ...
```

### Async Extension

```python
class PipelinedExecutor:
    async def execute_async(
        self,
        solution: Solution,
        evaluator: Evaluator,
    ) -> EvalResult:
        # For overlapping LLM gen + eval
        ...
```

---

## 1. Parallel Evaluation (requires §0)

Run multiple evaluations across workers simultaneously.

### Config Extension

Add `worker_ids` to execution config for parallel runs:

```python
# config.py (extension)
@dataclass
class ParallelConfig:
    worker_ids: list[int | str]     # Abstract worker identifiers (GPU ids, container names, etc.)
    timeout_secs: int = 60
```

### Patterns Supported

| Pattern | Description | Use Case |
|---------|-------------|----------|
| K variations | Same action, K code variants | Explore temperature/seed space |
| K frontier nodes | Different actions in parallel | Breadth-first tree exploration |
| Aggregate retry | Batch eval, combine failures, retry | Learn from multiple failures |

### Implementation

```python
# execution/parallel.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass
class ParallelConfig:
    worker_ids: list[int | str]
    timeout_secs: int = 60

class ParallelEvaluator:
    """Wraps Evaluator to run batch evaluations across workers."""

    def __init__(self, evaluator: Evaluator, config: ParallelConfig):
        self.evaluator = evaluator
        self.config = config

    def evaluate_batch(
        self,
        solutions: list[SolutionArtifact],
    ) -> list[EvaluationResult]:
        """
        Run up to len(worker_ids) solutions in parallel.
        Returns results in same order as input.
        """
        results: list[EvaluationResult | None] = [None] * len(solutions)

        with ProcessPoolExecutor(max_workers=len(self.config.worker_ids)) as executor:
            future_to_idx = {
                executor.submit(
                    self.evaluator.evaluate,
                    sol,
                    timeout_secs=self.config.timeout_secs,
                    context={"worker_id": self.config.worker_ids[i % len(self.config.worker_ids)]},
                ): i
                for i, sol in enumerate(solutions)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
```

### Usage Examples

**Pattern: K variations of same action**
```python
# Generator creates K variants
variations = [llm.generate(prompt, seed=i) for i in range(K)]
solutions = [task.make_solution(v) for v in variations]

# Parallel eval
outcomes = parallel_evaluator.evaluate_batch(solutions)

# Aggregate for codegen (show all K)
feedback = feedback_provider.for_codegen(outcomes)

# Best for world model (using scorer)
best = max(outcomes, key=lambda o: scorer.score(o.result))
```

**Pattern: K frontier nodes**
```python
# World model picks K diverse actions
actions = world_model.choose_top_K_actions(K)

# Generate code per action
solutions = [llm.generate(action.prompt) for action in actions]

# Parallel eval
outcomes = parallel_evaluator.evaluate_batch(solutions)

# Each action updates its tree node
wm_updates = feedback_provider.for_world_model(outcomes)
for action, update in zip(actions, wm_updates):
    world_model.attach(action.node_id, update)
```

**Pattern: Aggregate retry**
```python
# Round 1: K parallel attempts
solutions = [llm.generate(prompt) for _ in range(K)]
outcomes = parallel_evaluator.evaluate_batch(solutions)

# Aggregate all failures
feedback = feedback_provider.for_codegen(outcomes)

# Round 2: retry with combined insight
solutions = [llm.generate(prompt + feedback) for _ in range(K)]
outcomes = parallel_evaluator.evaluate_batch(solutions)
```

### World Model Changes

Add diversity-aware action selection:

```python
# world_model_manager.py (future)
def choose_top_K_actions(
    self,
    K: int,
    diversity_bonus: float = 0.2,
) -> list[ActionNode]:
    """
    Select K actions with diversity penalty for similarity.
    Prevents picking K nearly-identical actions.
    """
    selected = []
    frontier = self.get_open_frontier_nodes()

    while len(selected) < K and frontier:
        utilities = []
        for action in frontier:
            base_utility = self.compute_utility(action)
            diversity_penalty = sum(
                self._similarity(action, s) * diversity_bonus
                for s in selected
            )
            utilities.append(base_utility - diversity_penalty)

        best_idx = argmax(utilities)
        selected.append(frontier.pop(best_idx))

    return selected
```

### Files to Modify

| File | Change |
|------|--------|
| `task_framework/execution/parallel.py` | New file |
| `kernel_generator_world_model.py` | Use `ParallelEvaluator`, batch dispatch |
| `world_model_manager.py` | Add `choose_top_K_actions()` |

---

## 2. Async Pipelining

Overlap LLM generation with evaluation to maximize throughput.

### Evaluator Extension

Add `evaluate_async` to the `Evaluator` protocol. Default implementation wraps sync version in thread pool — implementers only write sync code:

```python
import asyncio
from typing import Any

class Evaluator(Protocol):
    def evaluate(
        self,
        solution: SolutionArtifact,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Sync evaluation — required."""
        ...

    async def evaluate_async(
        self,
        solution: SolutionArtifact,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Async evaluation — default wraps sync in thread pool.
        Override for true async (e.g., async subprocess I/O).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.evaluate(solution, timeout_secs=timeout_secs, context=context)
        )
```

Since evaluation often spawns a subprocess, the thread just waits on I/O — this gives concurrency without async complexity.

### Motivation

```
WITHOUT pipelining:
│ LLM 1 │ Eval 1 │ LLM 2 │ Eval 2 │ LLM 3 │ Eval 3 │
         ▲ idle    ▲ idle    ▲ idle

WITH pipelining:
│ LLM 1 │ LLM 2 │ LLM 3 │ LLM 4 │ ...
        │ Eval 1 │ Eval 2 │ Eval 3 │ ...
        ▲ overlapped
```

### Implementation

```python
# execution/pipeline.py
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

@dataclass
class PipelineConfig:
    max_pending_evals: int = 4
    max_pending_generations: int = 2

class PipelinedExecutor:
    """Overlaps LLM generation with evaluation."""

    def __init__(self, evaluator: Evaluator, config: PipelineConfig):
        self.evaluator = evaluator
        self.config = config
        self._eval_semaphore = asyncio.Semaphore(config.max_pending_evals)

    async def run_pipeline(
        self,
        generate_fn: Callable[[str], Awaitable[SolutionArtifact]],
        feedback_fn: Callable[[EvalOutcome], str],
        initial_prompt: str,
        max_rounds: int,
    ) -> AsyncIterator[EvalOutcome]:
        """
        Yields EvalOutcomes as they complete.
        Overlaps generation N+1 with evaluation N.
        """
        prompt = initial_prompt
        pending_eval: asyncio.Task | None = None

        for round_idx in range(max_rounds):
            # Await previous eval
            if pending_eval:
                outcome = await pending_eval
                yield outcome
                prompt = feedback_fn(outcome)

            # Generate next while eval runs
            solution = await generate_fn(prompt)

            # Start eval (runs in thread pool via evaluate_async default impl)
            async with self._eval_semaphore:
                pending_eval = asyncio.create_task(
                    self.evaluator.evaluate_async(solution)
                )

        # Drain final pending eval
        if pending_eval:
            yield await pending_eval
```

### Multi-Worker Pipelining

Combine pipelining with multiple workers:

```python
async def multi_worker_pipeline(
    evaluator: Evaluator,
    worker_ids: list[int | str],
    generate_fn: Callable[[str], Awaitable[SolutionArtifact]],
    ...
) -> AsyncIterator[EvalOutcome]:
    """Run independent pipelines per worker, merge results."""

    async def single_pipeline(worker_id: int | str):
        async for outcome in PipelinedExecutor(...).run_pipeline(...):
            yield outcome

    pipelines = [single_pipeline(wid) for wid in worker_ids]

    # Merge async iterators (yield from whichever completes first)
    async for outcome in merge_async_iterators(pipelines):
        yield outcome
```

### Files to Modify

| File | Change |
|------|--------|
| `task_framework/execution/pipeline.py` | New file |
| `kernel_generator_world_model.py` | Optional async entry point |

---

## 3. Analyzer Protocol and Implementations

Post-evaluation analysis beyond basic timing.

### Core Types

```python
# types.py (extension)
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AnalysisResult:
    """Result of post-evaluation analysis."""
    summary: str                              # For codegen prompts
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_artifact: bytes | None = None         # Full report (NCU, etc.)
```

### Analyzer Protocol

```python
# protocols/analyzer.py
from typing import Protocol

class Analyzer(Protocol):
    """Post-evaluation analysis (profiling, static analysis, etc.)."""

    def analyze(
        self,
        solution: SolutionArtifact,
        result: EvaluationResult,
    ) -> AnalysisResult:
        ...
```

When implemented, add to `TaskDefinition`:

```python
class TaskDefinition(Protocol):
    # ... existing fields ...
    analyzer: Analyzer | None  # Optional
```

And extend `EvalOutcome`:

```python
@dataclass
class EvalOutcome:
    solution: SolutionArtifact
    result: EvaluationResult
    analysis: AnalysisResult | None = None  # Populated when analyzer runs
```

### NCU Profiler

```python
# analyzers/ncu.py
import subprocess
from k_search.task_framework.types import AnalysisResult

class NcuAnalyzer:
    """Nsight Compute profiling."""

    def __init__(self, metrics: list[str] | None = None):
        self.metrics = metrics or [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]

    def analyze(self, solution: SolutionArtifact, result: EvaluationResult) -> AnalysisResult:
        if not result.is_success():
            return AnalysisResult(summary="Skipped (eval failed)", metrics={})

        # Run NCU
        report = self._run_ncu(solution)

        # Extract metrics
        metrics = self._parse_report(report)

        # Generate summary
        summary = self._format_summary(metrics)

        return AnalysisResult(
            summary=summary,
            metrics=metrics,
            raw_artifact=report,
        )

    def _format_summary(self, metrics: dict) -> str:
        """Concise summary for codegen prompt."""
        return (
            f"Profile: Compute {metrics.get('compute_pct', '?')}% | "
            f"Memory {metrics.get('memory_pct', '?')}% | "
            f"Occupancy {metrics.get('occupancy_pct', '?')}%"
        )
```

### Static Analyzer

```python
# analyzers/static.py
class StaticAnalyzer:
    """Static analysis of generated code."""

    def analyze(self, solution: SolutionArtifact, result: EvaluationResult) -> AnalysisResult:
        issues = []

        # solution.content is task-specific; GPU mode returns source files
        content = solution.content
        if isinstance(content, str):
            issues.extend(self._check_common_issues(content))
        elif isinstance(content, dict):
            for src in content.values():
                issues.extend(self._check_common_issues(src))

        return AnalysisResult(
            summary="\n".join(issues) if issues else "No issues found",
            metrics={"issue_count": len(issues)},
        )

    def _check_common_issues(self, code: str) -> list[str]:
        issues = []
        if "torch.cuda.synchronize()" in code:
            issues.append("Warning: explicit synchronize() may hurt performance")
        if ".cpu()" in code:
            issues.append("Warning: CPU transfer detected")
        return issues
```

### Composite Analyzer

```python
# analyzers/composite.py
class CompositeAnalyzer:
    """Run multiple analyzers and merge results."""

    def __init__(self, analyzers: list[Analyzer]):
        self.analyzers = analyzers

    def analyze(self, solution: SolutionArtifact, result: EvaluationResult) -> AnalysisResult:
        summaries = []
        all_metrics = {}

        for analyzer in self.analyzers:
            analysis = analyzer.analyze(solution, result)
            summaries.append(analysis.summary)
            all_metrics.update(analysis.metrics)

        return AnalysisResult(
            summary="\n".join(summaries),
            metrics=all_metrics,
        )
```

### Files to Add

| File | Purpose |
|------|---------|
| `task_framework/analyzers/__init__.py` | Exports |
| `task_framework/analyzers/ncu.py` | NCU profiling |
| `task_framework/analyzers/static.py` | Static analysis |
| `task_framework/analyzers/composite.py` | Combine multiple |

---

## 4. Enhanced Feedback Aggregation

Smarter aggregation of parallel evaluation results.

### Pattern Detection

```python
class SmartFeedbackProvider:
    """Detects patterns across multiple outcomes."""

    def for_codegen(
        self,
        outcomes: EvalOutcome | list[EvalOutcome],
    ) -> str:
        if isinstance(outcomes, EvalOutcome):
            return self._single_feedback(outcomes)

        # Analyze patterns
        passed = [o for o in outcomes if o.result.is_success()]
        failed = [o for o in outcomes if not o.result.is_success()]

        sections = []

        if passed:
            sections.append(self._summarize_passed(passed))

        if failed:
            # Group failures by error type
            error_groups = self._group_by_error(failed)
            sections.append(self._summarize_failures(error_groups))

        return "\n\n".join(sections)

    def _group_by_error(
        self,
        outcomes: list[EvalOutcome],
    ) -> dict[str, list[EvalOutcome]]:
        """Group outcomes by error signature."""
        groups: dict[str, list[EvalOutcome]] = {}
        for o in outcomes:
            sig = self._error_signature(o.result.get_log())
            groups.setdefault(sig, []).append(o)
        return groups

    def _summarize_failures(
        self,
        groups: dict[str, list[EvalOutcome]],
    ) -> str:
        """Summarize failure patterns."""
        lines = [f"Failures ({sum(len(g) for g in groups.values())} total):"]
        for sig, outcomes in groups.items():
            lines.append(f"  - {sig}: {len(outcomes)} occurrences")
            # Show one example
            lines.append(f"    Example: {outcomes[0].result.get_log()[:200]}...")
        return "\n".join(lines)
```

---

## 5. Configurable Output Limits

Make truncation limits configurable per-task or per-run.

### Implementation

Thread `OutputLimits` through the evaluation chain:

```python
@dataclass
class OutputLimits:
    """Implementation-specific truncation limits."""
    subprocess_output_bytes: int = 16384
    log_excerpt_chars: int = 8000

class ConfigurableEvaluator:
    def __init__(self, base_evaluator: Evaluator, limits: OutputLimits):
        self.base = base_evaluator
        self.limits = limits

    def evaluate(self, solution: SolutionArtifact, **kwargs) -> EvaluationResult:
        # Pass limits via context
        context = kwargs.pop("context", {}) or {}
        context["_output_limits"] = self.limits
        return self.base.evaluate(solution, context=context, **kwargs)
```

### Files to Modify

| File | Change |
|------|--------|
| `run_eval.py` | Accept `max_len` parameter |
| `evaluator.py` | Accept `log_excerpt_limit` parameter |
| `task_base.py` | Accept limits in `to_dict()` (already has params) |

---

## 6. Artifact Persistence

Serialize all evaluation artifacts losslessly for future analysis.

### Motivation

Current system truncates logs and discards raw artifacts. Persisting full-fidelity data enables:

- **Post-hoc analysis** — investigate why certain approaches failed
- **Summarizer component** — LLM-based pattern extraction across attempts, feed insights to world model
- **Training data** — collect successful optimization trajectories
- **Debugging** — reproduce failures without re-running

### Key Concept

An `ArtifactStore` protocol with `store()`, `retrieve()`, and `query()` methods. Each stored artifact includes:

- Solution source (full)
- Evaluation metrics
- Raw logs (untruncated)
- Profiling artifacts (NCU reports, etc.) — extensible to future profiling tools
- Tree position (parent, action taken, depth)

### Nice-to-Haves

- **W&B integration** — upload artifacts to Weights & Biases for centralized tracking, search UI, team collaboration
- **Summarizer component** — query artifacts, use LLM to extract patterns ("these 5 failures all hit the same memory issue")
- **Lineage tracking** — trace successful solutions back through their optimization path

### Design Notes

- Store should be extensible to new artifact types (profiling, static analysis, etc.)
- Query interface should support filtering by run, status, time range
- Binary artifacts (NCU reports) stored separately from JSON metadata

Detailed design deferred until core framework is stable.

---

## 7. LLM Query Mechanism

Let the LLM request additional context before acting. Currently, information flow is push-based (system decides what each LLM sees). This extension adds pull-based queries where the LLM declares what info it needs.

### Motivation

Both P_world (action selection) and P_gen (code generation) currently receive fixed context. But the optimal context varies:
- When selecting next action: might want to see top 3 implementations to understand what's working
- When debugging a failure: might want to see similar past failures
- When refining: might not need any extra context

### Design

**Two-phase prompting per round:**

```
Phase 1 (Query Request):
  Prompt: Tree summary + "Available queries: [...]. What info do you need?"
  LLM: Outputs query requests (e.g., "top_solutions(3)")

Phase 2 (Action with Context):
  Prompt: Original prompt + query results
  LLM: Takes action
```

**QueryProvider protocol:**

```python
class QueryProvider(Protocol):
    def get_available_queries(self) -> list[QuerySpec]:
        """Return specs for queries the LLM can request."""
        ...

    def execute(self, query: QueryRequest) -> QueryResult:
        """Execute a single query, return formatted result."""
        ...
```

**Core queries:**

| Query | Description |
|-------|-------------|
| `top_solutions(n, sort_by)` | Top N solutions with code + metrics |
| `solution(node_id)` | Single solution details |
| `failed_solutions(n)` | Recent failures |
| `parent(node_id)` | Parent node |
| `children(node_id)` | Child nodes |
| `path_to_root(node_id)` | Ancestry chain |

**Parsing** lives in `parsing/query_parser.py`, returns `ParseResult[list[QueryRequest]]` for retry support.

### Configuration

```python
@dataclass
class QueryConfig:
    enabled: bool = False
    available_queries: list[str] | None = None  # None = all
    max_queries_per_request: int = 5
```

### Integration

Query phase is optional — config flag to enable/disable. Can skip for simple "refine best" actions.

```
Round N:
  1. Format tree state (existing)
  2. Query phase (NEW, optional)
     - Send query menu to LLM
     - Parse requests, execute, append results
  3. Action phase (existing)
  4. Update tree (existing)
```

### Files to Add

| File | Purpose |
|------|---------|
| `protocols/query_provider.py` | QueryProvider protocol, QuerySpec, QueryRequest, QueryResult |
| `query/tree_query_provider.py` | Implementation backed by SolutionTree |
| `parsing/query_parser.py` | Parse LLM output into QueryRequest list |

---

## Implementation Order

Suggested sequence:

1. **Parallel evaluation** — Biggest throughput win, builds on core framework
2. **NCU analyzer** — Valuable feedback for optimization
3. **Async pipelining** — Overlaps with parallel, can be done in parallel
4. **Enhanced feedback aggregation** — Polish after parallel works
5. **Configurable limits** — Low priority, current defaults work
6. **LLM query mechanism** — Adds LLM agency, implement after core search loop stable

## References

- Parallel GPU search TODO: `docs/todos/parallel-gpu-search.md`
- NCU feedback TODO: `docs/todos/ncu-profile-feedback.md`
- Output limits TODO: `docs/todos/configurable-output-limits.md`
- Core framework: `docs/plans/2026-03-04-task-framework-design.md`
