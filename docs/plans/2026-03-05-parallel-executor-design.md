# Parallel Executor Design

Pipeline executor for maximizing throughput by overlapping LLM generation with GPU evaluation.

## Problem

Current search loop is strictly sequential:

```
prompt → LLM (5-30s) → eval (3-10s) → update tree → repeat
```

When pursuing multiple actions from the world model, GPU sits idle during LLM calls. Pipeline execution overlaps these stages.

## Overview

```
┌──────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Proposer │───▶│   LLM stage     │───▶│   Eval stage    │───▶ Update tree
└──────────┘    │ (queue_depth=4) │    │ (queue_depth=2) │
      ▲         └─────────────────┘    └─────────────────┘
      │                                        │
      └────────────────────────────────────────┘
```

- **Proposer** (internal coroutine): Calls `world_model.get_next_action()` when queue has capacity
- **LLM stage**: Generates implementations (bounded queue)
- **Eval stage**: Runs GPU benchmarks (bounded queue)
- **Update**: Applies results to tree, calls `world_model.update()`

Backpressure via bounded queues — when eval is slow, impl queue fills, LLM stage blocks on put, proposer naturally pauses.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Concurrency model | asyncio with bounded queues | Fine-grained backpressure, clean async/await |
| Queue config | Single depth int per stage | Queue size IS the throttle, no separate concurrency config |
| LLM throttling | Bounded queue | Backend (vLLM, API) handles actual scheduling |
| Eval workers | Match hardware (num GPUs) | Natural parallelism limit |
| Staleness | Ignore for MVP | Maximize throughput, tune later |
| Public API | Sync `run()` | Async is internal implementation detail |

## Action Node States

```
open → in_progress → closed
```

| Status | Meaning |
|--------|---------|
| `open` | Available for selection by executor |
| `in_progress` | In pipeline (queued, LLM generating, or evaluating) |
| `closed` | Done — check attached `Round` for pass/fail |

World model's `get_next_action()` only considers `open` actions. Executor marks `in_progress` immediately upon selection.

## Configuration

```python
@dataclass
class PipelineConfig:
    llm_queue_depth: int = 4    # max actions in LLM stage
    eval_queue_depth: int = 2   # max impls in eval stage
```

Queue depth controls buffering at each stage. Tune ratio to keep eval saturated.

## Protocols

### Executor Protocol

```python
@dataclass
class ExecutorResult:
    best_outcome: Round | None
    rounds_completed: int
    metrics: dict[str, Any]

class Executor(Protocol):
    def run(self, tree: SolutionTree, max_rounds: int) -> ExecutorResult:
        """Execute search, return best result."""
        ...
```

Sync API. Pipeline executor uses asyncio internally but hides it.

**Executor dependencies** — executors replace the loop entirely, so they need all loop infrastructure:

```python
class SequentialExecutor:
    def __init__(
        self,
        world_model: WorldModel,
        evaluator: Evaluator,
        llm: LLMCall,
        metrics_trackers: list[MetricsTracker] | None = None,
        artifact_store: ArtifactStore | None = None,
    ): ...

class PipelineExecutor:
    def __init__(
        self,
        world_model: WorldModel,
        evaluator: Evaluator,
        llm: LLMCall,
        config: PipelineConfig,
        metrics_trackers: list[MetricsTracker] | None = None,
        artifact_store: ArtifactStore | None = None,
    ): ...
```

### World Model Protocol

```python
# Sync version (Stage 1-2)
class WorldModel(Protocol):
    def get_next_action(self, tree: SolutionTree) -> ActionNode | None:
        """Return next action to try, or None if done."""
        ...

    def update(self, tree: SolutionTree, action: ActionNode, outcome: Round) -> None:
        """Incorporate result into tree."""
        ...

# Async version (Stage 3 - pipeline)
class AsyncWorldModel(Protocol):
    async def get_next_action(self, tree: SolutionTree) -> ActionNode | None: ...
    async def update(self, tree: SolutionTree, action: ActionNode, outcome: Round) -> None: ...
```

Simple single-action interface. Executor calls repeatedly to fill queue. Pipeline executor uses async variant.

> **Note:** Sync world models can be wrapped to async via `asyncio.to_thread()` for use with pipeline executor.

## Pipeline Internals

```python
class PipelineExecutor:
    def __init__(
        self,
        world_model: WorldModel,
        evaluator: Evaluator,
        llm: LLMCall,
        config: PipelineConfig,
    ): ...

    async def _run_async(self, tree, max_rounds):
        action_queue = asyncio.Queue(maxsize=self.config.llm_queue_depth)
        impl_queue = asyncio.Queue(maxsize=self.config.eval_queue_depth)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._proposer(action_queue, max_rounds))
            tg.create_task(self._llm_worker(action_queue, impl_queue))
            tg.create_task(self._eval_worker(impl_queue))

        return ExecutorResult(...)

    async def _proposer(self, action_queue, max_rounds):
        while self.rounds_completed < max_rounds:
            action = await self.world_model.get_next_action(self.tree)  # async
            if action is None:
                break
            action.status = "in_progress"
            await action_queue.put(action)  # blocks if full

    async def _llm_worker(self, action_queue, impl_queue):
        while True:
            action = await action_queue.get()
            try:
                prompt = build_prompt(self.tree, action)
                code = await self.llm(prompt)  # async LLM call
                impl = create_implementation(code, action)
                await impl_queue.put((action, impl))
            except Exception as e:
                logger.warning(f"LLM failed for action {action.id}: {e}")
                action.status = "closed"
                # TODO: retry logic

    async def _eval_worker(self, impl_queue):
        while True:
            action, impl = await impl_queue.get()
            try:
                # Eval stays sync (GPU work), wrap in to_thread
                result = await asyncio.to_thread(self.evaluator.evaluate, impl)
                outcome = Round(impl=impl, result=result)
                action.outcome = outcome
                action.status = "closed"
                await self.world_model.update(self.tree, action, outcome)  # async
                self.rounds_completed += 1
            except Exception as e:
                logger.warning(f"Eval failed for action {action.id}: {e}")
                action.status = "closed"
                # TODO: retry logic
```

## Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| `eval_utilization` | 0.0-1.0 | Are GPUs saturated? `busy_gpus / total_gpus` |
| `llm_queue_size` | gauge | Current queue depth |
| `eval_queue_size` | gauge | Current queue depth |

Primary tuning metric is `eval_utilization`. Target ~1.0 means pipeline is keeping GPUs fed.

## Termination

| Condition | Behavior |
|-----------|----------|
| `rounds_completed >= max_rounds` | Proposer stops, workers drain queues, return |
| `world_model.get_next_action() returns None` | No more actions, drain and return |

## Error Handling

MVP: try/except around each action, log, continue. Mark action as closed with error.

```python
# TODO: retry logic for transient failures
```

Hard failures (GPU OOM, API auth revoked) will fail repeatedly — observable from logs.

## Module Structure

```
k_search/modular/
├── executors/
│   ├── __init__.py
│   ├── protocol.py       # Executor protocol, ExecutorResult
│   ├── sequential.py     # SequentialExecutor
│   └── pipeline.py       # PipelineExecutor, PipelineConfig
└── ...
```

## Implementation Stages

### Stage 1: World Model Foundation

- `SolutionTree`, `SolutionNode`, `ActionNode` data structures
- `WorldModel` protocol (`get_next_action`, `update`)
- Initial world model implementation
- Test with ad-hoc loop (no formal executor yet)

### Stage 2: Executor Protocol + Sequential

- `Executor` protocol (sync `run()`)
- `SequentialExecutor` — formalizes the simple loop
- `ExecutorResult` dataclass

### Stage 3: Pipeline Executor

- Async interfaces for LLM and world model:
  ```python
  AsyncLLMCall = Callable[[str], Awaitable[str]]

  class AsyncWorldModel(Protocol):
      async def get_next_action(self, tree: SolutionTree) -> ActionNode | None: ...
      async def update(self, tree: SolutionTree, action: ActionNode, outcome: Round) -> None: ...
  ```
- `PipelineExecutor` with async internals (no `to_thread()` wrappers needed)
- `PipelineConfig` (queue depths)
- `eval_utilization` metric
- Error handling with TODO for retry

## Future Considerations

- **Staleness handling**: Tag actions with tree version, let world model decide relevance
- **Retry logic**: Configurable retry for transient LLM/eval failures
- **Early stopping**: Stop when target score reached
- **Adaptive queue depths**: Auto-tune based on observed utilization
- **Multi-GPU**: Scale eval workers to match available devices

## References

- Search V2 design: `2026-03-04-search-v2-design.md`
- Task framework: `2026-03-04-task-framework-design.md`
