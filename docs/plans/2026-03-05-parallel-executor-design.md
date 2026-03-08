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
┌──────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Proposer + Selector  │───▶│   LLM stage     │───▶│   Eval stage    │───▶ update()
│ propose() → select() │    │ (queue_depth=4) │    │ (queue_depth=2) │
└──────────────────────┘    └─────────────────┘    └─────────────────┘
```

- **Proposer**: Calls `world_model.propose()` to fill frontier, `select()` to pick nodes
- **LLM stage**: Generates implementations (bounded queue)
- **Eval stage**: Runs GPU benchmarks (bounded queue)
- **Update**: Applies results to tree, calls `world_model.update()`

Backpressure via bounded queues — when eval is slow, impl queue fills, LLM stage blocks on put, proposer naturally pauses.

## Protocols

### WorldModel Protocol (sync)

```python
class WorldModel(Protocol):
    def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Generate new frontier nodes. Returns empty list if nothing to propose."""
        ...

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Select frontier nodes to pursue. Returns empty list if none available."""
        ...

    def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """Update tree after cycle completes."""
        ...
```

### AsyncWorldModel Protocol

```python
class AsyncWorldModel(Protocol):
    async def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]: ...
    async def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]: ...
    async def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None: ...
```

### Sync-to-Async Wrapper

```python
class AsyncWorldModelWrapper:
    """Wrap sync WorldModel for use with PipelineExecutor."""

    def __init__(self, sync_model: WorldModel):
        self._sync = sync_model

    async def propose(self, tree, context=None) -> list[Node]:
        return await asyncio.to_thread(self._sync.propose, tree, context)

    async def select(self, tree, context=None) -> list[Node]:
        return await asyncio.to_thread(self._sync.select, tree, context)

    async def update(self, tree, context=None) -> None:
        await asyncio.to_thread(self._sync.update, tree, context)
```

### Executor Protocol

```python
class Executor(Protocol):
    # TBD - signature determined during implementation
    ...
```

## Node Status Transitions

```
open → in_progress → closed
```

| Status | Meaning |
|--------|---------|
| `open` | Available for selection |
| `in_progress` | In pipeline (selected, LLM generating, or evaluating) |
| `closed` | Done — has attached `Cycle` |

World model's `select()` only considers `open` nodes. Executor marks `in_progress` immediately upon selection.

## Configuration

```python
@dataclass
class PipelineConfig:
    llm_queue_depth: int = 4    # max nodes in LLM stage
    eval_queue_depth: int = 2   # max impls in eval stage
    # Additional config TBD (frontier_target, etc.)
```

Queue depth controls buffering at each stage. Tune ratio to keep eval saturated.

## Pipeline Internals (Schematic)

> **Note:** This is a schematic example illustrating the general shape. Actual implementation details will likely differ.

```python
class PipelineExecutor:
    async def _run_async(self, tree, max_rounds):
        action_queue = asyncio.Queue(maxsize=self.config.llm_queue_depth)
        impl_queue = asyncio.Queue(maxsize=self.config.eval_queue_depth)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._proposer(action_queue, max_rounds))
            tg.create_task(self._llm_worker(action_queue, impl_queue))
            tg.create_task(self._eval_worker(impl_queue))

        return ...  # TBD

    async def _proposer(self, action_queue, max_rounds):
        while self.rounds_completed < max_rounds:
            frontier = self.tree.get_frontier()

            # Refill frontier if needed
            if len(frontier) < self.config.frontier_target:
                # TODO: error handling
                await self.world_model.propose(self.tree, context=None)
                frontier = self.tree.get_frontier()

            if not frontier:
                break  # nothing left to explore

            # TODO: error handling
            nodes = await self.world_model.select(self.tree, context=None)
            for node in nodes:
                node.status = "in_progress"
                await action_queue.put(node)

    async def _llm_worker(self, action_queue, impl_queue):
        while True:
            node = await action_queue.get()
            # TODO: error handling
            prompt = build_prompt(self.tree, node)
            code = await self.llm(prompt)
            impl = create_impl(code, node)
            await impl_queue.put((node, impl))

    async def _eval_worker(self, impl_queue):
        while True:
            node, impl = await impl_queue.get()
            # TODO: error handling
            # Eval stays sync (GPU work), wrap in to_thread
            result = await asyncio.to_thread(self.evaluator.evaluate, impl)
            round_ = Round(impl=impl, result=result, ...)
            node.cycle = Cycle(rounds=[round_])
            node.status = "closed"
            context = {"completed_node": node, "round": round_}
            await self.world_model.update(self.tree, context)
            self.rounds_completed += 1
```

## Module Structure

```
k_search/modular/
├── protocols/
│   ├── world_model.py      # WorldModel, AsyncWorldModel (update for list[Node])
│   └── executor.py         # Executor protocol (new)
├── world_models/
│   ├── __init__.py
│   └── llm.py              # LLMWorldModel + AsyncWorldModelWrapper
├── executors/
│   ├── __init__.py
│   ├── sequential.py       # SequentialExecutor
│   └── pipeline.py         # PipelineExecutor, PipelineConfig
└── ...
```

## Implementation Stages

### Stage 1: World Model Foundation — ✅ DONE

- `Tree`, `Node`, `Action`, `Cycle` dataclasses
- `WorldModel` protocol (needs update: `propose/select -> list[Node]`)
- Tool infrastructure (`tools.py`, `apply_tool_call`)
- `StateFormatter` + `DefaultFormatter`

### Stage 2: LLMWorldModel Implementation

- Implement `LLMWorldModel` class against updated protocol
- `propose()` → LLM with forced `insert_node` tool calls → returns `list[Node]`
- `select()` → deterministic highest-score from frontier → returns `list[Node]`
- `update()` → LLM with optional tool calls for tree refinement
- `AsyncWorldModelWrapper` in same file

### Stage 3: SequentialExecutor

- `Executor` protocol
- `SequentialExecutor` - tree-aware loop replacing `run_search()`
- Integrates WorldModel with propose/select/update cycle
- Validates protocol works end-to-end

### Stage 4: PipelineExecutor

- `PipelineExecutor` with async internals
- `PipelineConfig` (queue depths)
- Bounded queues, backpressure, concurrent LLM/eval

## Deferred

- Error handling / retry logic - let exceptions propagate for MVP
- ExecutorResult definition - define when we know what we need
- Early stopping - stop when target score reached
- Metrics - `eval_utilization`, queue gauges (add when pipeline works)

## References

- LLM World Model design: `2026-03-06-llm-world-model-design.md`
- Tree data model: `2026-03-05-tree-data-model-design.md`
- Task framework: `2026-03-04-task-framework-design.md`
