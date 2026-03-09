# Async Pipeline Executor Design

Async executor for GPU kernel optimization with pipelined LLM generation and GPU evaluation.

## Problem

The sync executor (`scripts/gpu_mode_simple_linear_executor/run.py`) runs strictly sequential:

```
propose → LLM gen → GPU eval → repeat
```

GPU sits idle during LLM calls. With multiple GPUs available, only one is active at a time.

## Solution

Pipeline execution overlaps LLM generation with GPU evaluation using async queues and semaphores.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  Proposer   │────▶│  LLM Queue  │────▶│   Workers    │────▶│  GPU Eval │
│  (task)     │     │  (bounded)  │     │  (N tasks)   │     │(semaphore)│
└─────────────┘     └─────────────┘     └──────────────┘     └───────────┘
       ▲                                        │
       │            ┌─────────────┐             │
       └────────────│ Completion  │◀────────────┘
                    │   Queue     │
                    └─────────────┘
```

**Components:**
- **Proposer task**: Fills LLM queue initially, then adds one node per completion
- **LLM Queue**: `asyncio.Queue(maxsize=llm_queue_depth)` — buffers proposed nodes
- **Workers**: N concurrent async tasks, each pulls node → LLM gen → GPU eval
- **GPU Semaphore**: `asyncio.Semaphore(num_gpus)` — limits concurrent evals
- **Completion Queue**: Workers signal completion, proposer listens

## Proposal Logic

- **Initial burst** (no completed nodes yet): Generic "Write an optimized implementation" action for all
- **After completions**: LLM proposes action informed by feedback from completed nodes

The world model checks for *closed nodes with cycles* (completed evaluations), not just children, since there may be in-flight nodes that haven't finished yet.

## AsyncSimpleWorldModel

```python
class AsyncSimpleWorldModel:
    """World model for async executor - handles in-flight nodes gracefully."""

    def __init__(self, llm: AsyncLLM, action_prompt_fn: ActionPromptFn):
        self._llm = llm
        self._action_prompt_fn = action_prompt_fn

    async def propose(self, tree: Tree, context=None) -> list[Node]:
        completed = [n for n in tree._all_nodes()
                     if n.status == "closed" and n.cycle]

        if not completed:
            action_desc = "Write an optimized implementation."
        else:
            prompt = self._action_prompt_fn(tree, context)
            action_desc = await self._llm(prompt)

        parent = self._get_last_closed_node(tree) or tree.root
        node = Node(parent=parent, status="open", action=Action(title=action_desc))
        return [node]

    async def select(self, tree, context=None) -> list[Node]:
        return []  # Executor manages queue directly

    async def update(self, tree, context=None) -> None:
        pass

    def _get_last_closed_node(self, tree: Tree) -> Node | None:
        closed = [n for n in tree._all_nodes()
                  if n.status == "closed" and n.cycle]
        return max(closed, key=lambda n: int(n._id)) if closed else None
```

Key differences from `SimpleWorldModel`:
- Async methods
- Checks for *closed nodes with cycles*, not just children
- Parent is last closed node (or root)
- `select()` returns empty — executor manages queue directly

## AsyncPipelineExecutor

```python
class AsyncPipelineExecutor:
    def __init__(
        self,
        world_model: AsyncSimpleWorldModel,
        task: TaskDefinition,
        evaluator: Evaluator,
        llm: AsyncLLM,
        code_prompt_fn: CodePromptFn,
        tree: Tree,
        max_rounds: int,
        llm_queue_depth: int = 4,
        num_gpus: int = 1,
    ):
        ...

    async def run(self) -> Node | None:
        llm_queue = asyncio.Queue(maxsize=self._llm_queue_depth)
        completion_queue = asyncio.Queue()
        gpu_semaphore = asyncio.Semaphore(self._num_gpus)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._proposer(llm_queue, completion_queue))
            for _ in range(self._llm_queue_depth):
                tg.create_task(self._worker(llm_queue, completion_queue, gpu_semaphore))

        return self._tree.get_best_node()

    async def _proposer(self, llm_queue, completion_queue):
        # Initial burst
        for _ in range(self._llm_queue_depth):
            nodes = await self._world_model.propose(self._tree)
            for node in nodes:
                self._tree.add_node(node)
                await llm_queue.put(node)

        # On each completion, propose one more
        rounds_completed = self._llm_queue_depth
        while rounds_completed < self._max_rounds:
            await completion_queue.get()
            nodes = await self._world_model.propose(self._tree)
            for node in nodes:
                self._tree.add_node(node)
                await llm_queue.put(node)
            rounds_completed += 1

        # Signal workers to stop
        for _ in range(self._llm_queue_depth):
            await llm_queue.put(None)

    async def _worker(self, llm_queue, completion_queue, gpu_semaphore):
        while True:
            node = await llm_queue.get()
            if node is None:
                break  # Shutdown signal

            # LLM generation
            node.status = "in_progress"
            prompt = self._code_prompt_fn(node, self._task)
            code = await self._llm(prompt)
            impl = self._task.create_impl(code)

            # GPU eval (semaphore-bounded)
            async with gpu_semaphore:
                result = await asyncio.to_thread(self._evaluator.evaluate, impl)

            # Record result
            score = self._task.scorer.score(result)
            node.cycle = Cycle(rounds=[Round(impl=impl, result=result, ...)])
            node.status = "closed"

            await completion_queue.put(node)
```

## Configuration

CLI args (in addition to existing sync args):
- `--llm-queue-depth` (default 4): Buffer size for proposed nodes
- `--num-gpus` (default 1): Concurrent GPU evaluations

## Error Handling

Fail fast — any error propagates and cancels the pipeline via TaskGroup exception handling.

## File Structure

Single file at `scripts/gpu_mode_async_pipeline_executor/run.py`:

```
run.py
├── Imports (asyncio, openai, argparse, etc.)
├── Helper functions (from sync version)
│   ├── _extract_error_hint()
│   ├── _truncate_log()
│   ├── _extract_code_block()
│   ├── _get_last_evaluated_node()
│   ├── _analyze_failure() / _analyze_failure_for_action()
├── Prompt templates (ACTION_PROMPT_TEMPLATE, etc.)
├── create_action_prompt_fn()
├── create_code_prompt_fn()
├── AsyncSimpleWorldModel
├── AsyncPipelineExecutor
├── create_async_llm_call() — AsyncOpenAI wrapper
├── main()
```

## References

- Sync version: `scripts/gpu_mode_simple_linear_executor/run.py`
- Parallel executor design: `docs/plans/2026-03-05-parallel-executor-design.md`
- Simple world model: `k_search/modular/world_models/simple.py`
