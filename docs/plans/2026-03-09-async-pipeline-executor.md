# Async Pipeline Executor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an async pipelined executor that overlaps LLM generation with GPU evaluation for improved throughput.

**Architecture:** Producer-consumer pattern with bounded LLM queue, GPU semaphore for backpressure, and completion queue for proposer feedback. Single file implementation mirroring the sync version structure.

**Tech Stack:** asyncio, openai.AsyncOpenAI, asyncio.Queue, asyncio.Semaphore, asyncio.TaskGroup

---

### Task 1: Create Directory and File Scaffold

**Files:**
- Create: `scripts/gpu_mode_async_pipeline_executor/run.py`

**Step 1: Create directory**

Run: `mkdir -p scripts/gpu_mode_async_pipeline_executor`

**Step 2: Create scaffold with imports and helpers copied from sync version**

```python
#!/usr/bin/env python3
"""Async pipeline executor - overlaps LLM generation with GPU evaluation."""

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai

from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.modular.logging import prompt_color, response_color
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.round import Round
from k_search.modular.world.tree import Tree
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

for noisy_logger in ("httpcore", "httpx", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _extract_error_hint(log: str) -> str:
    """Extract first meaningful error line from log."""
    if not log:
        return ""
    for line in log.strip().split("\n"):
        line = line.strip()
        if any(kw in line.lower() for kw in ("error", "failed", "exception", "assert")):
            return line
    return ""


def _truncate_log(log: str, max_lines: int = 30) -> str:
    """Keep only the last N lines of an error log."""
    if not log:
        return ""
    lines = log.strip().split("\n")
    if len(lines) <= max_lines:
        return log
    return f"[...truncated {len(lines) - max_lines} lines...]\n" + "\n".join(
        lines[-max_lines:]
    )


def _extract_code_block(llm_response: str) -> str:
    """Extract just the code from an LLM response, stripping prose."""
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response


def _get_last_evaluated_node(tree: Tree) -> Node | None:
    """Return most recently evaluated node (highest ID, closed, has cycle)."""
    evaluated = [
        n
        for n in tree._all_nodes()
        if n.status == "closed" and n.cycle and n.cycle.rounds
    ]
    if not evaluated:
        return None
    return max(evaluated, key=lambda n: int(n._id))
```

**Step 3: Verify file runs without import errors**

Run: `cd /proj/data-eng/goon/flim/verl-experiments-k-search/K-Search && python -c "import scripts.gpu_mode_async_pipeline_executor.run"`

Expected: No output (imports succeed)

---

### Task 2: Add Prompt Templates and Prompt Functions

**Files:**
- Modify: `scripts/gpu_mode_async_pipeline_executor/run.py`

**Step 1: Add prompt templates after helpers**

```python
ACTION_PROMPT_TEMPLATE = """You are proposing the next optimization action for a GPU kernel.

## Task Specification
{task_spec}
{feedback_section}{last_round_section}
## Your Job
Propose ONE specific optimization action to try next.

Rules:
- Single-iteration implementable, SMALL change
- One concrete tweak (tiling OR memory OR scheduling - not multiple)
- Be specific (e.g., "use shared memory for input tile" not "optimize memory")

Respond with only the action title (one line, no explanation)."""

FAILURE_ANALYSIS_PROMPT = """Review this Triton/CUDA code that failed. Analyze the ENTIRE code for ALL potential problems, not just the error shown.

Error:
```
{error_log}
```

Code:
```python
{failed_code}
```

List ALL issues and how to fix each. Be concise - no code snippets needed."""

ACTION_FAILURE_ANALYSIS_PROMPT = """The previous optimization attempt failed with this error:
```
{error_log}
```

The attempted action was: {action}

In 2-3 sentences: What went wrong and what should the next action avoid or do differently?"""

AsyncLLMCallable = Callable[[str], Any]  # Returns Awaitable[str]


async def _analyze_failure(llm: AsyncLLMCallable, error_log: str, failed_code: str) -> str:
    """Ask LLM to analyze a failure and suggest fixes for code generation."""
    prompt = FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-2000:],
        failed_code=failed_code[-3000:],
    )
    return await llm(prompt)


async def _analyze_failure_for_action(llm: AsyncLLMCallable, error_log: str, action: str) -> str:
    """Ask LLM to explain failure and guide next action selection."""
    prompt = ACTION_FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-1000:],
        action=action,
    )
    return await llm(prompt)
```

**Step 2: Add create_action_prompt_fn (async version)**

```python
def create_action_prompt_fn(
    task_def: GpuModeTriMulTaskDefinition,
    llm: AsyncLLMCallable,
    *,
    analyze_failures: bool = False,
):
    """Create GPU mode specific action prompt function.

    Uses feedback from best round to inform next action.
    If analyze_failures=True, also uses LLM to analyze failures.
    """

    async def action_prompt_fn(tree: Tree, context: dict[str, Any] | None) -> str:
        task_spec = task_def.get_prompt_text()

        feedback_section = ""
        best_node = tree.get_best_node()
        if best_node and best_node.cycle:
            best_round = best_node.cycle.best_round
            if best_round:
                feedback = task_def.feedback_provider.for_codegen(best_round)
                feedback_section = f"\n## Previous Best Result\n{feedback}\n"

        last_round_section = ""
        if analyze_failures:
            last_node = _get_last_evaluated_node(tree)
            if last_node and last_node.cycle and last_node.cycle.rounds:
                last_round = last_node.cycle.rounds[-1]
                if not last_round.result.succeeded():
                    action = last_node.action.title if last_node.action else "initial"
                    error_log = last_round.result.get_log()
                    analysis = await _analyze_failure_for_action(llm, error_log, action)
                    last_round_section = f"\n## Last Round (FAILED)\nAction: {action}\nLesson: {analysis}\n"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            task_spec=task_spec,
            feedback_section=feedback_section,
            last_round_section=last_round_section,
        )
        logger.debug(
            prompt_color(
                f"[ACTION_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
            )
        )
        return prompt

    return action_prompt_fn
```

**Step 3: Add create_code_prompt_fn (sync, same as original)**

```python
def create_code_prompt_fn(
    task_def: GpuModeTriMulTaskDefinition,
    tree: Tree,
    llm: AsyncLLMCallable,
    *,
    analyze_failures: bool = False,
    v1_feedback: bool = False,
):
    """Create GPU mode specific code generation prompt function.

    Round 0 (no action): direct code generation from task prompt.
    Round 1+ (with action): generate code implementing the specific action.

    If v1_feedback=True, includes last round and best solution feedback.
    If analyze_failures=True, includes LLM-generated failure analysis.
    """

    def _format_round_summary(r) -> str:
        """Format V1-style round summary."""
        metrics = r.result.get_metrics()
        status = "passed" if r.result.succeeded() else "failed"
        lines = [f"Status: {status}"]
        if latency := metrics.get("latency_ms"):
            lines.append(f"Latency: {latency:.2f}ms")
        if speedup := metrics.get("speedup_factor"):
            lines.append(f"Speedup: {speedup:.2f}x")
        lines.append(f"Score: {r.score:.4f}")
        return " | ".join(lines)

    def code_prompt_fn(node: Node, task: TaskDefinition) -> str:
        prompt = task_def.get_prompt_text()

        if node.action:
            prompt += f"\n\nAction: {node.action.title}"

        last_node = (
            _get_last_evaluated_node(tree) if v1_feedback or analyze_failures else None
        )

        if v1_feedback:
            if last_node and last_node.cycle and last_node.cycle.rounds:
                last_round = last_node.cycle.rounds[-1]
                logs = _truncate_log(last_round.result.get_log(), max_lines=30)
                code = _extract_code_block(last_round.llm_response)
                summary = _format_round_summary(last_round)

                prompt += f"\n\nEvaluation Output (from your previous attempt):\n{logs}"
                prompt += f"\n\nYour Previous Code:\n```python\n{code}\n```"
                prompt += f"\n\nPrevious Round Summary:\n{summary}"

            best_node = tree.get_best_node()
            if best_node and best_node.cycle and best_node.cycle.best_round:
                best_round = best_node.cycle.best_round
                if last_node is None or best_node._id != last_node._id:
                    best_summary = _format_round_summary(best_round)
                    best_code = best_round.llm_response
                    prompt += f"\n\nBest Successful Solution So Far:\n{best_summary}\n\nCode:\n{best_code}"

        prompt += "\n\nGenerate the corrected and optimized implementation:"
        logger.debug(
            prompt_color(
                f"[CODE_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
            )
        )
        return prompt

    return code_prompt_fn
```

**Step 4: Verify syntax**

Run: `python -m py_compile scripts/gpu_mode_async_pipeline_executor/run.py`

Expected: No output (syntax valid)

---

### Task 3: Implement AsyncSimpleWorldModel

**Files:**
- Modify: `scripts/gpu_mode_async_pipeline_executor/run.py`

**Step 1: Add AsyncSimpleWorldModel class**

```python
INITIAL_ACTION = "Write an optimized implementation."

AsyncActionPromptFn = Callable[[Tree, dict[str, Any] | None], Any]  # Returns Awaitable[str]


class AsyncSimpleWorldModel:
    """World model for async executor - handles in-flight nodes gracefully.

    Key difference from SimpleWorldModel: checks for *closed nodes with cycles*
    (completed evaluations), not just children, since there may be in-flight
    nodes that haven't finished yet.
    """

    def __init__(self, llm: AsyncLLMCallable, action_prompt_fn: AsyncActionPromptFn):
        self._llm = llm
        self._action_prompt_fn = action_prompt_fn

    async def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Generate action via LLM if we have completed results, else generic action."""
        completed = [
            n for n in tree._all_nodes()
            if n.status == "closed" and n.cycle
        ]

        if not completed:
            logger.debug("No completed nodes yet, using initial action")
            action_desc = INITIAL_ACTION
        else:
            logger.debug("Completed nodes exist, requesting action from LLM")
            prompt = await self._action_prompt_fn(tree, context)
            raw_response = await self._llm(prompt)
            logger.debug(response_color(f"[ACTION_RESPONSE] {raw_response.strip()}"))
            action_desc = raw_response.strip()

        parent = self._get_last_closed_node(tree) or tree.root
        action = Action(title=action_desc)
        node = Node(parent=parent, status="open", action=action)
        logger.debug(
            "Created node: action=%r, parent_id=%s, status=%s",
            action.title[:50],
            parent._id,
            node.status,
        )
        logger.info(f"Proposed action: {action.title[:50]}...")
        return [node]

    async def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """No-op - executor manages queue directly."""
        return []

    async def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """No-op for simple model."""
        pass

    def _get_last_closed_node(self, tree: Tree) -> Node | None:
        """Get most recently closed node (completed evaluation)."""
        closed = [
            n for n in tree._all_nodes()
            if n.status == "closed" and n.cycle
        ]
        if not closed:
            return None
        return max(closed, key=lambda n: int(n._id))
```

**Step 2: Verify syntax**

Run: `python -m py_compile scripts/gpu_mode_async_pipeline_executor/run.py`

Expected: No output (syntax valid)

---

### Task 4: Implement AsyncPipelineExecutor

**Files:**
- Modify: `scripts/gpu_mode_async_pipeline_executor/run.py`

**Step 1: Add AsyncPipelineExecutor class**

```python
CodePromptFn = Callable[[Node, TaskDefinition], str]


class AsyncPipelineExecutor:
    """Async pipeline executor - overlaps LLM generation with GPU evaluation.

    Architecture:
        Proposer -> [LLM Queue] -> Workers -> (GPU Semaphore) -> Eval
                                     |
                        [Completion Queue] <- signal completion
                                     |
                           Proposer listens
    """

    def __init__(
        self,
        world_model: AsyncSimpleWorldModel,
        task: TaskDefinition,
        evaluator: Any,  # Evaluator protocol
        llm: AsyncLLMCallable,
        code_prompt_fn: CodePromptFn,
        tree: Tree,
        max_rounds: int,
        llm_queue_depth: int = 4,
        num_gpus: int = 1,
    ):
        self._world_model = world_model
        self._task = task
        self._evaluator = evaluator
        self._llm = llm
        self._code_prompt_fn = code_prompt_fn
        self._tree = tree
        self._max_rounds = max_rounds
        self._llm_queue_depth = llm_queue_depth
        self._num_gpus = num_gpus
        self._rounds_completed = 0

    async def run(self) -> Node | None:
        """Execute async pipeline search."""
        llm_queue: asyncio.Queue[Node | None] = asyncio.Queue(maxsize=self._llm_queue_depth)
        completion_queue: asyncio.Queue[Node] = asyncio.Queue()
        gpu_semaphore = asyncio.Semaphore(self._num_gpus)

        logger.info(
            f"Starting async pipeline: max_rounds={self._max_rounds}, "
            f"llm_queue_depth={self._llm_queue_depth}, num_gpus={self._num_gpus}"
        )

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._proposer(llm_queue, completion_queue))
            for worker_id in range(self._llm_queue_depth):
                tg.create_task(self._worker(worker_id, llm_queue, completion_queue, gpu_semaphore))

        logger.info(f"Pipeline complete. Rounds completed: {self._rounds_completed}")
        return self._tree.get_best_node()

    async def _proposer(
        self,
        llm_queue: asyncio.Queue[Node | None],
        completion_queue: asyncio.Queue[Node],
    ) -> None:
        """Manage proposal flow: initial burst, then one per completion."""
        # Initial burst to fill queue
        initial_count = min(self._llm_queue_depth, self._max_rounds)
        logger.info(f"[PROPOSER] Initial burst: proposing {initial_count} nodes")
        for i in range(initial_count):
            nodes = await self._world_model.propose(self._tree)
            for node in nodes:
                self._tree.add_node(node)
                await llm_queue.put(node)
                logger.debug(f"[PROPOSER] Queued initial node {i + 1}/{initial_count}")

        # On each completion, propose one more (up to max_rounds)
        rounds_proposed = initial_count
        while rounds_proposed < self._max_rounds:
            completed_node = await completion_queue.get()
            self._rounds_completed += 1
            logger.debug(
                f"[PROPOSER] Completion received ({self._rounds_completed}/{self._max_rounds}), "
                f"proposing next"
            )

            nodes = await self._world_model.propose(self._tree)
            for node in nodes:
                self._tree.add_node(node)
                await llm_queue.put(node)
            rounds_proposed += 1

        # Drain remaining completions
        remaining = initial_count - (self._max_rounds - self._rounds_completed)
        while self._rounds_completed < self._max_rounds:
            await completion_queue.get()
            self._rounds_completed += 1
            logger.debug(f"[PROPOSER] Draining completion ({self._rounds_completed}/{self._max_rounds})")

        # Signal workers to stop
        logger.info(f"[PROPOSER] Sending {self._llm_queue_depth} shutdown sentinels")
        for _ in range(self._llm_queue_depth):
            await llm_queue.put(None)

    async def _worker(
        self,
        worker_id: int,
        llm_queue: asyncio.Queue[Node | None],
        completion_queue: asyncio.Queue[Node],
        gpu_semaphore: asyncio.Semaphore,
    ) -> None:
        """Process nodes: LLM generation -> GPU evaluation."""
        while True:
            node = await llm_queue.get()
            if node is None:
                logger.debug(f"[WORKER-{worker_id}] Received shutdown sentinel")
                break

            logger.debug(f"[WORKER-{worker_id}] Processing node: {node.action.title[:30] if node.action else 'no-action'}...")
            node.status = "in_progress"

            # LLM generation (async, no resource limit)
            prompt = self._code_prompt_fn(node, self._task)
            code = await self._llm(prompt)
            logger.debug(
                response_color(
                    f"[CODE_RESPONSE] ({len(code)} chars, ~{len(code) // 4} toks)"
                )
            )

            impl = self._task.create_impl(code)

            # GPU evaluation (semaphore-bounded)
            async with gpu_semaphore:
                logger.debug(f"[WORKER-{worker_id}] Acquired GPU semaphore, evaluating...")
                result = await asyncio.to_thread(self._evaluator.evaluate, impl, context={})

            score = self._task.scorer.score(result)
            logger.debug(f"[WORKER-{worker_id}] Eval complete: score={score:.4f}, success={result.succeeded()}")

            # Record result
            round_ = Round(
                impl=impl,
                result=result,
                prompt=prompt,
                llm_response=code,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(code) // 4,
                duration_secs=0.0,
                score=score,
            )
            node.cycle = Cycle(rounds=[round_])
            node.status = "closed"

            logger.info(
                f"[WORKER-{worker_id}] Round complete: score={score:.4f}, success={result.succeeded()}"
            )

            if not result.succeeded():
                log_excerpt = result.get_log()
                if log_excerpt:
                    logger.debug(f"[WORKER-{worker_id}] Error:\n{_truncate_log(log_excerpt, 10)}")

            await completion_queue.put(node)
```

**Step 2: Verify syntax**

Run: `python -m py_compile scripts/gpu_mode_async_pipeline_executor/run.py`

Expected: No output (syntax valid)

---

### Task 5: Add Main Entry Point

**Files:**
- Modify: `scripts/gpu_mode_async_pipeline_executor/run.py`

**Step 1: Add async LLM call factory and main function**

```python
def create_async_llm_call(client: openai.AsyncOpenAI, model_name: str) -> AsyncLLMCallable:
    """Create async LLM callable."""

    async def llm_call(prompt: str) -> str:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    return llm_call


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("API key required (--api-key or LLM_API_KEY)")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent.parent
    task_dir = repo_root / "k_search" / "tasks" / "gpu_mode" / args.task
    if not task_dir.exists():
        logger.error(f"Task not found: {task_dir}")
        sys.exit(1)

    logger.info(f"Loading task: {args.task}")
    gpu_task = GpuModeTriMulTask(task_dir=task_dir)
    task_def = GpuModeTriMulTaskDefinition(gpu_task, language=args.language)
    evaluator = GpuModeEvaluator(gpu_task)

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if (rits_api_key := os.getenv("RITS_API_KEY")) is not None:
        client_kwargs["default_headers"] = {"RITS_API_KEY": rits_api_key}

    client = openai.AsyncOpenAI(**client_kwargs)
    llm = create_async_llm_call(client, args.model_name)

    tree = Tree(root=Node(status="closed"))

    action_prompt_fn = create_action_prompt_fn(
        task_def, llm, analyze_failures=args.analyze_action_failures
    )
    code_prompt_fn = create_code_prompt_fn(
        task_def,
        tree,
        llm,
        analyze_failures=args.analyze_code_failures,
        v1_feedback=args.v1_feedback,
    )

    world_model = AsyncSimpleWorldModel(llm, action_prompt_fn)

    executor = AsyncPipelineExecutor(
        world_model=world_model,
        task=task_def,
        evaluator=evaluator,
        llm=llm,
        code_prompt_fn=code_prompt_fn,
        tree=tree,
        max_rounds=args.max_rounds,
        llm_queue_depth=args.llm_queue_depth,
        num_gpus=args.num_gpus,
    )

    best_node = await executor.run()

    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    if best_node and best_node.cycle and best_node.cycle.best_round:
        best_round = best_node.cycle.best_round
        logger.info(f"Best score: {best_round.score:.4f}")
        metrics = best_round.result.get_metrics()
        logger.info(f"Speedup: {metrics.get('speedup_factor', 'N/A')}")
    else:
        logger.info("No successful solution found")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run async pipeline executor")
    parser.add_argument("--task", required=True, help="Task name (e.g., causal_conv1d)")
    parser.add_argument("--language", default="triton", choices=["triton", "cuda"])
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (or set LLM_API_KEY)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--analyze-code-failures",
        action="store_true",
        help="Use LLM to analyze failures before code generation",
    )
    parser.add_argument(
        "--analyze-action-failures",
        action="store_true",
        help="Use LLM to analyze failures before action selection",
    )
    parser.add_argument(
        "--v1-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include V1-style feedback (last round, best solution) in prompts",
    )
    # New async-specific args
    parser.add_argument(
        "--llm-queue-depth",
        type=int,
        default=4,
        help="Buffer size for proposed nodes (default: 4)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of concurrent GPU evaluations (default: 1)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs with --help**

Run: `python scripts/gpu_mode_async_pipeline_executor/run.py --help`

Expected output should include:
```
--llm-queue-depth LLM_QUEUE_DEPTH
                      Buffer size for proposed nodes (default: 4)
--num-gpus NUM_GPUS   Number of concurrent GPU evaluations (default: 1)
```

---

### Task 6: Manual Integration Test

**Step 1: Run with a real task (requires GPU and API key)**

Run:
```bash
cd /proj/data-eng/goon/flim/verl-experiments-k-search/K-Search
LLM_API_KEY=<your-key> python scripts/gpu_mode_async_pipeline_executor/run.py \
    --task tri_mul \
    --model-name <your-model> \
    --max-rounds 4 \
    --llm-queue-depth 2 \
    --num-gpus 1 \
    -v
```

Expected:
- See initial burst of 2 proposals
- See workers processing nodes concurrently
- See completion signals triggering new proposals
- Clean shutdown after 4 rounds

**Step 2: Verify speedup with multiple GPUs (if available)**

Compare timing:
```bash
# Sync baseline
time python scripts/gpu_mode_simple_linear_executor/run.py --task tri_mul --model-name <model> --max-rounds 4

# Async with 2 GPUs
time python scripts/gpu_mode_async_pipeline_executor/run.py --task tri_mul --model-name <model> --max-rounds 4 --num-gpus 2
```

---

### Task 7: Final Cleanup and Commit

**Step 1: Run ruff check and format**

```bash
ruff check scripts/gpu_mode_async_pipeline_executor/run.py --fix
ruff format scripts/gpu_mode_async_pipeline_executor/run.py
```

**Step 2: Commit everything**

```bash
git add scripts/gpu_mode_async_pipeline_executor/
git commit -m "feat(async-executor): add async pipeline executor with pipelined LLM/GPU eval"
```
