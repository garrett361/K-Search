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
from k_search.modular.llm import get_endpoint
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


async def _analyze_failure(
    llm: AsyncLLMCallable, error_log: str, failed_code: str
) -> str:
    """Ask LLM to analyze a failure and suggest fixes for code generation."""
    prompt = FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-2000:],
        failed_code=failed_code[-3000:],
    )
    return await llm(prompt)


async def _analyze_failure_for_action(
    llm: AsyncLLMCallable, error_log: str, action: str
) -> str:
    """Ask LLM to explain failure and guide next action selection."""
    prompt = ACTION_FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-1000:],
        action=action,
    )
    return await llm(prompt)


def create_action_prompt_fn(
    task_def: GpuModeTriMulTaskDefinition,
    llm: AsyncLLMCallable,
    *,
    analyze_failures: bool = False,
):
    """Create GPU mode specific action prompt function."""

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


def create_code_prompt_fn(
    task_def: GpuModeTriMulTaskDefinition,
    tree: Tree,
    llm: AsyncLLMCallable,
    *,
    analyze_failures: bool = False,
    v1_feedback: bool = False,
):
    """Create GPU mode specific code generation prompt function."""

    def _format_round_summary(r) -> str:
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


INITIAL_ACTION = "Write an optimized implementation."

AsyncActionPromptFn = Callable[[Tree, dict[str, Any] | None], Any]


class AsyncSimpleWorldModel:
    """World model for async executor - handles in-flight nodes gracefully."""

    def __init__(self, llm: AsyncLLMCallable, action_prompt_fn: AsyncActionPromptFn):
        self._llm = llm
        self._action_prompt_fn = action_prompt_fn

    async def propose(
        self, tree: Tree, context: dict[str, Any] | None = None
    ) -> list[Node]:
        completed = [n for n in tree._all_nodes() if n.status == "closed" and n.cycle]

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

    async def select(
        self, tree: Tree, context: dict[str, Any] | None = None
    ) -> list[Node]:
        return []

    async def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        pass

    def _get_last_closed_node(self, tree: Tree) -> Node | None:
        closed = [n for n in tree._all_nodes() if n.status == "closed" and n.cycle]
        if not closed:
            return None
        return max(closed, key=lambda n: int(n._id))


CodePromptFn = Callable[[Node, TaskDefinition], str]


class AsyncPipelineExecutor:
    """Async pipeline executor - overlaps LLM generation with GPU evaluation."""

    def __init__(
        self,
        world_model: AsyncSimpleWorldModel,
        task: TaskDefinition,
        evaluator: Any,
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
        llm_queue: asyncio.Queue[Node | None] = asyncio.Queue(
            maxsize=self._llm_queue_depth
        )
        completion_queue: asyncio.Queue[Node] = asyncio.Queue()
        gpu_semaphore = asyncio.Semaphore(self._num_gpus)

        logger.info(
            f"Starting async pipeline: max_rounds={self._max_rounds}, "
            f"llm_queue_depth={self._llm_queue_depth}, num_gpus={self._num_gpus}"
        )

        async with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
            tg.create_task(self._proposer(llm_queue, completion_queue))
            for worker_id in range(self._llm_queue_depth):
                tg.create_task(
                    self._worker(worker_id, llm_queue, completion_queue, gpu_semaphore)
                )

        logger.info(f"Pipeline complete. Rounds completed: {self._rounds_completed}")
        return self._tree.get_best_node()

    async def _proposer(
        self,
        llm_queue: asyncio.Queue[Node | None],
        completion_queue: asyncio.Queue[Node],
    ) -> None:
        initial_count = min(self._llm_queue_depth, self._max_rounds)
        logger.info(f"[PROPOSER] Initial burst: proposing {initial_count} nodes")
        for i in range(initial_count):
            nodes = await self._world_model.propose(self._tree)
            for node in nodes:
                self._tree.add_node(node)
                await llm_queue.put(node)
                logger.debug(f"[PROPOSER] Queued initial node {i + 1}/{initial_count}")

        rounds_proposed = initial_count
        while rounds_proposed < self._max_rounds:
            await completion_queue.get()
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

        while self._rounds_completed < self._max_rounds:
            await completion_queue.get()
            self._rounds_completed += 1
            logger.debug(
                f"[PROPOSER] Draining completion ({self._rounds_completed}/{self._max_rounds})"
            )

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
        while True:
            node = await llm_queue.get()
            if node is None:
                logger.debug(f"[WORKER-{worker_id}] Received shutdown sentinel")
                break

            logger.debug(
                f"[WORKER-{worker_id}] Processing node: {node.action.title[:30] if node.action else 'no-action'}..."
            )
            node.status = "in_progress"

            prompt = self._code_prompt_fn(node, self._task)
            code = await self._llm(prompt)
            logger.debug(
                response_color(
                    f"[CODE_RESPONSE] ({len(code)} chars, ~{len(code) // 4} toks)"
                )
            )

            impl = self._task.create_impl(code)

            async with gpu_semaphore:
                logger.debug(
                    f"[WORKER-{worker_id}] Acquired GPU semaphore, evaluating..."
                )
                result = await asyncio.to_thread(
                    self._evaluator.evaluate, impl, context={}
                )

            score = self._task.scorer.score(result)
            logger.debug(
                f"[WORKER-{worker_id}] Eval complete: score={score:.4f}, success={result.succeeded()}"
            )

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
                    logger.debug(
                        f"[WORKER-{worker_id}] Error:\n{_truncate_log(log_excerpt, 10)}"
                    )

            await completion_queue.put(node)


def create_async_llm_call(
    client: openai.AsyncOpenAI, model_name: str
) -> AsyncLLMCallable:
    async def llm_call(prompt: str) -> str:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    return llm_call


async def async_main(args: argparse.Namespace) -> None:
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

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
        "base_url": get_endpoint(args.model_name),
    }
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
