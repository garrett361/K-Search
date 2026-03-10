#!/usr/bin/env python3
"""GPU mode executor entry point - SimpleWorldModel + SequentialExecutor.

All code in one file for easy one-off scripting.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from collections.abc import Callable

import openai

from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.modular.executors import SequentialExecutor
from k_search.modular.llm import get_endpoint
from k_search.modular.logging import prompt_color
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import (
    SimpleWorldModel,
    SimpleWorldModelContext,
)
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP client logs
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
    import re

    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response


def _collect_all_nodes(tree: Tree) -> list[Node]:
    """Collect all nodes in tree via BFS."""
    result = []
    queue = [tree.root]
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _get_last_evaluated_node(tree: Tree) -> Node | None:
    """Return most recently evaluated node (highest ID, closed, has cycle)."""
    evaluated = [
        n
        for n in _collect_all_nodes(tree)
        if n.status == "closed" and n.cycle and n.cycle.rounds
    ]
    if not evaluated:
        return None
    return max(evaluated, key=lambda n: int(n.id))


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

LLMCallable = Callable[[str], str]


def _analyze_failure(llm: LLMCallable, error_log: str, failed_code: str) -> str:
    """Ask LLM to analyze a failure and suggest fixes for code generation."""
    prompt = FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-2000:],
        failed_code=failed_code[-3000:],
    )
    return llm(prompt)


def _analyze_failure_for_action(llm: LLMCallable, error_log: str, action: str) -> str:
    """Ask LLM to explain failure and guide next action selection."""
    prompt = ACTION_FAILURE_ANALYSIS_PROMPT.format(
        error_log=error_log[-1000:],
        action=action,
    )
    return llm(prompt)


def create_action_prompt_fn(
    task_def: GpuModeTriMulTaskDefinition,
    llm: LLMCallable,
    *,
    analyze_failures: bool = False,
):
    """Create GPU mode specific action prompt function.

    Uses feedback from best round to inform next action.
    If analyze_failures=True, also uses LLM to analyze failures.
    """

    def action_prompt_fn(context: SimpleWorldModelContext) -> str:
        task_spec = task_def.get_prompt_text()
        tree = context.tree

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
                    analysis = _analyze_failure_for_action(llm, error_log, action)
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
    llm: LLMCallable,
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
            # Last round feedback with descriptive headers
            if last_node and last_node.cycle and last_node.cycle.rounds:
                last_round = last_node.cycle.rounds[-1]
                logs = _truncate_log(last_round.result.get_log(), max_lines=30)
                code = _extract_code_block(last_round.llm_response)
                summary = _format_round_summary(last_round)

                prompt += f"\n\nEvaluation Output (from your previous attempt):\n{logs}"
                prompt += f"\n\nYour Previous Code:\n```python\n{code}\n```"
                prompt += f"\n\nPrevious Round Summary:\n{summary}"

            # Best so far (only if different from last)
            best_node = tree.get_best_node()
            if best_node and best_node.cycle and best_node.cycle.best_round:
                best_round = best_node.cycle.best_round
                if last_node is None or best_node.id != last_node.id:
                    best_summary = _format_round_summary(best_round)
                    best_code = best_round.llm_response
                    prompt += f"\n\nBest Successful Solution So Far:\n{best_summary}\n\nCode:\n{best_code}"

        if analyze_failures:
            # LLM failure analysis if last round failed
            if last_node and last_node.cycle and last_node.cycle.rounds:
                last_round = last_node.cycle.rounds[-1]
                if not last_round.result.succeeded():
                    error_log = last_round.result.get_log()
                    failed_code = last_round.llm_response
                    analysis = _analyze_failure(llm, error_log, failed_code)
                    prompt += f"\n\n## Failure Analysis\n{analysis}"

        prompt += "\n\nGenerate the corrected and optimized implementation:"
        logger.debug(
            prompt_color(
                f"[CODE_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
            )
        )
        return prompt

    return code_prompt_fn


def create_llm_call(client: openai.OpenAI, model_name: str):
    """Create LLM callable."""

    def llm_call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    return llm_call


def main():
    parser = argparse.ArgumentParser(description="Run GPU mode executor")
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
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    client_kwargs = {
        "api_key": api_key,
        "timeout": args.timeout,
        "base_url": get_endpoint(args.model_name),
    }
    if (rits_api_key := os.getenv("RITS_API_KEY")) is not None:
        client_kwargs["default_headers"] = {"RITS_API_KEY": rits_api_key}

    client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
    llm = create_llm_call(client, args.model_name)

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

    world_model = SimpleWorldModel(llm, action_prompt_fn)

    executor = SequentialExecutor(
        world_model=world_model,
        task=task_def,
        evaluator=evaluator,
        llm=llm,
        code_prompt_fn=code_prompt_fn,
        tree=tree,
        max_rounds=args.max_rounds,
    )

    logger.info(
        f"Starting executor: max_rounds={args.max_rounds}, model={args.model_name}"
    )
    best_node = executor.run()

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


if __name__ == "__main__":
    main()
