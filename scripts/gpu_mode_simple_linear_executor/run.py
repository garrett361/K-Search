#!/usr/bin/env python3
"""GPU mode executor entry point - SimpleWorldModel + SequentialExecutor.

All code in one file for easy one-off scripting.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import openai

from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.modular.executors import SequentialExecutor
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


ACTION_PROMPT_TEMPLATE = """You are proposing the next optimization action for a GPU kernel.

## Task Specification
{task_spec}
{feedback_section}
## Your Job
Propose ONE specific optimization action to try next.

Rules:
- Single-iteration implementable, SMALL change
- One concrete tweak (tiling OR memory OR scheduling - not multiple)
- Be specific (e.g., "use shared memory for input tile" not "optimize memory")

Respond with only the action title (one line, no explanation)."""


def create_action_prompt_fn(task_def: GpuModeTriMulTaskDefinition):
    """Create GPU mode specific action prompt function.

    Uses feedback from best round to inform next action proposal.
    """

    def action_prompt_fn(tree: Tree, context: dict[str, Any] | None) -> str:
        task_spec = task_def.get_prompt_text()

        feedback_section = ""
        best_node = tree.get_best_node()
        if best_node and best_node.cycle:
            best_round = best_node.cycle.best_round
            if best_round:
                feedback = task_def.feedback_provider.for_codegen(best_round)
                feedback_section = f"\n## Previous Best Result\n{feedback}\n"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            task_spec=task_spec,
            feedback_section=feedback_section,
        )
        logger.debug("ACTION PROMPT:\n\n%s\n", prompt)
        return prompt

    return action_prompt_fn


def create_code_prompt_fn(task_def: GpuModeTriMulTaskDefinition):
    """Create GPU mode specific code generation prompt function.

    Round 0 (no action): direct code generation from task prompt.
    Round 1+ (with action): generate code implementing the specific action.
    """

    def code_prompt_fn(node: Node, task: TaskDefinition) -> str:
        if node.action:
            prompt = f"{task_def.get_prompt_text()}\n\nAction: {node.action.title}\n\nGenerate the implementation:"
        else:
            prompt = f"{task_def.get_prompt_text()}\n\nGenerate the implementation:"
        logger.debug("CODE PROMPT:\n\n%s\n", prompt)
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
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (or set LLM_API_KEY)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
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

    client_kwargs = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if (rits_api_key := os.getenv("RITS_API_KEY")) is not None:
        client_kwargs["default_headers"] = {"RITS_API_KEY": rits_api_key}

    client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
    llm = create_llm_call(client, args.model_name)

    action_prompt_fn = create_action_prompt_fn(task_def)
    code_prompt_fn = create_code_prompt_fn(task_def)

    world_model = SimpleWorldModel(llm, action_prompt_fn)
    tree = Tree(root=Node(status="closed"))

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
