#!/usr/bin/env python3
"""V1 Case Search - world model cycles with stagnation detection.

Implements V1 case search algorithm using modular Tree/Node/Cycle/Round structures.
The modular Tree is authoritative for all metadata reads after sync from V1 manager.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import openai

from k_search.kernel_generators.world_model import render_world_model_section
from k_search.modular.llm import get_endpoint
from k_search.modular.logging import prompt_color, response_color
from k_search.kernel_generators.world_model_manager import (
    WorldModelConfig,
    WorldModelManager,
    WorldModelSelectionPolicy,
)
from k_search.kernel_generators.world_model_prompts import (
    get_debug_and_improve_from_spec_prompt_from_text,
    get_debug_generated_code_prompt_from_text,
    get_generate_code_from_action_prompt_from_text,
    get_generate_code_from_spec_with_action_prompt_from_text,
    get_improve_from_spec_prompt_from_text,
    get_improve_generated_code_prompt_from_text,
)
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.modular.protocols import Evaluator
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.round import Round
from k_search.modular.world.tree import Tree
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


@dataclass
class V1Action(Action):
    """V1-specific action with difficulty/confidence metadata."""

    difficulty: int = 3
    expected_vs_baseline_factor: float | None = None
    confidence: float = 0.5
    rationale: str = ""
    v1_action_data: dict[str, Any] | None = None


@dataclass
class V1Node(Node):
    """V1-specific node with v1 ID mapping."""

    v1_node_id: str = ""
    v1_parent_id: str = ""
    parent_is_root: bool = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP client logs
for noisy_logger in ("httpcore", "httpx", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

LLMCall = Callable[[str], str]


class ProposeContext(TypedDict):
    """Required context for V1WorldModel.propose()."""

    round_idx: int
    current_code: str


class UpdateContext(TypedDict):
    """Required context for V1WorldModel.update()."""

    cycle_succeeded: bool
    best_round: Round | None
    node: Node
    code: str
    round_idx: int
    attempts: int
    logs: str


@dataclass
class CycleConfig:
    """Configuration for cycle-based execution."""

    max_rounds_per_cycle: int = 10
    stagnation_rounds: int = 5


class V1WorldModel:
    """WorldModel implementation wrapping V1's WorldModelManager.

    After syncing from V1's JSON tree, all metadata reads come from modular
    Node/Action structures - the modular Tree is authoritative.
    """

    def __init__(
        self,
        manager: WorldModelManager,
        task_name: str,
        definition_text: str,
        language: str,
        target_gpu: str,
    ):
        self._manager = manager
        self._task_name = task_name
        self._definition_text = definition_text
        self._language = language
        self._target_gpu = target_gpu
        self._node_id_map: dict[str, Node] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._manager.ensure_initialized(
            definition_name=self._task_name,
            definition_text=self._definition_text,
        )
        self._initialized = True

    def propose(self, tree: Tree, context: ProposeContext) -> list[Node]:
        """Generate new action nodes by calling V1 manager's propose_action_nodes."""
        self._ensure_initialized()
        current_code: str = context["current_code"]
        round_idx: int = context["round_idx"]

        self._manager.propose_action_nodes(
            definition_name=self._task_name,
            definition_text=self._definition_text,
            current_code_excerpt=current_code if current_code else None,
            current_tree_path=self._manager.get_tree_path_text(
                definition_name=self._task_name
            ),
            baseline_targets_text="",
            round_index=round_idx,
        )

        return self._sync_frontier_from_manager(tree)

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Select next action using V1's policy-based chooser."""
        node_id = self._manager.choose_next_action_node_id(
            definition_name=self._task_name
        )
        if not node_id:
            return []

        node = self._node_id_map.get(node_id)
        if node:
            self._manager.set_active_leaf_id(
                definition_name=self._task_name, node_id=node_id
            )
            return [node]
        return []

    def update(self, tree: Tree, context: UpdateContext) -> None:
        """Update tree after cycle - attach solution or mark too hard."""
        success: bool = context["cycle_succeeded"]
        best_round: Round | None = context["best_round"]
        round_idx: int = context["round_idx"]
        code: str = context["code"]
        node: Node = context["node"]

        # Get action text from modular Node (authoritative source)
        action_text = ""
        if node and node.action:
            action_text = node.action.title

        if success and best_round:
            from k_search.tasks.task_base import EvalResult

            eval_dict = best_round.result.get_metrics()
            eval_result = EvalResult(
                status="passed" if best_round.result.succeeded() else "failed",
                latency_ms=eval_dict.get("latency_ms"),
                speedup_factor=eval_dict.get("speedup_factor"),
            )
            self._manager.refine(
                definition_name=self._task_name,
                definition_text=self._definition_text,
                chosen_action_text=action_text,
                current_code_excerpt=code,
                current_tree_path=self._manager.get_tree_path_text(
                    definition_name=self._task_name
                ),
                eval_result=eval_result,
                prediction=None,
                round_index=round_idx,
            )
        else:
            self._manager.note_action_too_hard(
                definition_name=self._task_name,
                definition_text=self._definition_text,
                chosen_action_text=action_text,
                current_code_excerpt=code,
                current_tree_path=self._manager.get_tree_path_text(
                    definition_name=self._task_name
                ),
                eval_result=None,
                debug_and_improve_round=context["attempts"],
                debug_and_improve_max_rounds=10,
                baseline_targets_text="",
                round_index=round_idx,
            )

    def _get_prompt_section(self, max_chars: int = 6000) -> str:
        """Get world model section to append to prompts."""
        wm_json = self._manager.get(self._task_name)
        return render_world_model_section(wm_json, max_chars=max_chars)

    def _sync_frontier_from_manager(self, tree: Tree) -> list[Node]:
        """Sync V1 JSON tree nodes to modular Tree, return new frontier nodes.

        All V1 metadata is copied to Node.annotations and Action.annotations
        so the modular Tree becomes the authoritative source for reads.
        """
        wm_json = self._manager.get(self._task_name)
        if not wm_json:
            return []

        try:
            wm = json.loads(wm_json)
        except json.JSONDecodeError:
            return []

        dt = wm.get("decision_tree", {})
        nodes_list = dt.get("nodes", [])
        new_nodes = []

        for node_data in nodes_list:
            node_id = node_data.get("node_id")
            if not node_id or node_id in self._node_id_map:
                continue
            if node_id == dt.get("root_id", "root"):
                continue

            parent_id = node_data.get("parent_id")
            action_data = node_data.get("action") or {}

            # Extract all V1 metadata
            title = action_data.get("title", "")
            difficulty = action_data.get("difficulty_1_to_5", 3)
            expected_vs_baseline = action_data.get("expected_vs_baseline_factor")
            confidence = action_data.get("confidence", 0.5)
            rationale = action_data.get("rationale", "")

            parent_node = self._node_id_map.get(parent_id, tree.root)

            new_node = V1Node(
                parent=parent_node,
                status="open",
                action=V1Action(
                    title=title,
                    difficulty=difficulty,
                    expected_vs_baseline_factor=expected_vs_baseline,
                    confidence=confidence,
                    rationale=rationale,
                    v1_action_data=action_data,
                ),
                v1_node_id=node_id,
                v1_parent_id=parent_id,
                parent_is_root=parent_id == "root" or parent_id is None,
            )
            tree.add_node(new_node)
            self._node_id_map[node_id] = new_node
            new_nodes.append(new_node)

        return new_nodes

    def get_action_context(self, node: Node) -> dict[str, Any]:
        """Get context for the selected action node - reads from V1Node/V1Action fields."""
        if not node:
            return {}

        v1_node = node  # type: V1Node
        v1_action = node.action  # type: V1Action | None

        parent_is_root = getattr(v1_node, "parent_is_root", node.parent is None or node.parent.parent is None)

        base_code = ""
        base_score = 0.0
        if node.parent and node.parent.cycle and node.parent.cycle.best_round:
            best = node.parent.cycle.best_round
            base_code = best.llm_response
            base_score = best.score

        return {
            "v1_node_id": getattr(v1_node, "v1_node_id", ""),
            "action_text": node.action.title if node.action else "",
            "difficulty": getattr(v1_action, "difficulty", 3) if v1_action else 3,
            "confidence": getattr(v1_action, "confidence", 0.5) if v1_action else 0.5,
            "rationale": getattr(v1_action, "rationale", "") if v1_action else "",
            "parent_is_root": parent_is_root,
            "base_code": base_code,
            "base_score": base_score,
        }


class V1PromptBuilder:
    """Handles V1-style prompt routing based on cycle phase."""

    def __init__(
        self,
        definition_text: str,
        language: str,
        target_gpu: str,
    ):
        self._definition_text = definition_text
        self._language = language
        self._target_gpu = target_gpu

    def build(
        self,
        action_text: str,
        attempt: int,
        last_round: Round | None,
        has_passed: bool,
        base_code: str,
        trace_logs: str,
        current_code: str,
        perf_summary: str = "",
    ) -> str:
        """Build prompt based on cycle phase.

        Prompt selection:
        - Attempt 0: action prompt (with or without base code)
        - Subsequent, no PASSED: debug prompt
        - Subsequent, has PASSED: improve prompt
        """
        has_base = bool(base_code and base_code.strip())

        if attempt == 0:
            if has_base:
                return get_generate_code_from_action_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    base_code=base_code,
                    action_text=action_text,
                    target_gpu=self._target_gpu,
                )
            else:
                return get_generate_code_from_spec_with_action_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    action_text=action_text,
                    target_gpu=self._target_gpu,
                )

        if not has_passed:
            if has_base:
                return get_debug_generated_code_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    trace_logs=trace_logs,
                    base_code=base_code,
                    buggy_code=current_code,
                    action_text=action_text,
                    debug_round=attempt,
                    max_rounds=10,
                    target_gpu=self._target_gpu,
                    perf_summary=perf_summary,
                )
            else:
                return get_debug_and_improve_from_spec_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    trace_logs=trace_logs,
                    current_code=current_code,
                    action_text=action_text,
                    debug_round=attempt,
                    max_rounds=10,
                    target_gpu=self._target_gpu,
                    perf_summary=perf_summary,
                )

        if has_base:
            return get_improve_generated_code_prompt_from_text(
                self._language,
                definition_text=self._definition_text,
                trace_logs=trace_logs,
                base_code=base_code,
                current_code=current_code,
                debug_round=attempt,
                max_rounds=10,
                target_gpu=self._target_gpu,
                perf_summary=perf_summary,
            )
        else:
            return get_improve_from_spec_prompt_from_text(
                self._language,
                definition_text=self._definition_text,
                trace_logs=trace_logs,
                current_code=current_code,
                debug_round=attempt,
                max_rounds=10,
                target_gpu=self._target_gpu,
                perf_summary=perf_summary,
            )


class V1SequentialExecutor:
    """Executor implementing V1 case search with world model cycles.

    Operates entirely on modular Tree/Node/Cycle/Round structures.
    All metadata reads come from Node.annotations and Action.annotations.
    """

    def __init__(
        self,
        world_model: V1WorldModel,
        task: TaskDefinition,
        evaluator: Evaluator,
        llm: LLMCall,
        prompt_builder: V1PromptBuilder,
        tree: Tree,
        max_rounds: int,
        cycle_config: CycleConfig | None = None,
    ):
        self._world_model = world_model
        self._task = task
        self._evaluator = evaluator
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._tree = tree
        self._max_rounds = max_rounds
        self._cycle_config = cycle_config or CycleConfig()

    def run(self) -> Node | None:
        """Execute search, return best node."""
        rounds_used = 0

        while rounds_used < self._max_rounds:
            logger.info(
                "[CYCLE_START] rounds_used=%d/%d", rounds_used, self._max_rounds
            )

            self._world_model.propose(
                self._tree, context={"round_idx": rounds_used, "current_code": ""}
            )

            selected = self._world_model.select(self._tree, context={})
            if not selected:
                logger.info("No more actions to select, stopping")
                break

            node = selected[0]
            node.status = "in_progress"
            logger.info(
                "[ACTION] Selected: %s", node.action.title if node.action else "unknown"
            )

            # Get action context from modular Node (authoritative source)
            action_ctx = self._world_model.get_action_context(node)

            cycle = self._run_cycle(
                node, action_ctx, rounds_remaining=self._max_rounds - rounds_used
            )
            node.cycle = cycle
            node.status = "closed"

            self._world_model.update(
                self._tree,
                context={
                    "cycle_succeeded": cycle.succeeded,
                    "best_round": cycle.best_round,
                    "node": node,
                    "code": cycle.best_round.llm_response if cycle.best_round else "",
                    "round_idx": rounds_used + len(cycle.rounds) - 1,
                    "attempts": len(cycle.rounds),
                    "logs": cycle.rounds[-1].result.get_log() if cycle.rounds else "",
                },
            )

            rounds_used += len(cycle.rounds)
            logger.info(
                "[CYCLE_END] cycle_rounds=%d, total=%d", len(cycle.rounds), rounds_used
            )

        return self._tree.get_best_node()

    def _run_cycle(
        self, node: Node, action_ctx: dict[str, Any], rounds_remaining: int
    ) -> Cycle:
        """Run multiple attempts on a single action with stagnation detection."""
        rounds: list[Round] = []
        best_score = 0.0
        best_speedup: float | None = None
        no_improve = 0

        # All metadata from action_ctx (which reads from modular Node)
        action_text = action_ctx.get(
            "action_text", node.action.title if node.action else ""
        )
        base_code = action_ctx.get("base_code", "")

        max_attempts = min(self._cycle_config.max_rounds_per_cycle, rounds_remaining)

        current_code = ""
        for attempt in range(max_attempts):
            best_speedup_str = f"{best_speedup:.2f}x" if best_speedup else "-"
            logger.info(
                "[ATTEMPT] %d/%d | best=%.4f (%s) | no_improve=%d/%d",
                attempt + 1,
                max_attempts,
                best_score,
                best_speedup_str,
                no_improve,
                self._cycle_config.stagnation_rounds,
            )

            last_round = rounds[-1] if rounds else None
            has_passed = best_score > 0
            trace_logs = last_round.result.get_log() if last_round else ""

            prompt = self._prompt_builder.build(
                action_text=action_text,
                attempt=attempt,
                last_round=last_round,
                has_passed=has_passed,
                base_code=base_code,
                trace_logs=trace_logs,
                current_code=current_code,
            )

            wm_section = self._world_model._get_prompt_section()
            if wm_section:
                prompt = prompt + "\n\n" + wm_section

            logger.debug(
                prompt_color(
                    f"[CODE_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
                )
            )

            code = self._llm(prompt)
            current_code = code

            logger.debug(
                response_color(f"[CODE_RESPONSE] ({len(code)} chars):\n\n{code}\n")
            )

            impl = self._task.create_impl(code)
            result = self._evaluator.evaluate(impl, context={"round_idx": attempt})
            score = self._task.scorer.score(result)

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
            rounds.append(round_)

            metrics = result.get_metrics()
            speedup = metrics.get("speedup_factor")
            speedup_str = f"{speedup:.2f}x" if speedup else "-"

            if result.succeeded() and score > best_score:
                best_score = score
                best_speedup = speedup
                no_improve = 0
                logger.info("[IMPROVED] score=%.4f, speedup=%s", score, speedup_str)
            else:
                no_improve += 1
                best_speedup_str = f"{best_speedup:.2f}x" if best_speedup else "-"
                logger.info(
                    "[NO_IMPROVE] score=%.4f, speedup=%s, best_speedup=%s, streak=%d",
                    score,
                    speedup_str,
                    best_speedup_str,
                    no_improve,
                )

            if no_improve >= self._cycle_config.stagnation_rounds:
                logger.info("[STAGNATION] no improvement for %d rounds", no_improve)
                break

        return Cycle(rounds=rounds)


ReasoningEffort = Literal["low", "medium", "high"]


def create_llm_call(
    client: openai.OpenAI,
    model_name: str,
    use_reasoning_api: bool = True,
    reasoning_effort: ReasoningEffort = "medium",
) -> LLMCall:
    """Create LLM callable for search loop."""

    def llm_call(prompt: str) -> str:
        if use_reasoning_api:
            response = client.responses.create(
                model=model_name,
                input=prompt,
                reasoning={"effort": reasoning_effort},
            )
            return (response.output_text or "").strip()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    return llm_call


def wrap_with_action_logging(llm: LLMCall) -> LLMCall:
    """Wrap LLM call with ACTION_PROMPT/ACTION_RESPONSE logging."""

    def logged_llm_call(prompt: str) -> str:
        logger.debug(
            prompt_color(
                f"[ACTION_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
            )
        )
        result = llm(prompt)
        logger.debug(
            response_color(f"[ACTION_RESPONSE] ({len(result)} chars):\n\n{result}\n")
        )
        return result

    return logged_llm_call


def main():
    parser = argparse.ArgumentParser(description="Run V1 case search with world model")
    parser.add_argument(
        "--task", required=True, help="Task name (e.g., causal_conv1d, trimul)"
    )
    parser.add_argument("--language", default="triton", choices=["triton", "cuda"])
    parser.add_argument("--max-rounds", type=int, default=128)
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument(
        "--api-key", default=None, help="API key; if omitted, uses LLM_API_KEY env var"
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--no-reasoning-api",
        dest="use_reasoning_api",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", choices=["low", "medium", "high"]
    )
    parser.add_argument("--max-rounds-per-cycle", type=int, default=10)
    parser.add_argument("--stagnation-rounds", type=int, default=5)
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=None,
        help="Max difficulty (1-5) for action selection",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("API key is required (pass --api-key or set LLM_API_KEY)")
        sys.exit(1)

    task_dir = (
        Path(__file__).parent.parent.parent
        / "k_search"
        / "tasks"
        / "gpu_mode"
        / args.task
    )
    if not task_dir.exists():
        logger.error("Task directory not found: %s", task_dir)
        sys.exit(1)

    logger.info("Loading task: %s", args.task)
    gpu_task = GpuModeTriMulTask(task_dir=task_dir)
    task_def = GpuModeTriMulTaskDefinition(gpu_task, language=args.language)
    evaluator = GpuModeEvaluator(gpu_task)

    definition_text = gpu_task.get_definition_text(args.language)

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
        "base_url": get_endpoint(args.model_name),
    }
    if rits_key := os.getenv("RITS_API_KEY"):
        client_kwargs["default_headers"] = {"RITS_API_KEY": rits_key}

    client = openai.OpenAI(**client_kwargs)
    reasoning_effort: ReasoningEffort = args.reasoning_effort
    llm = create_llm_call(
        client,
        args.model_name,
        use_reasoning_api=args.use_reasoning_api,
        reasoning_effort=reasoning_effort,
    )

    selection_policy = WorldModelSelectionPolicy()
    if args.max_difficulty is not None:
        selection_policy.max_difficulty_1_to_5 = int(args.max_difficulty)

    wm_config = WorldModelConfig(
        enabled=True,
        selection_policy=selection_policy,
    )
    wm_manager = WorldModelManager(
        llm_call=wrap_with_action_logging(llm),
        target_gpu="H100",
        language=args.language,
        config=wm_config,
    )

    world_model = V1WorldModel(
        manager=wm_manager,
        task_name=args.task,
        definition_text=definition_text,
        language=args.language,
        target_gpu="H100",
    )

    prompt_builder = V1PromptBuilder(
        definition_text=definition_text,
        language=args.language,
        target_gpu="H100",
    )

    cycle_config = CycleConfig(
        max_rounds_per_cycle=args.max_rounds_per_cycle,
        stagnation_rounds=args.stagnation_rounds,
    )

    tree = Tree(root=Node(status="closed"))

    executor = V1SequentialExecutor(
        world_model=world_model,
        task=task_def,
        evaluator=evaluator,
        llm=llm,
        prompt_builder=prompt_builder,
        tree=tree,
        max_rounds=args.max_rounds,
        cycle_config=cycle_config,
    )

    logger.info(
        "Starting V1 case search: max_rounds=%d, model=%s",
        args.max_rounds,
        args.model_name,
    )

    best_node = executor.run()

    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    if best_node and best_node.cycle and best_node.cycle.best_round:
        best = best_node.cycle.best_round
        logger.info("Best score: %.4f", best.score)
        logger.info("Best round metrics: %s", best.result.get_metrics())
    else:
        logger.info("No successful solution found")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
