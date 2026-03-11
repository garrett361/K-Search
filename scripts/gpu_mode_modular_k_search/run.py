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
from typing import Any, Callable, Literal, cast

import openai

from k_search.kernel_generators.world_model import render_world_model_section
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
from k_search.modular.llm import get_endpoint
from k_search.modular.logging import prompt_color, response_color
from k_search.modular.protocols import EvaluationResult, Evaluator
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


@dataclass
class V1Node(Node):
    """V1-specific node with v1 ID mapping."""

    action: V1Action | None = None
    node_id: str = ""
    parent_id: str = ""
    parent_is_root: bool = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


LLMCall = Callable[[str], str]


@dataclass
class V1ProposeContext:
    """Context for V1WorldModel.propose()."""

    tree: Tree
    round_idx: int
    current_code: str = ""


@dataclass
class V1SelectContext:
    """Context for V1WorldModel.select()."""

    tree: Tree


@dataclass
class V1UpdateContext:
    """Context for V1WorldModel.update()."""

    tree: Tree
    node: Node
    cycle: Cycle
    round_idx: int


@dataclass
class CycleConfig:
    """Configuration for cycle-based execution."""

    stagnation_rounds: int = 5
    max_debug_improve_rounds: int = 5


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
        self.task_name = task_name
        self._definition_text = definition_text
        self._language = language
        self._target_gpu = target_gpu
        self._node_id_map: dict[str, V1Node] = {}
        self._initialized = False
        self._cached_wm: dict | None = None
        self._cached_wm_json: str | None = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._manager.ensure_initialized(
            definition_name=self.task_name,
            definition_text=self._definition_text,
        )
        self._initialized = True

    def _get_parsed_wm(self) -> dict | None:
        """Get parsed world model, caching to avoid redundant json.loads()."""
        wm_json = self._manager.get(self.task_name)
        if wm_json is None:
            self._cached_wm = None
            self._cached_wm_json = None
            return None
        if wm_json != self._cached_wm_json:
            self._cached_wm = json.loads(wm_json)
            self._cached_wm_json = wm_json
        return self._cached_wm

    def invalidate_cache(self) -> None:
        """Call after operations that modify the world model."""
        self._cached_wm = None
        self._cached_wm_json = None

    def propose(self, context: V1ProposeContext) -> list[V1Node]:
        """Generate new action nodes by calling V1 manager's propose_action_nodes."""
        self._ensure_initialized()

        self._manager.propose_action_nodes(
            definition_name=self.task_name,
            definition_text=self._definition_text,
            current_code_excerpt=context.current_code if context.current_code else None,
            current_tree_path=self._manager.get_tree_path_text(
                definition_name=self.task_name
            ),
            baseline_targets_text="",
            round_index=context.round_idx,
        )
        self.invalidate_cache()

        return self._sync_frontier_from_manager(context.tree)

    def select(self, context: V1SelectContext) -> list[V1Node]:
        """Select next action using V1's policy-based chooser."""
        node_id = self._manager.choose_next_action_node_id(
            definition_name=self.task_name
        )
        if not node_id:
            return []

        node = self._node_id_map.get(node_id)
        if node:
            self._manager.set_active_leaf_id(
                definition_name=self.task_name, node_id=node_id
            )
            return [node]
        return []

    def update(self, context: V1UpdateContext) -> None:
        """Update tree after cycle - attach solution or mark too hard."""
        node = context.node
        cycle = context.cycle
        best_round = cycle.best_round
        code = best_round.llm_response if best_round else ""

        action_text = ""
        if node and node.action:
            action_text = node.action.title

        if best_round is not None:
            from k_search.tasks.task_base import EvalResult

            eval_dict = best_round.result.get_metrics()
            if best_round.result.succeeded():
                eval_result = EvalResult(
                    status="passed",
                    latency_ms=eval_dict["latency_ms"],
                    speedup_factor=eval_dict["speedup_factor"],
                )
            else:
                eval_result = EvalResult(status="failed")
            self._manager.refine(
                definition_name=self.task_name,
                definition_text=self._definition_text,
                chosen_action_text=action_text,
                current_code_excerpt=code,
                current_tree_path=self._manager.get_tree_path_text(
                    definition_name=self.task_name
                ),
                eval_result=eval_result,
                prediction=None,
                round_index=context.round_idx,
            )
        else:
            self._manager.note_action_too_hard(
                definition_name=self.task_name,
                definition_text=self._definition_text,
                chosen_action_text=action_text,
                current_code_excerpt=code,
                current_tree_path=self._manager.get_tree_path_text(
                    definition_name=self.task_name
                ),
                eval_result=None,
                debug_and_improve_round=len(cycle.rounds),
                debug_and_improve_max_rounds=10,
                baseline_targets_text="",
                round_index=context.round_idx,
            )
        self.invalidate_cache()

    def _get_prompt_section(self, max_chars: int = 6000) -> str:
        """Get world model section to append to prompts."""
        wm_dict = self._get_parsed_wm()
        if wm_dict is None:
            return ""
        return render_world_model_section(self._cached_wm_json, max_chars=max_chars)

    def _sync_frontier_from_manager(self, tree: Tree) -> list[V1Node]:
        """Sync V1 JSON tree nodes to modular Tree, return new frontier nodes."""
        wm = self._get_parsed_wm()
        if wm is None:
            raise ValueError(f"No world model found for task {self.task_name}")

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
                ),
                node_id=node_id,
                parent_id=parent_id,
                parent_is_root=parent_id == "root" or parent_id is None,
            )
            tree.add_node(new_node)
            self._node_id_map[node_id] = new_node
            new_nodes.append(new_node)

        return new_nodes

    def get_action_context(self, node: V1Node) -> dict[str, Any]:
        """Get context for the selected action node."""
        base_code = ""
        base_score = -1.0  # V1 semantics: -1.0 when no parent solution
        base_result: EvaluationResult | None = None
        if node.parent and node.parent.cycle and node.parent.cycle.best_round:
            best = node.parent.cycle.best_round
            base_code = best.llm_response
            base_score = best.score
            base_result = best.result

        action: V1Action | None = node.action
        return {
            "node_id": node.node_id,
            "action_text": action.title if action else "",
            "difficulty": action.difficulty if action else 3,
            "confidence": action.confidence if action else 0.5,
            "rationale": action.rationale if action else "",
            "parent_is_root": node.parent_is_root,
            "base_code": base_code,
            "base_score": base_score,
            "base_result": base_result,
        }


def _get_perf_summary_lines(result: EvaluationResult | None, prefix: str) -> list[str]:
    """Extract perf_summary_lines from EvaluationResult wrapper."""
    if result is None:
        return []
    inner = getattr(result, "_inner", result)
    if hasattr(inner, "perf_summary_lines"):
        return cast(Any, inner).perf_summary_lines(prefix=prefix)
    return []


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
        parent_is_root: bool = False,
        max_debug_improve_rounds: int = 5,
    ) -> str:
        """Build prompt based on cycle phase.

        Prompt selection (V1 4-level nesting):
        - Attempt 0: action prompt (with or without base code)
        - Subsequent, parent_is_root or no base: debug/improve from spec
        - Subsequent, has base: debug/improve vs base code
        """
        has_base = bool(base_code and base_code.strip())
        # V1 semantics: 1-indexed, capped at max_debug_improve_rounds
        debug_round = min(attempt + 1, max_debug_improve_rounds)

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

        # V1 semantics: parent_is_root means we use spec-based prompts even if
        # base_code exists (base_code would be from root, not a real parent solution)
        if not has_passed:
            if parent_is_root or not has_base:
                # TYPE C: debug from spec
                return get_debug_and_improve_from_spec_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    trace_logs=trace_logs,
                    current_code=current_code,
                    action_text=action_text,
                    debug_round=debug_round,
                    max_rounds=max_debug_improve_rounds,
                    target_gpu=self._target_gpu,
                    perf_summary=perf_summary,
                )
            else:
                # TYPE D: debug vs base code
                return get_debug_generated_code_prompt_from_text(
                    self._language,
                    definition_text=self._definition_text,
                    trace_logs=trace_logs,
                    base_code=base_code,
                    buggy_code=current_code,
                    action_text=action_text,
                    debug_round=debug_round,
                    max_rounds=max_debug_improve_rounds,
                    target_gpu=self._target_gpu,
                    perf_summary=perf_summary,
                )

        # has_passed == True
        if parent_is_root or not has_base:
            # TYPE E: improve from spec
            return get_improve_from_spec_prompt_from_text(
                self._language,
                definition_text=self._definition_text,
                trace_logs=trace_logs,
                current_code=current_code,
                debug_round=debug_round,
                max_rounds=max_debug_improve_rounds,
                target_gpu=self._target_gpu,
                perf_summary=perf_summary,
            )
        else:
            # TYPE F: improve vs base code
            return get_improve_generated_code_prompt_from_text(
                self._language,
                definition_text=self._definition_text,
                trace_logs=trace_logs,
                base_code=base_code,
                current_code=current_code,
                debug_round=debug_round,
                max_rounds=max_debug_improve_rounds,
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
        self.world_model = world_model
        self.task = task
        self.evaluator = evaluator
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.tree = tree
        self.max_rounds = max_rounds
        self.cycle_config = cycle_config or CycleConfig()
        self.global_best_round: Round | None = None
        self.global_best_score: float = 0.0

    def _check_unsupported_task_features(self) -> None:
        """Error if task provides features V2 doesn't support yet."""
        # V1 uses these to add context to prompts. V2 doesn't implement them yet.
        # Error early to prevent silent degradation.
        task_obj = getattr(self.task, "_task", self.task)

        if callable(getattr(task_obj, "get_code_format_text", None)):
            raise NotImplementedError(
                "Task provides get_code_format_text() but V2 executor doesn't support it yet. "
                "Use V1 executor or implement code_format support in V2."
            )

        if callable(getattr(task_obj, "get_baseline_targets_text", None)):
            raise NotImplementedError(
                "Task provides get_baseline_targets_text() but V2 executor doesn't support it yet. "
                "Use V1 executor or implement baseline_targets support in V2."
            )

    def run(self) -> Node | None:
        """Execute search, return best node."""
        self._check_unsupported_task_features()

        rounds_used = 0
        select_context = V1SelectContext(tree=self.tree)

        while rounds_used < self.max_rounds:
            logger.info("[CYCLE_START] rounds_used=%d/%d", rounds_used, self.max_rounds)

            propose_context = V1ProposeContext(
                tree=self.tree,
                round_idx=rounds_used,
            )
            self.world_model.propose(propose_context)

            selected = self.world_model.select(select_context)
            if not selected:
                logger.info("No more actions to select, stopping")
                break

            node = selected[0]
            node.status = "in_progress"
            logger.info(
                "[ACTION] Selected: %s", node.action.title if node.action else "unknown"
            )

            # Get action context from modular Node (authoritative source)
            action_ctx = self.world_model.get_action_context(node)

            cycle = self._run_cycle(
                node,
                action_ctx,
                rounds_remaining=self.max_rounds - rounds_used,
                rounds_used=rounds_used,
            )
            node.cycle = cycle
            node.status = "closed"

            update_context = V1UpdateContext(
                tree=self.tree,
                node=node,
                cycle=cycle,
                round_idx=rounds_used + len(cycle.rounds) - 1,
            )
            self.world_model.update(update_context)

            if cycle.best_round and cycle.best_round.score > self.global_best_score:
                self.global_best_round = cycle.best_round
                self.global_best_score = cycle.best_round.score

            rounds_used += len(cycle.rounds)
            logger.info(
                "[CYCLE_END] cycle_rounds=%d, total=%d", len(cycle.rounds), rounds_used
            )

        return self.tree.get_best_node()

    def _run_cycle(
        self,
        node: Node,
        action_ctx: dict[str, Any],
        rounds_remaining: int,
        rounds_used: int,
    ) -> Cycle:
        """Run multiple attempts on a single action with stagnation detection."""
        rounds: list[Round] = []
        best_score = 0.0
        best_speedup: float | None = None
        best_code = ""
        best_result: EvaluationResult | None = None
        no_improve = 0
        no_improve_over_base = 0

        # All metadata from action_ctx (which reads from modular Node)
        action_text = action_ctx.get(
            "action_text", node.action.title if node.action else ""
        )
        base_code = action_ctx.get("base_code", "")
        base_score = action_ctx.get("base_score", 0.0)
        base_result: EvaluationResult | None = action_ctx.get("base_result")
        parent_is_root = action_ctx.get("parent_is_root", False)

        max_attempts = rounds_remaining

        # Compute once - WM doesn't change during cycle attempts
        wm_section = self.world_model._get_prompt_section()

        current_code = ""
        for attempt in range(max_attempts):
            global_speedup = None
            if self.global_best_round:
                global_speedup = self.global_best_round.result.get_metrics().get(
                    "speedup_factor"
                )
            global_speedup_str = f"{global_speedup:.2f}x" if global_speedup else "-"
            logger.info(
                "[ATTEMPT] cycle_round=%d/%d | global_round=%d/%d | best=%.4f (%s) | no_improve=%d/%d | no_improve_over_base=%d/%d",
                attempt + 1,
                max_attempts,
                rounds_used + attempt + 1,
                self.max_rounds,
                self.global_best_score,
                global_speedup_str,
                no_improve,
                self.cycle_config.stagnation_rounds,
                no_improve_over_base,
                self.cycle_config.stagnation_rounds,
            )

            last_round = rounds[-1] if rounds else None
            last_result = last_round.result if last_round else None
            has_passed = best_score > 0
            trace_logs = last_round.result.get_log() if last_round else ""

            # Select better reference for debug prompts (V1 semantics)
            # V1 validates string is non-empty before score comparison
            effective_base_code = base_code
            use_best = (
                best_code
                and best_code.strip()
                and (best_score > base_score or base_score <= 0)
            )
            if use_best:
                effective_base_code = best_code

            # V1 semantics: select base_perf_eval based on which code was chosen
            if use_best and best_result:
                base_perf_eval = best_result
            elif base_result:
                base_perf_eval = base_result
            else:
                base_perf_eval = None

            # Build perf_summary from last_result + base_perf_eval (V1 semantics)
            perf_lines: list[str] = []
            perf_lines.extend(
                _get_perf_summary_lines(last_result, prefix="last_attempt")
            )
            perf_lines.extend(_get_perf_summary_lines(base_perf_eval, prefix="base"))
            perf_summary = "\n".join(perf_lines)

            prompt = self.prompt_builder.build(
                action_text=action_text,
                attempt=attempt,
                last_round=last_round,
                has_passed=has_passed,
                base_code=effective_base_code,
                trace_logs=trace_logs,
                current_code=current_code,
                perf_summary=perf_summary,
                parent_is_root=parent_is_root,
                max_debug_improve_rounds=self.cycle_config.max_debug_improve_rounds,
            )

            if wm_section:
                prompt = prompt + "\n\n" + wm_section

            logger.debug(
                prompt_color(
                    f"[CODE_PROMPT] ({len(prompt)} chars, ~{len(prompt) // 4} toks):\n\n{prompt}\n"
                )
            )

            code = self.llm(prompt)
            current_code = code

            logger.debug(
                response_color(f"[CODE_RESPONSE] ({len(code)} chars):\n\n{code}\n")
            )

            impl = self.task.create_impl(code)
            result = self.evaluator.evaluate(impl, context={"round_idx": attempt})

            # V1 semantics: -1.0 on failure, 1/latency on success
            if result.succeeded():
                metrics = result.get_metrics()
                latency = metrics.get("latency_ms")
                if isinstance(latency, (int, float)) and latency > 0:
                    score = 1.0 / latency
                else:
                    score = -1.0
            else:
                score = -1.0

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
                best_code = code
                best_result = result
                no_improve = 0
                logger.info(
                    "[IMPROVED] score=%.4f, speedup=%s, best_speedup=%s",
                    score,
                    speedup_str,
                    speedup_str,
                )
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

            # Track failure to beat base (V1 semantics: only when we have a passing
            # solution AND base exists)
            if best_score > 0 and base_score > 0:
                if best_score > base_score:
                    no_improve_over_base = 0
                else:
                    no_improve_over_base += 1

            if no_improve >= self.cycle_config.stagnation_rounds:
                logger.info("[STAGNATION] no improvement for %d rounds", no_improve)
                break
            if no_improve_over_base >= self.cycle_config.stagnation_rounds:
                logger.info(
                    "[STAGNATION] no improvement over base for %d rounds",
                    no_improve_over_base,
                )
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
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--no-reasoning-api",
        dest="use_reasoning_api",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--stagnation-rounds",
        type=int,
        default=5,
        help="Stagnation window (v1: --wm-stagnation-window)",
    )
    parser.add_argument(
        "--max-debug-improve-rounds",
        type=int,
        default=5,
        help="Max debug/improve rounds shown in prompts (v1: num_debug_and_improve_rounds)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Max difficulty (1-5) for action selection (v1: --wm-max-difficulty)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    for noisy_logger in ("httpcore", "httpx", "openai", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

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
        "max_retries": args.max_retries,
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
        stagnation_rounds=args.stagnation_rounds,
        max_debug_improve_rounds=args.max_debug_improve_rounds,
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
    if best_node:
        logger.info(
            "Best node: %s", best_node.action.title if best_node.action else "root"
        )
    global_best = executor.global_best_round
    if global_best:
        logger.info("Best score: %.4f", global_best.score)
        logger.info("Best round metrics: %s", global_best.result.get_metrics())
    else:
        logger.info("No successful solution found")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
