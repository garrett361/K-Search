"""Core search loop implementation."""

import logging
import time
from typing import Callable

from k_search.modular.artifacts import NoOpArtifactStore
from k_search.modular.config import MetricsConfig, SearchConfig, SearchResult
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.prompts import build_prompt
from k_search.modular.protocols import (
    ArtifactStore,
    Evaluator,
    EvaluationResult,
    MetricsTracker,
)
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.round import Round

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], str]


def _build_round_metrics(
    round_time_secs: float,
    score: float,
    result: EvaluationResult,
    best_score: float,
    prompt_toks: int,
    completion_toks: int,
    cumulative_prompt_toks: int,
    cumulative_completion_toks: int,
) -> dict[str, float | int]:
    metrics = {
        "round_time_secs": round_time_secs,
        "score": score,
        "is_success": int(result.is_success()),
        "best_score": best_score,
        "toks/prompt": prompt_toks,
        "toks/completion": completion_toks,
        "toks/total": prompt_toks + completion_toks,
        "toks/cumulative_prompt": cumulative_prompt_toks,
        "toks/cumulative_completion": cumulative_completion_toks,
        "toks/cumulative_total": cumulative_prompt_toks + cumulative_completion_toks,
    }

    for key, val in result.get_metrics().items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics[key] = val

    return metrics


def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
    metrics_trackers: list[MetricsTracker] | None = None,
    metrics_config: MetricsConfig | None = None,
    artifact_stores: list[ArtifactStore] | None = None,
) -> SearchResult:
    """Run simple sequential optimization loop.

    Args:
        task: Task definition with prompt, scoring, and implementation creation
        evaluator: Evaluates implementations
        llm: Callable that takes prompt and returns generated code
        config: Search configuration

    Returns:
        SearchResult with best implementation found
    """
    metrics_trackers = metrics_trackers or [NoOpMetricsTracker()]
    artifact_stores = artifact_stores or [NoOpArtifactStore()]

    metrics_config = metrics_config or MetricsConfig()

    cumulative_prompt_toks = 0
    cumulative_completion_toks = 0

    best_round: Round | None = None
    best_score = 0.0

    for round_idx in range(config.max_rounds):
        if best_round:
            metrics = best_round.result.get_metrics()
            speedup = metrics.get("speedup_factor", "N/A")
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | "
                f"Best: {best_score:.4f} (speedup: {speedup})"
            )
        else:
            logger.info(
                f"Round {round_idx + 1}/{config.max_rounds} | No solution found yet"
            )

        round_start = time.perf_counter()

        prompt = build_prompt(task, best_round)
        code = llm(prompt)
        impl = task.create_implementation(code)
        result = evaluator.evaluate(impl)
        score = task.scorer.score(result)

        prompt_toks = len(prompt) // metrics_config.chars_per_token
        completion_toks = len(code) // metrics_config.chars_per_token
        cumulative_prompt_toks += prompt_toks
        cumulative_completion_toks += completion_toks

        round_elapsed = time.perf_counter() - round_start

        round_ = Round(
            impl=impl,
            result=result,
            prompt=prompt,
            llm_response=code,
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            duration_secs=round_elapsed,
            score=score,
        )

        if score > best_score:
            best_round = round_
            best_score = score

        cumulative_total_toks = cumulative_prompt_toks + cumulative_completion_toks
        logger.info(
            f"Round {round_idx + 1} complete | "
            f"Score: {score:.4f} | "
            f"Time: {round_elapsed:.1f}s | "
            f"Toks: {cumulative_total_toks:,} ({cumulative_prompt_toks:,} prompt + {cumulative_completion_toks:,} completion)"
        )

        round_metrics = _build_round_metrics(
            round_time_secs=round_elapsed,
            score=score,
            result=result,
            best_score=best_score,
            prompt_toks=prompt_toks,
            completion_toks=completion_toks,
            cumulative_prompt_toks=cumulative_prompt_toks,
            cumulative_completion_toks=cumulative_completion_toks,
        )
        for tracker in metrics_trackers:
            tracker.log(round_metrics, step=round_idx)

        for store in artifact_stores:
            store.store(round_, round_idx)

    return SearchResult(
        impl=best_round.impl if best_round else None,
        score=best_score,
        result=best_round.result if best_round else None,
        rounds_completed=config.max_rounds,
    )
