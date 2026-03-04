"""Core search loop implementation."""

import logging
import time
from typing import Callable

from k_search.search_v2.config import SearchConfig, SearchResult
from k_search.search_v2.prompts import build_prompt, create_implementation
from k_search.task_framework.protocols.evaluator import Evaluator
from k_search.task_framework.protocols.task_definition import TaskDefinition
from k_search.task_framework.types import EvalOutcome

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], str]


def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
) -> SearchResult:
    """Run simple sequential optimization loop.

    Args:
        task: Task definition with prompt and scoring
        evaluator: Evaluates implementations
        llm: Callable that takes prompt and returns generated code
        config: Search configuration

    Returns:
        SearchResult with best implementation found
    """
    best_impl = None
    best_score = float("-inf")
    best_result = None

    for round_idx in range(config.max_rounds):
        if best_result:
            metrics = best_result.get_metrics()
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

        last_outcome = None
        if best_impl and best_result:
            last_outcome = EvalOutcome(impl=best_impl, result=best_result)

        prompt = build_prompt(task, last_outcome)
        code = llm(prompt)
        impl = create_implementation(code, round_idx, task_name=task.name)
        result = evaluator.evaluate(impl)
        score = task.scorer.score(result)

        if score > best_score:
            best_impl = impl
            best_score = score
            best_result = result

        round_elapsed = time.perf_counter() - round_start
        logger.info(
            f"Round {round_idx + 1} complete | "
            f"Score: {score:.4f} | "
            f"Time: {round_elapsed:.1f}s"
        )

    return SearchResult(
        impl=best_impl,
        score=best_score,
        result=best_result,
        rounds_completed=config.max_rounds,
    )
