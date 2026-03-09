#!/usr/bin/env python3
"""V2 Search Loop entry point - modular framework with metrics and artifacts.

Inlines the core search loop for visibility while importing shared infrastructure.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import openai

from k_search.modular import SearchConfig, ArtifactConfig
from k_search.modular.artifacts import NoOpArtifactStore, create_artifact_stores
from k_search.modular.config import MetricsConfig, SearchResult, build_run_config
from k_search.modular.metrics import NoOpMetricsTracker, create_metrics_trackers
from k_search.modular.protocols import (
    ArtifactStore,
    Evaluator,
    EvaluationResult,
    MetricsTracker,
    TaskDefinition,
)
from k_search.modular.world.round import Round
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LLMCall = Callable[[str], str]


def strip_markdown_fences(code: str | None) -> str | None:
    """Strip markdown code fences from LLM output."""
    if not code or "```" not in code:
        return code

    m = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```", code)
    if m:
        return (m.group(1) or "").strip()

    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        code = "\n".join(lines)

    if code.endswith("```"):
        lines = code.split("\n")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.replace("```", "").strip()


def build_prompt(task: TaskDefinition, last_round: Round | None) -> str:
    """Build prompt for next code generation round."""
    base = task.get_prompt_text()
    if last_round:
        feedback = task.feedback_provider.for_codegen(last_round)
        return f"{base}\n\n{feedback}"
    return base


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
        "succeeded": int(result.succeeded()),
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
    """Run simple sequential optimization loop."""
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
        impl = task.create_impl(code)
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


def create_llm_call(
    client: openai.OpenAI,
    model_name: str,
    use_reasoning_api: bool = True,
    reasoning_effort: str = "medium",
):
    """Create LLM callable for search loop."""

    def llm_call(prompt: str) -> str:
        if use_reasoning_api:
            response = client.responses.create(
                model=model_name,
                input=prompt,
                reasoning={"effort": reasoning_effort},  # type: ignore[arg-type]
            )
            return (response.output_text or "").strip()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    return llm_call


def main():
    parser = argparse.ArgumentParser(description="Run V2 search loop")
    parser.add_argument(
        "--task", required=True, help="Task name (e.g., causal_conv1d, trimul)"
    )
    parser.add_argument("--language", default="triton", choices=["triton", "cuda"])
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key", default=None, help="API key; if omitted, uses LLM_API_KEY env var"
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--no-reasoning-api",
        dest="use_reasoning_api",
        action="store_false",
        default=True,
        help="Disable reasoning API (use standard chat completions instead)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for responses API",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Wandb project name (default: inferred by wandb)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Wandb run name (default: generated by wandb)",
    )
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help="Directory for wandb local files (default: cwd)",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Wandb group name (e.g., experiment name)",
    )
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated wandb tags (e.g., baseline,causal_conv1d,triton)",
    )
    parser.add_argument(
        "--artifact-output-dir",
        default=None,
        help="Directory to save artifacts (code, metadata)",
    )
    parser.add_argument(
        "--artifact-mode",
        default="successes",
        choices=["successes", "all"],
        help="Which artifacts to store: 'successes' (default) or 'all'",
    )

    args = parser.parse_args()

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
        logger.error(f"Task directory not found: {task_dir}")
        sys.exit(1)

    logger.info(f"Loading task: {args.task}")
    gpu_task = GpuModeTriMulTask(task_dir=task_dir)
    task_def = GpuModeTriMulTaskDefinition(gpu_task, language=args.language)
    evaluator = GpuModeEvaluator(gpu_task)

    client_kwargs = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url is not None:
        client_kwargs["base_url"] = args.base_url
    if (rits_api_key := os.getenv("RITS_API_KEY")) is not None:
        client_kwargs["default_headers"] = {"RITS_API_KEY": rits_api_key}

    client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
    llm = create_llm_call(
        client,
        args.model_name,
        use_reasoning_api=args.use_reasoning_api,
        reasoning_effort=args.reasoning_effort,
    )

    config = SearchConfig(max_rounds=args.max_rounds)
    metrics_config = MetricsConfig(wandb=args.wandb)
    artifact_config = ArtifactConfig(
        output_dir=args.artifact_output_dir,
        only_store_successes=(args.artifact_mode == "successes"),
        wandb=args.wandb,
    )

    wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None

    run_config = build_run_config(
        run_id=args.run_name
        or f"{args.task}-{args.model_name.replace('/', '-')}-r{args.max_rounds}",
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
        search_config=config,
        metrics_config=metrics_config,
        artifact_config=artifact_config,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_group=args.wandb_group,
        wandb_tags=wandb_tags,
        task=args.task,
        language=args.language,
    )

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=args.wandb_dir,
            config=run_config,
            group=args.wandb_group,
            tags=wandb_tags,
        )
        logger.info(f"Wandb enabled: project={args.wandb_project}, run={args.run_name}")

    metrics_trackers = create_metrics_trackers(
        metrics_config,
        output_dir=args.artifact_output_dir,
        run_config=run_config,
    )
    artifact_stores = create_artifact_stores(artifact_config)

    logger.info(
        f"Starting V2 search: max_rounds={args.max_rounds}, model={args.model_name}"
    )

    result = run_search(
        task_def,
        evaluator,
        llm,
        config,
        metrics_trackers=metrics_trackers,
        artifact_stores=artifact_stores,
    )

    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info(f"Rounds completed: {result.rounds_completed}")
    logger.info(f"Best score: {result.score:.4f}")
    if result.result:
        metrics = result.result.get_metrics()
        logger.info(f"Speedup factor: {metrics.get('speedup_factor', 'N/A')}")
        logger.info(f"Status: {metrics.get('status', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
