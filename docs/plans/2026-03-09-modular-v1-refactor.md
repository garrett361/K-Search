# Modular V1 Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `k_search/modular/loop.py` and `run_search_v2.py` to `scripts/gpu_mode_modular_v1/` as a self-contained script

**Architecture:** Inline `run_search`, `_build_round_metrics`, and `build_prompt` into `scripts/gpu_mode_modular_v1/run.py`. Keep importing shared infrastructure (config, metrics, artifacts, adapters) from `k_search/modular/`. Move all related tests to `scripts/gpu_mode_modular_v1/test_run.py`.

**Tech Stack:** Python 3.12, pytest, openai SDK

---

## Task 1: Create scripts/gpu_mode_modular_v1 Directory Structure

**Files:**
- Create: `scripts/gpu_mode_modular_v1/__init__.py`

**Step 1: Create the directory and empty __init__.py**

```bash
mkdir -p scripts/gpu_mode_modular_v1
touch scripts/gpu_mode_modular_v1/__init__.py
```

**Step 2: Commit**

```bash
git add scripts/gpu_mode_modular_v1/__init__.py
git commit -m "chore: create scripts/gpu_mode_modular_v1 directory"
```

---

## Task 2: Create run.py with Inlined Functions

**Files:**
- Create: `scripts/gpu_mode_modular_v1/run.py`
- Reference: `k_search/modular/loop.py`, `k_search/modular/prompts.py`, `run_search_v2.py`

**Step 1: Create run.py**

Create `scripts/gpu_mode_modular_v1/run.py` with the following content:

```python
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
```

**Step 2: Verify syntax**

Run: `python -m py_compile scripts/gpu_mode_modular_v1/run.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add scripts/gpu_mode_modular_v1/run.py
git commit -m "feat(modular): add self-contained run.py to scripts/gpu_mode_modular_v1"
```

---

## Task 3: Create test_run.py with Merged Tests

**Files:**
- Create: `scripts/gpu_mode_modular_v1/test_run.py`
- Reference: `tests/modular/test_loop.py`, `tests/modular/test_e2e_search.py`, `tests/modular/test_metrics.py:93-215`

**Step 1: Create test_run.py**

Create `scripts/gpu_mode_modular_v1/test_run.py` with the following content:

```python
"""Tests for GPU mode modular V1 entry point."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from k_search.modular import SearchConfig, SearchResult, Round
from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

from scripts.gpu_mode_modular_v1.run import (
    build_prompt,
    run_search,
    strip_markdown_fences,
    _build_round_metrics,
)


def make_eval_result_mock(succeeded: bool = True, metrics: dict | None = None) -> Mock:
    """Create a mock EvaluationResult."""
    result = Mock()
    result.succeeded.return_value = succeeded
    result.get_metrics.return_value = metrics or {"speedup_factor": 1.0}
    result.get_log.return_value = "test log"
    return result


def make_impl_mock(name: str = "test_impl") -> Mock:
    """Create a mock Implementation."""
    impl = Mock()
    impl.name = name
    return impl


def make_task_mock(name: str = "test_task") -> Mock:
    """Create a mock TaskDefinition."""
    task = Mock()
    task.name = name
    task.get_prompt_text.return_value = "Generate optimized kernel code"
    task.scorer.score.return_value = 0.5
    task.feedback_provider.for_codegen.return_value = "Previous attempt feedback"

    impl_counter = [0]

    def create_impl(llm_output: str) -> Mock:
        impl = make_impl_mock(f"{name}_r{impl_counter[0]}")
        impl_counter[0] += 1
        return impl

    task.create_impl.side_effect = create_impl
    return task


def make_evaluator_mock(result: Mock | None = None) -> Mock:
    """Create a mock Evaluator."""
    evaluator = Mock()
    evaluator.evaluate.return_value = result or make_eval_result_mock()
    return evaluator


def stub_llm(prompt: str) -> str:
    return "def custom_kernel(): pass"


class TestRunSearch:
    def test_completes_max_rounds(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.rounds_completed == 3
        assert evaluator.evaluate.call_count == 3
        assert task.scorer.score.call_count == 3

    def test_single_round(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.rounds_completed == 1
        assert evaluator.evaluate.call_count == 1

    def test_tracks_best_score_improvement(self):
        task = make_task_mock()
        scores = [0.1, 0.5, 0.3]
        task.scorer.score.side_effect = scores

        results = [
            make_eval_result_mock(metrics={"speedup_factor": 1.1}),
            make_eval_result_mock(metrics={"speedup_factor": 1.5}),
            make_eval_result_mock(metrics={"speedup_factor": 1.3}),
        ]
        evaluator = Mock()
        evaluator.evaluate.side_effect = results

        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.5
        assert result.result is not None
        assert result.result.get_metrics()["speedup_factor"] == 1.5

    def test_tracks_best_score_first_best(self):
        task = make_task_mock()
        scores = [0.9, 0.5, 0.3]
        task.scorer.score.side_effect = scores

        results = [
            make_eval_result_mock(metrics={"speedup_factor": 2.0}),
            make_eval_result_mock(metrics={"speedup_factor": 1.5}),
            make_eval_result_mock(metrics={"speedup_factor": 1.3}),
        ]
        evaluator = Mock()
        evaluator.evaluate.side_effect = results

        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.9
        assert result.result is not None
        assert result.result.get_metrics()["speedup_factor"] == 2.0

    def test_first_round_no_feedback(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=1)

        run_search(task, evaluator, capture_llm, config)

        assert len(prompts_received) == 1
        assert "Previous attempt feedback" not in prompts_received[0]
        task.feedback_provider.for_codegen.assert_not_called()

    def test_subsequent_rounds_include_feedback(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=3)

        run_search(task, evaluator, capture_llm, config)

        assert len(prompts_received) == 3
        assert "Previous attempt feedback" not in prompts_received[0]
        assert "Previous attempt feedback" in prompts_received[1]
        assert "Previous attempt feedback" in prompts_received[2]

    def test_returns_search_result_type(self):
        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)

        result = run_search(task, evaluator, stub_llm, config)

        assert isinstance(result, SearchResult)
        assert result.impl is not None
        assert result.result is not None
        assert result.score == 0.5

    def test_impl_naming_uses_task_name(self):
        task = make_task_mock(name="custom_task")
        captured_impls = []

        def capture_evaluate(impl, **kwargs):
            captured_impls.append(impl)
            return make_eval_result_mock()

        evaluator = Mock()
        evaluator.evaluate.side_effect = capture_evaluate
        config = SearchConfig(max_rounds=2)

        run_search(task, evaluator, stub_llm, config)

        assert len(captured_impls) == 2
        assert "custom_task_r0" in captured_impls[0].name
        assert "custom_task_r1" in captured_impls[1].name

    def test_zero_scores_handled(self):
        task = make_task_mock()
        task.scorer.score.side_effect = [0.0, 0.5, 0.0]

        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)

        result = run_search(task, evaluator, stub_llm, config)

        assert result.score == 0.5
        assert result.rounds_completed == 3

    def test_llm_receives_task_prompt(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Custom kernel specification"
        evaluator = make_evaluator_mock()
        prompts_received = []

        def capture_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return "def custom_kernel(): pass"

        config = SearchConfig(max_rounds=1)

        run_search(task, evaluator, capture_llm, config)

        assert "Custom kernel specification" in prompts_received[0]


class TestSearchConfig:
    def test_default_values(self):
        config = SearchConfig()
        assert config.max_rounds == 10
        assert config.timeout_secs is None

    def test_custom_max_rounds(self):
        config = SearchConfig(max_rounds=5)
        assert config.max_rounds == 5

    def test_custom_timeout(self):
        config = SearchConfig(timeout_secs=300)
        assert config.timeout_secs == 300


class TestSearchResult:
    def test_default_rounds_completed(self):
        result = SearchResult(impl=None, score=0.0, result=None)
        assert result.rounds_completed == 0

    def test_all_fields(self):
        mock_impl = Mock()
        mock_eval_result = Mock()
        result = SearchResult(
            impl=mock_impl,
            score=0.75,
            result=mock_eval_result,
            rounds_completed=5,
        )
        assert result.impl is mock_impl
        assert result.score == 0.75
        assert result.result is mock_eval_result
        assert result.rounds_completed == 5


class TestBuildPrompt:
    def test_first_round_returns_task_prompt(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Task specification here"

        result = build_prompt(task, None)

        assert result == "Task specification here"
        task.get_prompt_text.assert_called_once()

    def test_with_round_includes_feedback(self):
        task = make_task_mock()
        task.get_prompt_text.return_value = "Task spec"
        task.feedback_provider.for_codegen.return_value = "Error on line 42"

        mock_impl = Mock()
        mock_result = make_eval_result_mock()
        round_ = Round(
            impl=mock_impl,
            result=mock_result,
            prompt="test",
            llm_response="test",
            prompt_tokens=0,
            completion_tokens=0,
            duration_secs=0.0,
            score=0.0,
        )

        result = build_prompt(task, round_)

        assert "Task spec" in result
        assert "Error on line 42" in result
        task.feedback_provider.for_codegen.assert_called_once_with(round_)


class TestCreateImpl:
    def test_increments_counter(self):
        mock_task = MagicMock()
        mock_task.name = "test_task"

        with patch("k_search.modular.adapters.gpu_mode.task_definition._load_reference_module"):
            task_def = GpuModeTriMulTaskDefinition(mock_task)

        assert task_def.create_impl("code1").name == "test_task_r0"
        assert task_def.create_impl("code2").name == "test_task_r1"
        assert task_def._impl_counter == 2


class TestStripMarkdownFences:
    def test_no_fences_unchanged(self):
        code = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == code

    def test_strips_python_fences(self):
        code = "```python\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_triton_fences(self):
        code = "```triton\n@triton.jit\ndef kernel():\n    pass\n```"
        expected = "@triton.jit\ndef kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_strips_bare_fences(self):
        code = "```\ndef custom_kernel():\n    pass\n```"
        expected = "def custom_kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_extracts_first_fenced_block(self):
        code = (
            "Here's the code:\n```python\ndef kernel():\n    pass\n```\nAnd more text."
        )
        expected = "def kernel():\n    pass"
        assert strip_markdown_fences(code) == expected

    def test_handles_empty_string(self):
        assert strip_markdown_fences("") == ""
        assert strip_markdown_fences(None) is None

    def test_handles_unclosed_fence(self):
        code = "```python\ndef kernel():\n    pass"
        result = strip_markdown_fences(code)
        assert result is not None
        assert "```" not in result
        assert "def kernel():" in result


class TestBuildRoundMetrics:
    @pytest.fixture
    def mock_eval_result(self):
        result = Mock()
        result.succeeded.return_value = True
        result.get_metrics.return_value = {"speedup_factor": 1.5}
        return result

    def test_builds_correct_structure(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        assert metrics == {
            "round_time_secs": 3.5,
            "score": 0.6,
            "succeeded": 1,
            "best_score": 0.7,
            "toks/prompt": 200,
            "toks/completion": 100,
            "toks/total": 300,
            "toks/cumulative_prompt": 800,
            "toks/cumulative_completion": 400,
            "toks/cumulative_total": 1200,
            "speedup_factor": 1.2,
        }

    def test_is_success_converts_to_int(self, mock_eval_result):
        mock_eval_result.succeeded.return_value = False
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.0,
            result=mock_eval_result,
            best_score=0.0,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["succeeded"] == 0

    def test_includes_numeric_eval_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 2.5,
            "latency_ms": 10.3,
            "memory_mb": 512,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 2.5
        assert metrics["latency_ms"] == 10.3
        assert metrics["memory_mb"] == 512

    def test_excludes_non_numeric_and_bool_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 1.5,
            "status": "success",
            "error_msg": None,
            "passed": True,
            "has_errors": False,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 1.5
        assert "status" not in metrics
        assert "error_msg" not in metrics
        assert "passed" not in metrics
        assert "has_errors" not in metrics

    def test_returns_expected_keys(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        expected_keys = {
            "round_time_secs",
            "score",
            "succeeded",
            "best_score",
            "toks/prompt",
            "toks/completion",
            "toks/total",
            "toks/cumulative_prompt",
            "toks/cumulative_completion",
            "toks/cumulative_total",
            "speedup_factor",
        }
        assert set(metrics.keys()) == expected_keys


CAUSAL_CONV1D_DIR = (
    Path(__file__).parent.parent.parent
    / "k_search"
    / "tasks"
    / "gpu_mode"
    / "causal_conv1d"
)


@pytest.mark.cuda
@pytest.mark.cuda_subprocess
class TestE2ESearch:
    """E2E tests validating modular loop with real GPU evaluation."""

    @pytest.fixture
    def task_dir(self) -> Path:
        return CAUSAL_CONV1D_DIR

    @pytest.fixture
    def valid_triton_code(self, task_dir: Path) -> str:
        submission_path = task_dir / "submission.py"
        return submission_path.read_text()

    def test_single_round_with_valid_code(self, task_dir: Path, valid_triton_code: str):
        """Run single search round with valid Triton code, verify score and metrics."""
        gpu_task = GpuModeTriMulTask(task_dir=task_dir)
        task_def = GpuModeTriMulTaskDefinition(gpu_task)
        evaluator = GpuModeEvaluator(gpu_task)

        def mock_llm(prompt: str) -> str:
            return valid_triton_code

        config = SearchConfig(max_rounds=1)
        result = run_search(task_def, evaluator, mock_llm, config)

        assert result.rounds_completed == 1
        assert result.impl is not None
        assert result.result is not None
        assert result.score > 0, "Valid code should produce positive score"

        metrics = result.result.get_metrics()
        assert "speedup_factor" in metrics or "latency_ms" in metrics

    def test_two_rounds_best_tracked(self, task_dir: Path, valid_triton_code: str):
        """Run two rounds, verify best result is tracked correctly."""
        gpu_task = GpuModeTriMulTask(task_dir=task_dir)
        task_def = GpuModeTriMulTaskDefinition(gpu_task)
        evaluator = GpuModeEvaluator(gpu_task)

        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return valid_triton_code

        config = SearchConfig(max_rounds=2)
        result = run_search(task_def, evaluator, mock_llm, config)

        assert result.rounds_completed == 2
        assert call_count == 2
        assert result.score > 0
        assert result.result is not None
```

**Step 2: Run tests to verify**

Run: `pytest scripts/gpu_mode_modular_v1/test_run.py -v --ignore-glob='*cuda*' -k 'not E2E'`
Expected: All tests pass (E2E tests skipped since they need GPU)

**Step 3: Commit**

```bash
git add scripts/gpu_mode_modular_v1/test_run.py
git commit -m "test(modular): add merged tests to scripts/gpu_mode_modular_v1"
```

---

## Task 4: Update k_search/modular/__init__.py

**Files:**
- Modify: `k_search/modular/__init__.py`

**Step 1: Remove loop imports**

Edit `k_search/modular/__init__.py` to remove the `run_search` and `LLMCall` imports and exports.

Remove line 9:
```python
from k_search.modular.loop import LLMCall, run_search
```

Remove from `__all__` list (lines 47-48):
```python
    "LLMCall",
    "run_search",
```

**Step 2: Run import check**

Run: `python -c "from k_search.modular import SearchConfig, ArtifactConfig, Round"`
Expected: No errors

**Step 3: Commit**

```bash
git add k_search/modular/__init__.py
git commit -m "refactor(modular): remove run_search and LLMCall exports from __init__"
```

---

## Task 5: Update tests/modular/test_metrics.py

**Files:**
- Modify: `tests/modular/test_metrics.py`

**Step 1: Remove TestBuildRoundMetrics class**

Edit `tests/modular/test_metrics.py` to remove:
- Line 9: `from k_search.modular.loop import _build_round_metrics`
- Lines 93-215: The entire `TestBuildRoundMetrics` class

Keep only:
- `TestWandbMetricsTracker`
- `TestCreateMetricsTrackers`
- `TestLocalMetricsTracker`

**Step 2: Run remaining tests**

Run: `pytest tests/modular/test_metrics.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/modular/test_metrics.py
git commit -m "test(modular): remove TestBuildRoundMetrics (moved to scripts/)"
```

---

## Task 6: Delete Old Files

**Files:**
- Delete: `k_search/modular/loop.py`
- Delete: `k_search/modular/prompts.py`
- Delete: `run_search_v2.py`
- Delete: `tests/modular/test_loop.py`
- Delete: `tests/modular/test_e2e_search.py`

**Step 1: Delete files**

```bash
git rm k_search/modular/loop.py
git rm k_search/modular/prompts.py
git rm run_search_v2.py
git rm tests/modular/test_loop.py
git rm tests/modular/test_e2e_search.py
```

**Step 2: Run all tests to verify**

Run: `pytest tests/ scripts/gpu_mode_modular_v1/test_run.py scripts/gpu_mode_simple_linear_executor/test_run.py -v --ignore-glob='*cuda*' -k 'not E2E and not cuda'`
Expected: All tests pass

**Step 3: Commit**

```bash
git commit -m "refactor(modular): delete loop.py, prompts.py, run_search_v2.py and old tests"
```

---

## Task 7: Update runme.yaml in verl-experiments-k-search

**Files:**
- Modify: `../k_search_expr/runme.yaml` (in the parent repo, not the worktree)

**Step 1: Update run_search_modular recipe**

Change line 194 from:
```yaml
"$VENV/bin/python" run_search_v2.py \
```

To:
```yaml
"$VENV/bin/python" scripts/gpu_mode_modular_v1/run.py \
```

**Step 2: Commit in parent repo**

```bash
cd /proj/data-eng/goon/flim/verl-experiments-k-search
git add k_search_expr/runme.yaml
git commit -m "refactor(k-search): update run_search_modular to use scripts/gpu_mode_modular_v1/run.py"
```

---

## Task 8: Final Verification and Cleanup

**Step 1: Run full test suite in worktree**

```bash
cd /proj/data-eng/goon/flim/verl-experiments-k-search/K-Search/.claude/worktrees/modular-v1-refactor
pytest tests/ scripts/ -v --ignore-glob='*cuda*' -k 'not E2E and not cuda'
```
Expected: All tests pass

**Step 2: Run ruff checks**

```bash
ruff check scripts/gpu_mode_modular_v1/
ruff format --check scripts/gpu_mode_modular_v1/
```
Expected: No errors

**Step 3: Verify CLI works**

```bash
python scripts/gpu_mode_modular_v1/run.py --help
```
Expected: Help message displays

**Step 4: Update CLAUDE.md entry points**

Edit `CLAUDE.md` to update the entry points section from:
```
Entry points:
- `run_search_v2.py` - Modular search loop
```

To:
```
Entry points:
- `scripts/gpu_mode_modular_v1/run.py` - Modular search loop
```

**Step 5: Commit CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "docs: update entry points in CLAUDE.md"
```

---

## Summary of Changes

**Created:**
- `scripts/gpu_mode_modular_v1/__init__.py`
- `scripts/gpu_mode_modular_v1/run.py`
- `scripts/gpu_mode_modular_v1/test_run.py`

**Modified:**
- `k_search/modular/__init__.py` (removed `run_search`, `LLMCall` exports)
- `tests/modular/test_metrics.py` (removed `TestBuildRoundMetrics`)
- `CLAUDE.md` (updated entry points)
- `../k_search_expr/runme.yaml` (updated script path)

**Deleted:**
- `k_search/modular/loop.py`
- `k_search/modular/prompts.py`
- `run_search_v2.py`
- `tests/modular/test_loop.py`
- `tests/modular/test_e2e_search.py`
