"""Generic kernel evaluator orchestrator for custom Triton tasks."""

from __future__ import annotations

import logging
from typing import Any, Callable

from k_search.tasks.task_base import EvalResult, Solution

from .aggregation import AggregationStrategy, MeanAggregation
from .benchmarking import BenchmarkConfig, BenchmarkHarness, clear_l2_cache
from .correctness import CorrectnessConfig, check_correctness


logger = logging.getLogger(__name__)


class GenericKernelEvaluator:
    """
    Reusable evaluation orchestrator for custom kernel tasks.

    Handles the standard flow:
    1. For each workload configuration, generate test inputs
    2. Run candidate kernel and reference kernel
    3. Check correctness (if reference provided)
    4. Benchmark passed workloads
    5. Aggregate metrics across workloads
    6. Return standardized EvalResult

    Backends plug in:
    - BenchmarkHarness: how to execute/benchmark the kernel
    - CorrectnessConfig: tolerance settings
    - AggregationStrategy: how to combine multi-config results
    """

    def __init__(
        self,
        harness: BenchmarkHarness,
        correctness_config: CorrectnessConfig | None = None,
        aggregation: AggregationStrategy | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            harness: Backend-specific benchmark runner
            correctness_config: Tolerance settings (default: rtol=1e-2, atol=1e-2)
            aggregation: Metric aggregation strategy (default: MeanAggregation)
        """
        self._harness = harness
        self._correctness_config = correctness_config or CorrectnessConfig()
        self._aggregation = aggregation or MeanAggregation()

    def evaluate(
        self,
        solution: Solution,
        reference_fn: Callable | None = None,
        workloads: list[dict[str, Any]] | None = None,
        input_generator: Callable[[dict], dict[str, Any]] | None = None,
    ) -> EvalResult:
        """
        Evaluate a solution across multiple workload configurations.

        Args:
            solution: Candidate kernel solution to evaluate
            reference_fn: Optional reference implementation for correctness checking
            workloads: List of workload configurations (dicts with parameters)
            input_generator: Function that takes a workload config and returns
                           input tensors dict

        Returns:
            EvalResult with aggregated metrics
        """
        if not workloads:
            return EvalResult(
                status="failed",
                log_excerpt="No workload configurations provided",
            )

        if input_generator is None:
            return EvalResult(
                status="failed",
                log_excerpt="No input generator provided",
            )

        passed_latencies: list[float] = []
        passed_speedups: list[float] = []
        failed_workloads: list[str] = []
        correctness_errors: list[str] = []

        for workload_idx, workload_config in enumerate(workloads):
            try:
                inputs = input_generator(workload_config)
            except Exception as e:
                error_msg = f"Workload {workload_idx} input generation failed: {e}"
                logger.warning(error_msg)
                failed_workloads.append(error_msg)
                continue

            try:
                candidate_output = self._run_candidate(solution, inputs)
            except Exception as e:
                error_msg = f"Workload {workload_idx} candidate execution failed: {e}"
                logger.warning(error_msg)
                failed_workloads.append(error_msg)
                continue

            if reference_fn is not None:
                try:
                    reference_output = reference_fn(**inputs)
                except Exception as e:
                    error_msg = f"Workload {workload_idx} reference execution failed: {e}"
                    logger.warning(error_msg)
                    failed_workloads.append(error_msg)
                    continue

                passed, details = check_correctness(
                    candidate_output,
                    reference_output,
                    self._correctness_config,
                )

                if not passed:
                    error_msg = (
                        f"Workload {workload_idx} correctness check failed: "
                        f"{details.get('error_message', 'Unknown error')}"
                    )
                    logger.warning(error_msg)
                    correctness_errors.append(error_msg)
                    continue

            try:
                benchmark_config = BenchmarkConfig()
                clear_l2_cache()

                benchmark_results = self._harness.run(
                    kernel_fn=lambda **kwargs: self._run_candidate(solution, kwargs),
                    inputs=inputs,
                    config=benchmark_config,
                )

                latency_ms = benchmark_results.get("latency_ms")
                if latency_ms is None or latency_ms <= 0:
                    failed_workloads.append(f"Workload {workload_idx}: invalid latency {latency_ms}")
                    continue

                passed_latencies.append(latency_ms)

                if reference_fn is not None:
                    clear_l2_cache()
                    reference_results = self._harness.run(
                        kernel_fn=reference_fn,
                        inputs=inputs,
                        config=benchmark_config,
                    )
                    ref_latency_ms = reference_results.get("latency_ms")
                    if ref_latency_ms and ref_latency_ms > 0:
                        speedup = ref_latency_ms / latency_ms
                        passed_speedups.append(speedup)

            except Exception as e:
                error_msg = f"Workload {workload_idx} benchmark failed: {e}"
                logger.warning(error_msg)
                failed_workloads.append(error_msg)
                continue

        if not passed_latencies:
            all_errors = failed_workloads + correctness_errors
            return EvalResult(
                status="failed",
                log_excerpt="\n".join(all_errors[:10]) if all_errors else "All workloads failed",
            )

        aggregated = self._aggregation.aggregate(
            latencies=passed_latencies,
            speedups=passed_speedups if passed_speedups else None,
        )

        num_passed = len(passed_latencies)
        num_total = len(workloads)

        log_lines = [
            f"Passed: {num_passed}/{num_total} workloads",
            f"Mean latency: {aggregated['mean_latency_ms']:.4f} ms",
        ]
        if passed_speedups:
            log_lines.append(f"Mean speedup: {aggregated['mean_speedup']:.3f}x")

        if failed_workloads:
            log_lines.append(f"Failed workloads ({len(failed_workloads)}): {failed_workloads[:3]}")
        if correctness_errors:
            log_lines.append(f"Correctness errors ({len(correctness_errors)}): {correctness_errors[:3]}")

        return EvalResult(
            status="passed" if num_passed == num_total else "partial",
            latency_ms=aggregated["mean_latency_ms"],
            speedup_factor=aggregated.get("mean_speedup"),
            mean_vs_baseline_factor=aggregated.get("mean_speedup"),
            log_excerpt="\n".join(log_lines),
            metrics={
                "num_passed": num_passed,
                "num_total": num_total,
                "score": aggregated["score"],
                "passed_latencies": passed_latencies,
                "passed_speedups": passed_speedups,
            },
        )

    def _run_candidate(self, solution: Solution, inputs: dict[str, Any]) -> Any:
        """
        Execute candidate kernel from solution.

        This is a placeholder - subclasses/harnesses should override
        or provide proper execution logic.
        """
        raise NotImplementedError("Subclass must implement _run_candidate or use harness.run")
