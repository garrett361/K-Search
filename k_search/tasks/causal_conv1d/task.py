"""Causal Conv1d task implementation for K-Search."""

from __future__ import annotations

import logging
from typing import Any

import torch

from k_search.tasks.custom_triton import (
    CorrectnessConfig,
    GenericKernelEvaluator,
    MeanAggregation,
)
from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
)

from .harness import CausalConv1dHarness
from .reference import create_reference_inputs, fla_causal_conv1d_reference
from .spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON


logger = logging.getLogger(__name__)


class CausalConv1dTask:
    """
    K-Search task for optimizing causal_conv1d_fwd kernels.

    Uses the generic custom_triton evaluation framework with:
    - FLA (Flash Linear Attention) as reference implementation
    - 4 representative training workload configurations
    - Triton kernel compilation and benchmarking
    """

    WORKLOADS = [
        {"B": 2, "T": 4096, "D": 2048, "W": 4, "activation": "silu"},
        {"B": 4, "T": 4096, "D": 4096, "W": 4, "activation": "silu"},
        {"B": 8, "T": 8192, "D": 2048, "W": 4, "activation": "silu"},
        {"B": 8, "T": 8192, "D": 4096, "W": 4, "activation": "silu"},
    ]

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self._dtype = dtype
        self._device = device

        self._harness = CausalConv1dHarness()
        self._evaluator = GenericKernelEvaluator(
            harness=self._harness,
            correctness_config=CorrectnessConfig(rtol=2e-2, atol=2e-2),
            aggregation=MeanAggregation(),
        )

    @property
    def name(self) -> str:
        return "causal_conv1d"

    def _input_generator(self, workload: dict[str, Any]) -> dict[str, Any]:
        """
        Generate test inputs for a workload configuration.

        Args:
            workload: Configuration dict with B, T, D, W, activation

        Returns:
            Dictionary of input tensors
        """
        inputs = create_reference_inputs(
            B=workload["B"],
            T=workload["T"],
            D=workload["D"],
            W=workload["W"],
            dtype=self._dtype,
            device=self._device,
            with_bias=True,
            with_residual=False,
        )

        inputs["activation"] = workload.get("activation", "silu")

        return inputs

    def get_definition_text(self, language: str | None = None) -> str:
        """
        Get kernel specification text for LLM prompts.

        Args:
            language: Programming language (ignored, always Triton)

        Returns:
            Kernel specification text
        """
        return CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def get_solution(self, solution_name: str) -> Solution | None:
        """
        Get a named solution (e.g., baseline).

        Args:
            solution_name: Name of solution to retrieve

        Returns:
            Solution object or None if not found
        """
        if solution_name.lower() in ("baseline", "fla", "reference"):
            try:
                from fla.modules.convolution import causal_conv1d_fwd_kernel

                kernel_code = ""
                if hasattr(causal_conv1d_fwd_kernel, "__code__"):
                    import inspect

                    kernel_code = inspect.getsource(causal_conv1d_fwd_kernel)

                return Solution(
                    name="fla_baseline",
                    definition="causal_conv1d",
                    author="flash-linear-attention",
                    spec=BuildSpec(
                        language=SupportedLanguages.TRITON,
                        target_hardware=["cuda"],
                        entry_point="causal_conv1d_fwd.py::causal_conv1d_fwd_kernel",
                    ),
                    sources=[
                        SourceFile(
                            path="causal_conv1d_fwd.py",
                            content=kernel_code if kernel_code else "# FLA baseline (source not extractable)",
                        )
                    ],
                    description="Flash Linear Attention baseline implementation",
                )
            except Exception as e:
                logger.warning(f"Failed to load FLA baseline solution: {e}")
                return None

        return None

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        """
        Evaluate a solution across all workload configurations.

        Args:
            solution: Candidate kernel solution
            config: Optional configuration (unused)
            dump_traces: Whether to dump trace logs (unused)
            round_num: Current optimization round (unused)

        Returns:
            EvalResult with aggregated metrics
        """
        entry_source = solution.get_entry_source()
        if entry_source is None:
            return EvalResult(
                status="failed",
                log_excerpt="No entry source found in solution",
            )

        try:
            self._harness.compile_kernel(entry_source.content)
        except Exception as e:
            return EvalResult(
                status="failed",
                log_excerpt=f"Kernel compilation failed: {e}",
            )

        def candidate_fn(**inputs):
            """Wrapper for candidate kernel execution."""
            return self._harness.execute_once(
                x=inputs["x"],
                weight=inputs["weight"],
                bias=inputs.get("bias"),
                residual=inputs.get("residual"),
                activation=inputs.get("activation", "silu"),
            )

        result = self._evaluator.evaluate(
            solution=solution,
            reference_fn=fla_causal_conv1d_reference,
            workloads=self.WORKLOADS,
            input_generator=self._input_generator,
        )

        return result

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        """
        Format code for world model context.

        Args:
            raw: Raw code (string or Solution)
            language: Programming language

        Returns:
            Formatted code string
        """
        if isinstance(raw, str):
            return raw

        if isinstance(raw, Solution):
            entry_source = raw.get_entry_source()
            if entry_source:
                return entry_source.content

        return str(raw)

    def seed_eval_for_base_solution(
        self,
        *,
        base_solution: Solution,
        config: Any = None,
    ) -> EvalResult:
        """
        Generate seed evaluation for baseline solution.

        Args:
            base_solution: Baseline solution to evaluate
            config: Optional configuration

        Returns:
            EvalResult with status='seeded'
        """
        result = self.run_benchmark(solution=base_solution, config=config)

        return EvalResult(
            status="seeded",
            latency_ms=result.latency_ms,
            reference_latency_ms=result.latency_ms,
            speedup_factor=1.0,
            mean_vs_baseline_factor=1.0,
            log_excerpt=result.log_excerpt,
            metrics=result.metrics,
        )

    def get_config_for_logging(self) -> dict[str, Any]:
        """
        Get task configuration for logging.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "task": self.name,
            "dtype": str(self._dtype),
            "device": self._device,
            "num_workloads": len(self.WORKLOADS),
            "workloads": self.WORKLOADS,
        }
