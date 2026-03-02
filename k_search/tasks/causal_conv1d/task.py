"""Causal Conv1d task implementation for K-Search."""

from __future__ import annotations

import logging
from typing import Any

import torch

from k_search.tasks.custom_triton import (
    BenchmarkConfig,
    CorrectnessConfig,
    MeanAggregation,
    benchmark_triton_kernel,
    check_correctness,
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
    """K-Search task for optimizing causal_conv1d_fwd kernels."""

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
        self._correctness_config = CorrectnessConfig(rtol=2e-2, atol=2e-2)

    @property
    def name(self) -> str:
        return "causal_conv1d"

    def get_definition_text(self, language: str | None = None) -> str:
        return CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def get_solution(self, solution_name: str) -> Solution | None:
        if solution_name.lower() in ("baseline", "fla", "reference"):
            return None  # No extractable baseline; use reference fn directly
        return None

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        code_text = str(cleaned_code or raw_code or "")
        return Solution(
            name=f"{model_name}_{self.name}_{language}_r{round_num}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[str(target_gpu)],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(path="submission.py", content=code_text)],
            description=f"Causal conv1d optimized kernel (round {round_num})",
        )

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        entry_source = solution.get_entry_source()
        if entry_source is None:
            return EvalResult(status="failed", log_excerpt="No entry source found")

        try:
            self._harness.compile_kernel(entry_source.content)
        except Exception as e:
            return EvalResult(status="failed", log_excerpt=f"Compilation failed: {e}")

        passed_latencies: list[float] = []
        passed_speedups: list[float] = []
        errors: list[str] = []

        for workload in self.WORKLOADS:
            try:
                inputs = self._make_inputs(workload)

                candidate_out = self._harness.execute_once(
                    x=inputs["x"],
                    weight=inputs["weight"],
                    bias=inputs.get("bias"),
                    residual=inputs.get("residual"),
                    activation=inputs.get("activation", "silu"),
                )
                reference_out = fla_causal_conv1d_reference(**inputs)

                ok, details = check_correctness(candidate_out, reference_out, self._correctness_config)
                if not ok:
                    errors.append(f"Workload {workload}: {details.get('error_message', 'mismatch')}")
                    continue

                def run_candidate(**kw: Any) -> torch.Tensor:
                    return self._harness.execute_once(
                        x=kw["x"], weight=kw["weight"], bias=kw.get("bias"),
                        residual=kw.get("residual"), activation=kw.get("activation", "silu"),
                    )

                cand_bench = benchmark_triton_kernel(run_candidate, inputs, BenchmarkConfig())
                ref_bench = benchmark_triton_kernel(fla_causal_conv1d_reference, inputs, BenchmarkConfig())

                lat = cand_bench["latency_ms"]
                ref_lat = ref_bench["latency_ms"]
                passed_latencies.append(lat)
                passed_speedups.append(ref_lat / lat if lat > 0 else 0.0)

            except Exception as e:
                errors.append(f"Workload {workload}: {e}")

        if not passed_latencies:
            return EvalResult(
                status="failed",
                log_excerpt=f"All workloads failed: {errors[:3]}",
                metrics={"num_passed": 0, "num_total": len(self.WORKLOADS)},
            )

        mean_lat = sum(passed_latencies) / len(passed_latencies)
        mean_speedup = sum(passed_speedups) / len(passed_speedups)
        num_passed = len(passed_latencies)

        return EvalResult(
            status="passed" if num_passed == len(self.WORKLOADS) else "failed",
            latency_ms=mean_lat,
            speedup_factor=mean_speedup,
            mean_vs_baseline_factor=mean_speedup,
            log_excerpt=f"{num_passed}/{len(self.WORKLOADS)} passed, {mean_lat:.3f}ms, {mean_speedup:.2f}x",
            metrics={
                "num_passed": num_passed,
                "num_total": len(self.WORKLOADS),
                "score_name": "speedup",
                "score": mean_speedup,
            },
        )

    def _make_inputs(self, workload: dict[str, Any]) -> dict[str, Any]:
        inputs = create_reference_inputs(
            B=workload["B"], T=workload["T"], D=workload["D"], W=workload["W"],
            dtype=self._dtype, device=self._device, with_bias=True, with_residual=False,
        )
        inputs["activation"] = workload.get("activation", "silu")
        return inputs

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        # No compilable baseline; return a seeded result using reference timing
        inputs = self._make_inputs(self.WORKLOADS[0])
        ref_bench = benchmark_triton_kernel(fla_causal_conv1d_reference, inputs, BenchmarkConfig())
        return EvalResult(
            status="seeded",
            latency_ms=ref_bench["latency_ms"],
            reference_latency_ms=ref_bench["latency_ms"],
            speedup_factor=1.0,
            mean_vs_baseline_factor=1.0,
            log_excerpt="Seeded with FLA reference timing",
            metrics={"score_name": "speedup", "score": 1.0},
        )

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = []
        for sol in solutions or []:
            if sol is None:
                continue
            er = self.run_benchmark(solution=sol, round_num=None)
            results.append({
                "solution": sol.name,
                "status": er.status,
                "latency_ms": er.latency_ms,
                "speedup": er.speedup_factor,
            })
        return {"task": self.name, "solutions": results}

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, Solution):
            src = raw.get_entry_source()
            return src.content if src else ""
        return str(raw)

    def get_config_for_logging(self) -> dict[str, Any]:
        return {
            "task": self.name,
            "dtype": str(self._dtype),
            "device": self._device,
            "num_workloads": len(self.WORKLOADS),
        }
