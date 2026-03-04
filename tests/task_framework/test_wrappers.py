"""Tests for GpuMode wrappers."""

from k_search.tasks.task_base import (
    EvalResult,
    Solution,
    BuildSpec,
    SourceFile,
    SupportedLanguages,
)


class TestGpuModeEvaluationResult:
    def test_wraps_eval_result(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.5, log_excerpt="test log")
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.is_success() is True
        assert wrapper.get_log() == "test log"

    def test_is_success_false_for_failed(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(status="failed", log_excerpt="error")
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.is_success() is False

    def test_get_metrics_excludes_log(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(
            status="passed",
            latency_ms=1.5,
            speedup_factor=2.0,
            log_excerpt="long log",
        )
        wrapper = GpuModeEvaluationResult(inner)
        metrics = wrapper.get_metrics()

        assert "latency_ms" in metrics
        assert "log_excerpt" not in metrics

    def test_backwards_compat_is_passed(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.0)
        wrapper = GpuModeEvaluationResult(inner)

        # V1 interface
        assert wrapper.is_passed() is True
        assert wrapper.latency_ms == 1.0
        assert wrapper.status == "passed"

    def test_backwards_compat_to_dict(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.0, log_excerpt="log")
        wrapper = GpuModeEvaluationResult(inner)

        d = wrapper.to_dict(include_log_excerpt=True)
        assert d["status"] == "passed"
        assert d["latency_ms"] == 1.0

    def test_backwards_compat_score(self):
        from k_search.task_framework.adapters.wrappers import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=2.0)
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.score() == 0.5  # 1/latency


class TestGpuModeSolutionArtifact:
    def test_wraps_solution(self):
        from k_search.task_framework.adapters.wrappers import GpuModeSolutionArtifact

        inner = Solution(
            name="test_sol",
            definition="test_def",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[
                SourceFile(path="submission.py", content="def custom_kernel(): pass")
            ],
        )
        wrapper = GpuModeSolutionArtifact(inner)

        assert wrapper.name == "test_sol"
        assert "custom_kernel" in wrapper.content
