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
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.5, log_excerpt="test log")
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.succeeded() is True
        assert wrapper.get_log() == "test log"

    def test_is_success_false_for_failed(self):
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

        inner = EvalResult(status="failed", log_excerpt="error")
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.succeeded() is False

    def test_get_metrics_excludes_log(self):
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

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
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.0)
        wrapper = GpuModeEvaluationResult(inner)

        # V1 interface
        assert wrapper.is_passed() is True
        assert wrapper.latency_ms == 1.0
        assert wrapper.status == "passed"

    def test_backwards_compat_to_dict(self):
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=1.0, log_excerpt="log")
        wrapper = GpuModeEvaluationResult(inner)

        d = wrapper.to_dict(include_log_excerpt=True)
        assert d["status"] == "passed"
        assert d["latency_ms"] == 1.0

    def test_backwards_compat_score(self):
        from k_search.modular.adapters.gpu_mode import GpuModeEvaluationResult

        inner = EvalResult(status="passed", latency_ms=2.0)
        wrapper = GpuModeEvaluationResult(inner)

        assert wrapper.score() == 0.5  # 1/latency


class TestGpuModeImplementation:
    def test_wraps_solution(self):
        from k_search.modular.adapters.gpu_mode import GpuModeImplementation

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
        wrapper = GpuModeImplementation(inner)

        assert wrapper.name == "test_sol"
        assert "custom_kernel" in wrapper.content.sources[0].content


class TestGpuModeImplementationArtifactDir:
    def test_yields_directory_with_source_files(self):
        from k_search.modular.adapters.gpu_mode import GpuModeImplementation

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[
                SourceFile(path="kernel.py", content="def custom_kernel(): pass"),
                SourceFile(path="utils.py", content="# utils"),
            ],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            assert src_dir is not None
            assert (src_dir / "kernel.py").read_text() == "def custom_kernel(): pass"
            assert (src_dir / "utils.py").read_text() == "# utils"

    def test_yields_none_when_no_sources(self):
        from k_search.modular.adapters.gpu_mode import GpuModeImplementation

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            assert src_dir is None

    def test_cleans_up_temp_dir_after_context(self):
        from k_search.modular.adapters.gpu_mode import GpuModeImplementation

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[SourceFile(path="kernel.py", content="code")],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            assert src_dir is not None
            dir_path = src_dir

        assert not dir_path.exists()


class TestRound:
    def test_round_holds_impl_and_result(self):
        from k_search.modular.adapters.gpu_mode import (
            GpuModeEvaluationResult,
            GpuModeImplementation,
        )
        from k_search.modular import Round
        from k_search.tasks.task_base import (
            EvalResult,
            Solution,
            BuildSpec,
            SourceFile,
            SupportedLanguages,
        )

        sol = Solution(
            name="test",
            definition="def",
            author="author",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(path="submission.py", content="code")],
        )
        result = EvalResult(status="passed", latency_ms=1.0)

        round_ = Round(
            impl=GpuModeImplementation(sol),
            result=GpuModeEvaluationResult(result),
            prompt="test prompt",
            llm_response="test response",
            prompt_tokens=100,
            completion_tokens=50,
            duration_secs=1.0,
            score=1.0,
        )

        assert round_.impl.name == "test"
        assert round_.result.succeeded()
