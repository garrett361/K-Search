"""Unit tests for causal_conv1d spec."""

import pytest


class TestCausalConv1dSpec:
    def test_spec_loads(self):
        from k_search.tasks.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

        assert CAUSAL_CONV1D_SPEC_TEXT_TRITON is not None
        assert len(CAUSAL_CONV1D_SPEC_TEXT_TRITON) > 1000

    def test_spec_contains_required_sections(self):
        from k_search.tasks.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

        required_sections = [
            "Task:",
            "Data interface:",
            "Inputs:",
            "Output:",
            "Correctness:",
            "Test Cases",
            "Reference code",
        ]
        for section in required_sections:
            assert section in CAUSAL_CONV1D_SPEC_TEXT_TRITON, f"Missing section: {section}"

    def test_spec_contains_tolerances(self):
        from k_search.tasks.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

        assert "rtol=2e-2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "atol=2e-2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_contains_test_cases(self):
        from k_search.tasks.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

        assert '"B": 2' in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert '"T": 4096' in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert '"D": 2048' in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_contains_reference_kernel(self):
        from k_search.tasks.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON

        assert "causal_conv1d_fwd_kernel" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "@triton.jit" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_task_tolerances_match_spec(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()
        assert task._correctness_config.rtol == 2e-2
        assert task._correctness_config.atol == 2e-2


class TestCausalConv1dTaskIntegration:
    def test_has_required_methods(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()

        assert hasattr(task, "name")
        assert hasattr(task, "get_definition_text")
        assert hasattr(task, "run_benchmark")
        assert hasattr(task, "run_final_evaluation")
        assert hasattr(task, "make_solution_from_generated_code")
        assert hasattr(task, "seed_eval_for_base_solution")
        assert hasattr(task, "code_for_world_model_from_raw")
        assert hasattr(task, "get_config_for_logging")

    def test_make_solution_from_generated_code(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()
        sol = task.make_solution_from_generated_code(
            cleaned_code="def custom_kernel(data): pass",
            raw_code="",
            round_num=1,
            model_name="test-model",
            target_gpu="H100",
            language="triton",
        )

        assert sol.name == "test-model_causal_conv1d_triton_r1"
        assert sol.definition == "causal_conv1d"
        assert sol.spec.entry_point == "submission.py::custom_kernel"
        assert len(sol.sources) == 1
        assert sol.sources[0].path == "submission.py"

    def test_run_final_evaluation_returns_dict(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()
        result = task.run_final_evaluation(solutions=[], config=None)

        assert isinstance(result, dict)
        assert "task" in result
        assert "solutions" in result
        assert result["task"] == "causal_conv1d"

    def test_code_for_world_model_from_raw_string(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()
        code = "def foo(): pass"
        result = task.code_for_world_model_from_raw(raw=code, language="triton")
        assert result == code

    def test_get_config_for_logging(self):
        from k_search.tasks.causal_conv1d.task import CausalConv1dTask

        task = CausalConv1dTask()
        config = task.get_config_for_logging()

        assert config["task"] == "causal_conv1d"
        assert "dtype" in config
        assert "device" in config
        assert config["num_workloads"] == 4
