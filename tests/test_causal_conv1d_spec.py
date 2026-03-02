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
        config = task._evaluator._correctness_config
        assert config.rtol == 2e-2
        assert config.atol == 2e-2
