"""Tests for gpu_mode causal_conv1d task."""

import sys
from pathlib import Path

import pytest
import torch

_TASK_DIR = Path(__file__).parent.parent / "k_search" / "tasks" / "gpu_mode" / "causal_conv1d"
sys.path.insert(0, str(_TASK_DIR))

from reference import check_implementation, generate_input, ref_kernel  # noqa: E402  # ty: ignore[unresolved-import]
from spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON  # noqa: E402  # ty: ignore[unresolved-import]
from submission import custom_kernel  # noqa: E402  # ty: ignore[unresolved-import]


class TestSpec:
    def test_spec_text_loads(self):
        assert isinstance(CAUSAL_CONV1D_SPEC_TEXT_TRITON, str)
        assert len(CAUSAL_CONV1D_SPEC_TEXT_TRITON) > 500

    def test_spec_contains_interface(self):
        assert "custom_kernel(data)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "(x, weight, bias, residual, config)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_contains_reference(self):
        try:
            import fla.modules.convolution  # noqa: F401

            assert "causal_conv1d_fwd_kernel" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
            assert "@triton.jit" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        except ImportError:
            pytest.skip("FLA not installed")


@pytest.mark.cuda
class TestGenerateInput:
    def test_generate_input_shapes(self):
        B, T, D, W = 2, 128, 64, 4
        x, weight, bias, residual, config = generate_input(
            B=B, T=T, D=D, W=W, seed=42, activation="silu", withbias=True, withresidual=True
        )

        assert x.shape == (B, T, D)
        assert weight.shape == (D, W)
        assert bias.shape == (D,)
        assert residual.shape == (B, T, D)
        assert config == {"activation": "silu"}

    def test_generate_input_seeded(self):
        data1 = generate_input(B=2, T=64, D=32, W=4, seed=123, activation="silu", withbias=True, withresidual=False)
        data2 = generate_input(B=2, T=64, D=32, W=4, seed=123, activation="silu", withbias=True, withresidual=False)

        assert torch.allclose(data1[0], data2[0])
        assert torch.allclose(data1[1], data2[1])
        assert torch.allclose(data1[2], data2[2])

    def test_generate_input_bias_none(self):
        x, weight, bias, residual, config = generate_input(
            B=2, T=64, D=32, W=4, seed=42, activation="silu", withbias=False, withresidual=True
        )

        assert bias is None

    def test_generate_input_residual_none(self):
        x, weight, bias, residual, config = generate_input(
            B=2, T=64, D=32, W=4, seed=42, activation="silu", withbias=True, withresidual=False
        )

        assert residual is None


@pytest.mark.cuda
class TestIntegration:
    def test_baseline_correctness(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42, activation="silu", withbias=True, withresidual=False)

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    def test_check_implementation_passes_baseline(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42, activation="silu", withbias=True, withresidual=True)
        output = custom_kernel(data)

        passed, message = check_implementation(data, output)
        assert passed, f"Baseline should pass: {message}"

    def test_check_implementation_fails_wrong_output(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42, activation="silu", withbias=True, withresidual=False)
        wrong_output = torch.zeros_like(data[0])

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong output"
