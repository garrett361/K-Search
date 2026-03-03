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
    def test_generate_input_returns_three_tuple(self):
        """Verify generate_input returns (x, weight, config) tuple."""
        data = generate_input(B=2, T=64, D=32, W=4, seed=42)

        assert isinstance(data, tuple)
        assert len(data) == 3
        x, weight, config = data
        assert isinstance(x, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        assert isinstance(config, dict)

    def test_generate_input_shapes_and_dtype(self):
        """Verify generate_input produces correct shapes and dtype."""
        B, T, D, W = 2, 128, 64, 4
        x, weight, config = generate_input(B=B, T=T, D=D, W=W, seed=42)

        assert x.shape == (B, T, D)
        assert weight.shape == (D, W)
        assert x.dtype == torch.bfloat16
        assert weight.dtype == torch.bfloat16
        assert isinstance(config, dict)

    def test_generate_input_seeded(self):
        """Verify same seed produces same outputs."""
        data1 = generate_input(B=2, T=64, D=32, W=4, seed=123)
        data2 = generate_input(B=2, T=64, D=32, W=4, seed=123)

        assert torch.allclose(data1[0], data2[0])
        assert torch.allclose(data1[1], data2[1])


@pytest.mark.cuda
class TestReference:
    def test_ref_kernel_output_shape(self):
        """Verify ref_kernel produces correct output shape."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = ref_kernel(data)

        assert output.shape == (2, 128, 64)
        assert output.dtype == torch.bfloat16

    def test_ref_kernel_causal_property(self):
        """Verify output at time t only depends on inputs up to time t."""
        B, T, D, W = 1, 10, 4, 4
        x, weight, config = generate_input(B=B, T=T, D=D, W=W, seed=42)

        # Modify input after time t=5
        x_modified = x.clone()
        x_modified[:, 6:, :] = 0

        output_original = ref_kernel((x, weight, config))
        output_modified = ref_kernel((x_modified, weight, config))

        # Output at t=5 and earlier should be unchanged
        assert torch.allclose(output_original[:, :6, :], output_modified[:, :6, :])


@pytest.mark.cuda
class TestIntegration:
    @pytest.mark.skip(reason="Requires Task 3 (ref_kernel) and Task 4 (custom_kernel) to be updated")
    def test_baseline_correctness(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    @pytest.mark.skip(reason="Requires Task 3 (ref_kernel) and Task 4 (custom_kernel) to be updated")
    def test_check_implementation_passes_baseline(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = custom_kernel(data)

        passed, message = check_implementation(data, output)
        assert passed, f"Baseline should pass: {message}"

    @pytest.mark.skip(reason="Requires Task 3 (ref_kernel) and Task 4 (custom_kernel) to be updated")
    def test_check_implementation_fails_wrong_output(self):
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        wrong_output = torch.zeros_like(data[0])

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong output"
