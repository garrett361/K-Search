"""Tests for gpu_mode causal_conv1d task."""

import sys
from pathlib import Path

import pytest
import torch

_TASK_DIR = Path(__file__).parent.parent.parent.parent / "k_search" / "tasks" / "gpu_mode" / "causal_conv1d"
sys.path.insert(0, str(_TASK_DIR))

from reference import check_implementation, generate_input, ref_kernel  # noqa: E402  # ty: ignore[unresolved-import]
from spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON  # noqa: E402  # ty: ignore[unresolved-import]
from submission import custom_kernel  # noqa: E402  # ty: ignore[unresolved-import]


class TestSpec:
    def test_spec_text_loads(self):
        """Verify spec text loads and is non-empty."""
        assert isinstance(CAUSAL_CONV1D_SPEC_TEXT_TRITON, str)
        assert len(CAUSAL_CONV1D_SPEC_TEXT_TRITON) > 500

    def test_spec_contains_interface(self):
        """Verify spec documents the 3-tuple interface."""
        assert "custom_kernel(data)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "(x, weight, config)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_mentions_silu(self):
        """Verify spec documents SiLU activation."""
        assert "silu" in CAUSAL_CONV1D_SPEC_TEXT_TRITON.lower()

    def test_spec_has_test_case(self):
        """Verify spec contains test case constraints."""
        assert "B = 2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON or "B=2" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "4096" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "2048" in CAUSAL_CONV1D_SPEC_TEXT_TRITON

    def test_spec_contains_baseline_submission(self):
        """Verify spec includes baseline submission.py code for LLM reference."""
        assert "Reference code (baseline `submission.py`)" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "def custom_kernel(data: input_t) -> output_t:" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "F.conv1d" in CAUSAL_CONV1D_SPEC_TEXT_TRITON
        assert "F.silu" in CAUSAL_CONV1D_SPEC_TEXT_TRITON


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
    def test_baseline_matches_reference(self):
        """Verify custom_kernel matches ref_kernel exactly."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    def test_baseline_output_shape_and_dtype(self):
        """Verify custom_kernel produces correct output."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = custom_kernel(data)

        assert output.shape == (2, 128, 64)
        assert output.dtype == torch.bfloat16

    def test_check_implementation_passes_baseline(self):
        """Verify baseline submission passes check_implementation."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        output = custom_kernel(data)

        passed, message = check_implementation(data, output)
        assert passed, f"Baseline should pass: {message}"

    def test_check_implementation_fails_wrong_output(self):
        """Verify check_implementation catches incorrect outputs."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        wrong_output = torch.zeros_like(data[0])

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong output"

    def test_check_implementation_fails_wrong_shape(self):
        """Verify check_implementation rejects wrong shape."""
        data = generate_input(B=2, T=128, D=64, W=4, seed=42)
        wrong_output = torch.zeros(2, 64, 64, device="cuda", dtype=torch.bfloat16)

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong shape"


@pytest.mark.cuda
@pytest.mark.slow
class TestPerformance:
    def test_full_scale_test_case(self):
        """Verify reference and baseline work on full-scale test case."""
        # B=2, T=4096, D=2048, W=4 from spec
        data = generate_input(B=2, T=4096, D=2048, W=4, seed=42)

        ref_output = ref_kernel(data)
        baseline_output = custom_kernel(data)

        assert ref_output.shape == (2, 4096, 2048)
        assert baseline_output.shape == (2, 4096, 2048)
        assert torch.allclose(ref_output, baseline_output, rtol=2e-2, atol=2e-2)
