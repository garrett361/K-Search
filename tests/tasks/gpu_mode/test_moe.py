"""Tests for gpu_mode MoE task."""

import sys
from pathlib import Path

import pytest
import torch

_TASK_DIR = Path(__file__).parent.parent.parent.parent / "k_search" / "tasks" / "gpu_mode" / "moe"
sys.path.insert(0, str(_TASK_DIR))

from reference import check_implementation, generate_input, ref_kernel  # noqa: E402  # ty: ignore[unresolved-import]
from spec import MOE_SPEC_TEXT_TRITON  # noqa: E402  # ty: ignore[unresolved-import]
from submission import custom_kernel  # noqa: E402  # ty: ignore[unresolved-import]


class TestSpec:
    def test_spec_text_loads(self):
        """Verify spec text loads and is non-empty."""
        assert isinstance(MOE_SPEC_TEXT_TRITON, str)
        assert len(MOE_SPEC_TEXT_TRITON) > 500

    def test_spec_contains_interface(self):
        """Verify spec documents the 6-tuple interface."""
        assert "custom_kernel(data)" in MOE_SPEC_TEXT_TRITON
        assert "(x, w1, w2, w3, num_tokens_per_expert, config)" in MOE_SPEC_TEXT_TRITON

    def test_spec_does_not_mention_grouped_mm(self):
        """Verify spec does not leak torch._grouped_mm hint to LLM."""
        assert "grouped_mm" not in MOE_SPEC_TEXT_TRITON.lower()

    def test_spec_has_test_case(self):
        """Verify spec contains test case constraints."""
        assert "seq_tokens=4096" in MOE_SPEC_TEXT_TRITON
        assert "num_experts=256" in MOE_SPEC_TEXT_TRITON
        assert "top_k=4" in MOE_SPEC_TEXT_TRITON

    def test_spec_contains_baseline_submission(self):
        """Verify spec includes baseline submission.py code for LLM reference."""
        assert "Reference code (baseline `submission.py`)" in MOE_SPEC_TEXT_TRITON
        assert "def custom_kernel(data: input_t) -> output_t:" in MOE_SPEC_TEXT_TRITON
        assert "F.silu" in MOE_SPEC_TEXT_TRITON


@pytest.mark.cuda
class TestGenerateInput:
    def test_generate_input_returns_six_tuple(self):
        """Verify generate_input returns (x, w1, w2, w3, num_tokens_per_expert, config) tuple."""
        data = generate_input(
            seq_tokens=64, top_k=2, dim=32, hidden_dim=64, num_experts=4, seed=42
        )

        assert isinstance(data, tuple)
        assert len(data) == 6
        x, w1, w2, w3, num_tokens_per_expert, config = data
        assert isinstance(x, torch.Tensor)
        assert isinstance(w1, torch.Tensor)
        assert isinstance(w2, torch.Tensor)
        assert isinstance(w3, torch.Tensor)
        assert isinstance(num_tokens_per_expert, torch.Tensor)
        assert isinstance(config, dict)

    def test_generate_input_shapes_and_dtype(self):
        """Verify generate_input produces correct shapes and dtype."""
        seq_tokens, top_k, dim, hidden_dim, num_experts = 128, 2, 64, 128, 8
        x, w1, w2, w3, num_tokens_per_expert, config = generate_input(
            seq_tokens=seq_tokens,
            top_k=top_k,
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            seed=42,
        )

        total_tokens = seq_tokens * top_k
        assert x.shape == (total_tokens, dim)
        assert w1.shape == (num_experts, hidden_dim, dim)
        assert w2.shape == (num_experts, dim, hidden_dim)
        assert w3.shape == (num_experts, hidden_dim, dim)
        assert num_tokens_per_expert.shape == (num_experts,)

        assert x.dtype == torch.bfloat16
        assert w1.dtype == torch.bfloat16
        assert w2.dtype == torch.bfloat16
        assert w3.dtype == torch.bfloat16
        assert num_tokens_per_expert.dtype == torch.int32

    def test_generate_input_seeded(self):
        """Verify same seed produces same outputs."""
        data1 = generate_input(
            seq_tokens=64, top_k=2, dim=32, hidden_dim=64, num_experts=4, seed=123
        )
        data2 = generate_input(
            seq_tokens=64, top_k=2, dim=32, hidden_dim=64, num_experts=4, seed=123
        )

        assert torch.allclose(data1[0], data2[0])
        assert torch.allclose(data1[1], data2[1])
        assert torch.allclose(data1[2], data2[2])
        assert torch.allclose(data1[3], data2[3])
        assert torch.equal(data1[4], data2[4])

    def test_generate_input_balanced_distribution(self):
        """Verify tokens are balanced across experts."""
        seq_tokens, top_k, num_experts = 100, 4, 8
        total_tokens = seq_tokens * top_k  # 400
        data = generate_input(
            seq_tokens=seq_tokens,
            top_k=top_k,
            dim=32,
            hidden_dim=64,
            num_experts=num_experts,
            seed=42,
        )
        num_tokens_per_expert = data[4]

        assert num_tokens_per_expert.sum().item() == total_tokens
        assert num_tokens_per_expert.min().item() == total_tokens // num_experts
        assert num_tokens_per_expert.max().item() <= total_tokens // num_experts + 1

    def test_generate_input_all_tensors_on_cuda(self):
        """Verify all input tensors are on CUDA."""
        x, w1, w2, w3, num_tokens_per_expert, config = generate_input(
            seq_tokens=64, top_k=2, dim=32, hidden_dim=64, num_experts=4, seed=42
        )

        assert x.is_cuda, f"x should be on CUDA, got {x.device}"
        assert w1.is_cuda, f"w1 should be on CUDA, got {w1.device}"
        assert w2.is_cuda, f"w2 should be on CUDA, got {w2.device}"
        assert w3.is_cuda, f"w3 should be on CUDA, got {w3.device}"
        assert num_tokens_per_expert.is_cuda, f"num_tokens_per_expert should be on CUDA, got {num_tokens_per_expert.device}"


@pytest.mark.cuda
class TestReference:
    def test_ref_kernel_output_shape(self):
        """Verify ref_kernel produces correct output shape."""
        seq_tokens, top_k, dim = 128, 2, 64
        data = generate_input(
            seq_tokens=seq_tokens,
            top_k=top_k,
            dim=dim,
            hidden_dim=128,
            num_experts=8,
            seed=42,
        )
        output = ref_kernel(data)

        total_tokens = seq_tokens * top_k
        assert output.shape == (total_tokens, dim)
        assert output.dtype == torch.bfloat16


@pytest.mark.cuda
class TestGroupedMmEquivalence:
    def test_forloop_matches_grouped_mm(self):
        """Verify for-loop reference matches torch._grouped_mm (if available)."""
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip("torch._grouped_mm not available")

        seq_tokens, top_k, dim, hidden_dim, num_experts = 128, 2, 64, 128, 8
        data = generate_input(
            seq_tokens=seq_tokens,
            top_k=top_k,
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            seed=42,
        )
        x, w1, w2, w3, num_tokens_per_expert, _ = data

        forloop_output = ref_kernel(data)

        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate_out = torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs)
        up_out = torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs)
        h = torch.nn.functional.silu(gate_out) * up_out
        grouped_mm_output = torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)

        assert torch.allclose(forloop_output, grouped_mm_output, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
class TestIntegration:
    def test_baseline_matches_reference(self):
        """Verify custom_kernel matches ref_kernel exactly."""
        data = generate_input(
            seq_tokens=128, top_k=2, dim=64, hidden_dim=128, num_experts=8, seed=42
        )

        expected = ref_kernel(data)
        actual = custom_kernel(data)

        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)

    def test_baseline_output_shape_and_dtype(self):
        """Verify custom_kernel produces correct output."""
        seq_tokens, top_k, dim = 128, 2, 64
        data = generate_input(
            seq_tokens=seq_tokens,
            top_k=top_k,
            dim=dim,
            hidden_dim=128,
            num_experts=8,
            seed=42,
        )
        output = custom_kernel(data)

        total_tokens = seq_tokens * top_k
        assert output.shape == (total_tokens, dim)
        assert output.dtype == torch.bfloat16

    def test_check_implementation_passes_baseline(self):
        """Verify baseline submission passes check_implementation."""
        data = generate_input(
            seq_tokens=128, top_k=2, dim=64, hidden_dim=128, num_experts=8, seed=42
        )
        output = custom_kernel(data)

        passed, message = check_implementation(data, output)
        assert passed, f"Baseline should pass: {message}"

    def test_check_implementation_fails_wrong_output(self):
        """Verify check_implementation catches incorrect outputs."""
        data = generate_input(
            seq_tokens=128, top_k=2, dim=64, hidden_dim=128, num_experts=8, seed=42
        )
        wrong_output = torch.ones_like(data[0]) * 100  # Large values clearly wrong

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong output"

    def test_check_implementation_fails_wrong_shape(self):
        """Verify check_implementation rejects wrong shape."""
        data = generate_input(
            seq_tokens=128, top_k=2, dim=64, hidden_dim=128, num_experts=8, seed=42
        )
        wrong_output = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)

        passed, message = check_implementation(data, wrong_output)
        assert not passed, "Should fail for wrong shape"


@pytest.mark.cuda
@pytest.mark.slow
class TestPerformance:
    def test_mixtral_scale_test_case(self):
        """Verify reference and baseline work on Mixtral-like config."""
        data = generate_input(
            seq_tokens=4096, top_k=2, dim=4096, hidden_dim=14336, num_experts=8, seed=42
        )

        ref_output = ref_kernel(data)
        baseline_output = custom_kernel(data)

        total_tokens = 4096 * 2
        assert ref_output.shape == (total_tokens, 4096)
        assert baseline_output.shape == (total_tokens, 4096)
        assert torch.allclose(ref_output, baseline_output, rtol=2e-2, atol=2e-2)
