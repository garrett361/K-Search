"""Generic correctness checking for custom Triton kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CorrectnessConfig:
    """Configuration for correctness checking tolerances."""

    rtol: float = 1e-2
    atol: float = 1e-2


def check_correctness(
    received: torch.Tensor,
    expected: torch.Tensor,
    config: CorrectnessConfig,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness using torch.testing.assert_close.

    Args:
        received: Output from candidate implementation
        expected: Output from reference implementation
        config: Correctness checking configuration

    Returns:
        Tuple of (passed, details_dict) where details_dict contains:
        - max_abs_error: Maximum absolute error
        - max_rel_error: Maximum relative error
        - error_message: Description of any mismatch
    """
    if received.shape != expected.shape:
        return False, {
            "max_abs_error": float("inf"),
            "max_rel_error": float("inf"),
            "error_message": f"Shape mismatch: received {received.shape}, expected {expected.shape}",
        }

    rec_float = received.detach().to(torch.float32)
    exp_float = expected.detach().to(torch.float32)

    diff = torch.abs(rec_float - exp_float)
    max_abs_error = float(diff.max().item())

    with torch.no_grad():
        safe_exp = torch.where(exp_float != 0, exp_float, torch.ones_like(exp_float))
        rel_errors = diff / torch.abs(safe_exp)
        max_rel_error = float(rel_errors.max().item())

    try:
        torch.testing.assert_close(
            received,
            expected,
            rtol=config.rtol,
            atol=config.atol,
            equal_nan=True,
        )
        return True, {
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "error_message": f"All elements match within tolerance (max abs error: {max_abs_error:.6g})",
        }
    except AssertionError as e:
        return False, {
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "error_message": str(e),
        }
