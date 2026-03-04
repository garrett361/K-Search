"""Core types for task framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.task_framework.protocols.results import (
        EvaluationResult,
        Implementation,
    )


@dataclass
class CheckResult:
    """Result of correctness check."""

    passed: bool
    message: str = ""
    criteria: dict[str, Any] | None = None


@dataclass
class AnalysisResult:
    """Result of post-evaluation analysis."""

    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_artifact: bytes | None = None
    strategic_guidance: str | None = None


@dataclass
class EvalOutcome:
    """Complete result of evaluating an implementation."""

    impl: Implementation
    result: EvaluationResult
    analysis: AnalysisResult | None = None
