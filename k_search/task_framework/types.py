"""Core types for task framework."""

from dataclasses import dataclass, field
from typing import Any

from k_search.task_framework.protocols.results import EvaluationResult, SolutionArtifact


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
    """Complete result of evaluating a solution."""

    solution: SolutionArtifact
    result: EvaluationResult
    analysis: AnalysisResult | None = None
