"""Analyzer protocol."""

from typing import Any, Protocol

from k_search.task_framework.protocols.results import EvaluationResult, SolutionArtifact
from k_search.task_framework.types import AnalysisResult


class Analyzer(Protocol):
    """Post-evaluation analysis (profiling, pattern detection, etc.)."""

    def analyze(
        self,
        solution: SolutionArtifact,
        result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        Analyze solution and result.

        Context may contain:
        - 'tree': SolutionTree for tree-aware analysis
        - 'recent_failures': list[EvalOutcome] for pattern detection
        """
        ...
