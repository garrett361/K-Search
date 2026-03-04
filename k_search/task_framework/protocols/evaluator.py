"""Evaluator protocol."""

from typing import Any, Protocol

from k_search.task_framework.protocols.results import EvaluationResult, SolutionArtifact


class Evaluator(Protocol):
    """Executes a solution and produces evaluation results."""

    def evaluate(
        self,
        solution: SolutionArtifact,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate solution and return result."""
        ...
