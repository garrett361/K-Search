"""Evaluator protocol."""

from typing import Any, Protocol

from k_search.task_framework.protocols.results import EvaluationResult, Implementation


class Evaluator(Protocol):
    """Executes an implementation and produces evaluation results."""

    def evaluate(
        self,
        impl: Implementation,
        *,
        timeout_secs: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate implementation and return result."""
        ...
