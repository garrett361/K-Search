"""Evaluator protocol."""

from typing import Any, Protocol

from k_search.modular.protocols.eval_result import EvaluationResult
from k_search.modular.protocols.impl import Implementation


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
