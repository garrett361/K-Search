"""Scorer protocol."""

from typing import Protocol

from k_search.modular.protocols.eval_result import EvaluationResult


class Scorer(Protocol):
    """Converts evaluation results to comparable scalar."""

    def score(self, result: EvaluationResult) -> float:
        """Return score. Higher is better. Negative for failures."""
        ...
