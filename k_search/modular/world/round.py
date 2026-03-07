"""Round container for complete iteration context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.protocols.eval_result import EvaluationResult
    from k_search.modular.protocols.impl import Implementation
    from k_search.modular.results import AnalysisResult


@dataclass
class Round:
    """Complete result of a search iteration.

    Contains all context needed by downstream consumers (FeedbackProvider,
    ArtifactStore, metrics) to understand and report on the round.
    """

    impl: Implementation
    result: EvaluationResult
    prompt: str
    llm_response: str
    prompt_tokens: int
    completion_tokens: int
    duration_secs: float
    score: float
    analysis: AnalysisResult | None = None
