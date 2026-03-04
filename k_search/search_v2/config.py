"""Search configuration and result types."""

from dataclasses import dataclass

from k_search.task_framework.protocols.results import EvaluationResult, Implementation


@dataclass
class SearchConfig:
    """Configuration for search loop."""

    max_rounds: int = 10
    timeout_secs: int | None = None


@dataclass
class SearchResult:
    """Result from a search run."""

    impl: Implementation | None
    score: float
    result: EvaluationResult | None
    rounds_completed: int = 0
