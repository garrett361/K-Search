"""ArtifactStore protocol definition."""

from typing import Protocol

from k_search.task_framework.types import EvalOutcome


class ArtifactStore(Protocol):
    """Protocol for storing artifacts during search."""

    def store(self, outcome: EvalOutcome, round_idx: int) -> None: ...
