"""No-op artifact store implementation."""

from k_search.task_framework.types import EvalOutcome


class NoOpArtifactStore:
    """Artifact store that does nothing."""

    def store(self, outcome: EvalOutcome, round_idx: int) -> None:
        pass
