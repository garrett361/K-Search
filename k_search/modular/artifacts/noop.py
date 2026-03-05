"""No-op artifact store implementation."""

from k_search.modular.round import Round


class NoOpArtifactStore:
    """Artifact store that does nothing."""

    def store(self, round_: Round, round_idx: int) -> None:
        pass
