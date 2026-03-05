"""ArtifactStore protocol definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from k_search.modular.round import Round


class ArtifactStore(Protocol):
    """Protocol for storing artifacts during search."""

    def store(self, round_: Round, round_idx: int) -> None:
        """Store artifacts from a search round."""
        ...
