"""Cycle dataclass for tracking attempt rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.round import Round


@dataclass
class Cycle:
    """Result of attempting an Action - all rounds."""

    rounds: list[Round] = field(default_factory=list)

    @property
    def best_round(self) -> Round | None:
        """Return highest-scoring successful round, or None."""
        successful = [r for r in self.rounds if r.result.is_success()]
        return max(successful, key=lambda r: r.score) if successful else None

    @property
    def succeeded(self) -> bool:
        """Return True if any round succeeded."""
        return any(r.result.is_success() for r in self.rounds)
