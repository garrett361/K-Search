"""WorldModel protocol for search tree operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from k_search.modular.world.node import Node


class WorldModel(Protocol):
    """World model interface (P_world from the paper).

    Context is implementation-defined. See specific implementations
    for their context requirements.
    """

    def propose(self, context: Any) -> list[Node]:
        """Generate new frontier nodes with actions.

        Returns empty list if no proposals available.
        """
        ...

    def select(self, context: Any) -> list[Node]:
        """Select frontier nodes to pursue.

        Returns empty list if nothing to select.
        """
        ...

    def update(self, context: Any) -> None:
        """Update tree after cycle completes."""
        ...
