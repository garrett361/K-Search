"""WorldModel protocol for search tree operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class WorldModel(Protocol):
    """World model interface (P_world from the paper)."""

    def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Generate new frontier nodes with actions.

        Returns empty list if no proposals available.
        """
        ...

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Select frontier nodes to pursue.

        Returns empty list if nothing to select.
        """
        ...

    def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """Update tree after cycle completes."""
        ...
