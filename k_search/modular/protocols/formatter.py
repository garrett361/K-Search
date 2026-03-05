"""StateFormatter protocol for tree serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class StateFormatter(Protocol):
    """Tree serialization for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        """Format tree for P_world prompt."""
        ...

    def format_node(self, node: Node) -> str:
        """Format single node for display."""
        ...
