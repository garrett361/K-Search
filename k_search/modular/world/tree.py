"""Tree dataclass for search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.node import Node


@dataclass
class Tree:
    """Search tree container."""

    root: Node
    annotations: dict[str, Any] | None = None
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._assign_id(self.root)

    def _assign_id(self, node: Node) -> None:
        node._id = str(self._next_id)
        self._next_id += 1

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to its parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent (use root for root node)")
        self._assign_id(node)
        node.parent.children.append(node)

    def _get_node_by_id(self, id: str) -> Node | None:
        """Look up node by ID. For tools.py only."""
        for node in self._all_nodes():
            if node._id == id:
                return node
        return None

    def get_frontier(self) -> list[Node]:
        """Return all nodes with status 'open'."""
        return [n for n in self._all_nodes() if n.status == "open"]

    def get_best_node(self) -> Node | None:
        """Return best completed node by score, or None."""
        completed = [
            n
            for n in self._all_nodes()
            if n.status == "closed" and n.cycle and n.cycle.succeeded
        ]
        if not completed:
            return None
        return max(completed, key=lambda n: n.cycle.best_round.score)

    def get_path_to_root(self, node: Node) -> list[Node]:
        """Return path from node to root (inclusive)."""
        path = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def _all_nodes(self) -> list[Node]:
        """Return all nodes in tree via BFS."""
        result = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result
