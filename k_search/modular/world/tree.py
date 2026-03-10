"""Tree dataclass for search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node


@dataclass
class Tree:
    """Search tree container."""

    root: Node
    _next_id: int = field(default=0, init=False)
    _nodes_by_id: dict[str, Node] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[self.root.id] = self.root

    def add_node(self, node: Node) -> None:
        """Add node to tree, attaching to its parent's children list."""
        if node.parent is None:
            raise ValueError("Cannot add node without parent (use root for root node)")
        node.id = str(self._next_id)
        self._next_id += 1
        self._nodes_by_id[node.id] = node
        node.parent.children.append(node)

    def get_node_by_id(self, id: str) -> Node | None:
        """Look up node by ID. O(1) via dict."""
        return self._nodes_by_id.get(id)

    def get_frontier(self) -> list[Node]:
        """Return all nodes with status 'open'."""
        return [n for n in self._nodes_by_id.values() if n.status == "open"]

    def get_best_node(self) -> Node | None:
        """Return best completed node by score, or None."""
        completed = [
            n
            for n in self._nodes_by_id.values()
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

    def split_node(self, node: Node, children: list[Node]) -> None:
        """Mark node closed and add pre-constructed children to tree."""
        node.status = "closed"
        for child in children:
            child.parent = node
            self.add_node(child)

    def delete_node(self, node: Node) -> None:
        """Soft delete - mark node deleted, preserves tree structure."""
        node.status = "deleted"
