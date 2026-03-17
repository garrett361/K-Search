"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """Base tree node with parent/children structure only."""

    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
