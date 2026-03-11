"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.action import Action
    from k_search.modular.world.cycle import Cycle


@dataclass
class Node:
    """Search tree node.

    Future: Consider splitting into base Node (id, parent, children, status)
    and subclasses that add action/cycle. This would enable multiple cycles
    per node and cleaner separation of tree structure from domain data.
    """

    id: str = ""
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = ""
    action: Action | None = None
    cycle: Cycle | None = None
