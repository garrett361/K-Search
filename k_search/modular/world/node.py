"""Node dataclass for search tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.action import Action
    from k_search.modular.world.cycle import Cycle


@dataclass
class Node:
    """Search tree node."""

    _id: str = ""
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    status: str = "open"  # "open" | "in_progress" | "closed"

    action: Action | None = None
    cycle: Cycle | None = None
    annotations: dict[str, Any] | None = None
