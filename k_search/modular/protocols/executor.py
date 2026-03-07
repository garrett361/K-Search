"""Executor protocol for search orchestration."""

from typing import Protocol

from k_search.modular.world.node import Node


class Executor(Protocol):
    """Search executor interface.

    Minimal protocol - all configuration passed via __init__.
    """

    def run(self) -> Node | None:
        """Execute search, return best node or None."""
        ...
