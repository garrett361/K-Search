"""Execution context for Node lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field

from k_search.modular.timer import Timer
from k_search.modular.world.node import Node


@dataclass
class Span:
    """Execution context for a Node's lifecycle.

    Wraps a Node, owns timing via Timer, extensible for future attributes.
    Passive container — executor manages lifecycle.
    """

    node: Node
    timer: Timer = field(default_factory=Timer)

    def get_metrics(self) -> dict[str, float]:
        """Return metrics dict suitable for MetricsTracker.log()."""
        return self.timer.get_timing_secs()
