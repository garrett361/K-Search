"""Simple tree formatter for LLM prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class SimpleStateFormatter:
    """Minimal tree formatting for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        return "\n".join(self.format_node(n) for n in tree._all_nodes())

    def format_node(self, node: Node) -> str:
        title = node.action.title if node.action else "(root)"
        score = (
            f" (score: {node.cycle.best_round.score:.2f})"
            if node.cycle and node.cycle.best_round
            else ""
        )
        return f"[{node._id}] {node.status}: {title}{score}"
