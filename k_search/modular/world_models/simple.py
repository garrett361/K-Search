"""Simple world model for protocol validation."""

import logging
from collections.abc import Callable
from typing import Any

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

logger = logging.getLogger(__name__)

ActionPromptFn = Callable[[Tree, dict[str, Any] | None], str]


INITIAL_ACTION = "Write an optimized implementation."


class SimpleWorldModel:
    """Reference world model with two-step LLM pattern.

    - propose(): LLM generates action description, returns nodes (no tree mutation)
    - select(): Returns latest node (linear tree)
    - update(): No-op

    Round 0 uses a generic initial action. Round 1+ asks LLM for specific actions.
    """

    def __init__(self, llm: Callable[[str], str], action_prompt_fn: ActionPromptFn):
        self._llm = llm
        self._action_prompt_fn = action_prompt_fn

    def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Generate action via LLM, return node (don't add to tree)."""
        parent = self._get_last_node(tree)

        # Round 0: use generic initial action (no LLM call)
        # Round 1+: ask LLM for specific action based on feedback
        if tree.get_best_node() is None:
            action_description = INITIAL_ACTION
        else:
            prompt = self._action_prompt_fn(tree, context)
            action_description = self._llm(prompt).strip()

        action = Action(title=action_description)
        node = Node(parent=parent, status="open", action=action)
        logger.info(f"Proposed action: {action.title[:50]}...")
        return [node]

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Return latest node (last in frontier)."""
        frontier = tree.get_frontier()
        return frontier[-1:] if frontier else []

    def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """No-op for simple model."""
        pass

    def _get_last_node(self, tree: Tree) -> Node:
        """Get the last node in the linear chain."""
        node = tree.root
        while node.children:
            if len(node.children) > 1:
                raise ValueError(
                    f"SimpleWorldModel expects linear tree, but node has {len(node.children)} children"
                )
            node = node.children[-1]
        return node
