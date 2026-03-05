"""V1-compatible JSON formatter for parity validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class LegacyJSONFormatter:
    """V1-compatible JSON format for parity validation."""

    def format_tree(self, tree: Tree) -> str:
        return json.dumps(self._tree_to_dict(tree), indent=2)

    def format_node(self, node: Node) -> str:
        return json.dumps(self._node_to_dict(node))

    def _tree_to_dict(self, tree: Tree) -> dict[str, Any]:
        nodes = [self._node_to_dict(n) for n in tree._all_nodes()]
        best = tree.get_best_node()
        return {
            "decision_tree": {
                "root_id": tree.root._id,
                "active_leaf_id": best._id if best else tree.root._id,
                "nodes": nodes,
            }
        }

    def _node_to_dict(self, node: Node) -> dict[str, Any]:
        action_dict = (
            {
                "title": node.action.title,
                "description": "",
                "annotations": node.action.annotations or {},
            }
            if node.action
            else None
        )
        score = (
            node.cycle.best_round.score
            if node.cycle and node.cycle.best_round
            else None
        )
        return {
            "node_id": node._id,
            "parent_id": node.parent._id if node.parent else None,
            "status": node.status,
            "action": action_dict,
            "score": score,
            "annotations": node.annotations or {},
        }
