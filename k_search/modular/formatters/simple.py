"""Default tree formatter for LLM prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from k_search.modular.world.node import Node
    from k_search.modular.world.tree import Tree


class DefaultFormatter:
    """File-tree-like formatting with full node info for LLM prompts."""

    def format_tree(self, tree: Tree) -> str:
        lines: list[str] = []
        self._format_subtree(tree.root, "", True, lines)
        return "\n".join(lines)

    def _format_subtree(
        self, node: Node, prefix: str, is_last: bool, lines: list[str]
    ) -> None:
        lines.append(self._format_node_line(node, prefix, is_last))
        children = node.children
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            if prefix == "":
                child_prefix = ""
            else:
                child_prefix = prefix[:-4] + ("    " if is_last else "│   ")
            self._format_subtree(
                child,
                child_prefix + ("└── " if child_is_last else "├── "),
                child_is_last,
                lines,
            )

    def _format_node_line(self, node: Node, prefix: str, is_last: bool) -> str:
        node_str = self.format_node(node)
        if prefix:
            return prefix[:-4] + ("└── " if is_last else "├── ") + node_str
        return node_str

    def format_node(self, node: Node) -> str:
        parts: list[str] = [f"id={node._id}"]
        parts.append(f"status={node.status}")

        title = node.action.title if node.action else "root"
        parts.append(f'title="{title}"')

        if node.annotations:
            ann_str = ", ".join(f"{k}: {v}" for k, v in node.annotations.items())
            parts.append(f"annotations={{{ann_str}}}")

        return "(" + ", ".join(parts) + ")"
