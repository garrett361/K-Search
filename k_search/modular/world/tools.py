"""Tool schemas and application for tree operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.parse_result import ParseResult

if TYPE_CHECKING:
    from k_search.modular.world.tree import Tree

TREE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "insert_node",
            "description": "Add a new action node to the tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {"type": "string", "description": "ID of parent node"},
                    "title": {"type": "string", "description": "Action title"},
                    "annotations": {
                        "type": "object",
                        "description": "Optional metadata",
                    },
                },
                "required": ["parent_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node",
            "description": "Update annotations on an existing node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of node to update",
                    },
                    "annotations": {
                        "type": "object",
                        "description": "Annotations to merge",
                    },
                },
                "required": ["node_id", "annotations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_node",
            "description": "Split a node into multiple child actions",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of node to split"},
                    "children": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "annotations": {"type": "object"},
                            },
                            "required": ["title"],
                        },
                        "description": "Child actions to create",
                    },
                },
                "required": ["node_id", "children"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_node",
            "description": "Mark a node as deleted",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of node to delete",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_node",
            "description": "Select a node to pursue next",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of node to select",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
]


def get_tree_tools(enabled: set[str] | None = None) -> list[dict[str, Any]]:
    """Return tool schemas, optionally filtered to enabled set."""
    if enabled is None:
        return TREE_TOOLS
    return [t for t in TREE_TOOLS if t["function"]["name"] in enabled]


def apply_tool_call(
    tree: Tree, tool_name: str, args: dict[str, Any]
) -> ParseResult[Node]:
    """Route tool call to Tree method. Returns ParseResult for error handling."""
    if tool_name == "insert_node":
        parent = tree._get_node_by_id(args.get("parent_id", ""))
        if parent is None:
            return ParseResult.fail(f"parent_id not found: {args.get('parent_id')}")
        node = Node(
            parent=parent,
            action=Action(title=args["title"], annotations=args.get("annotations")),
        )
        tree.add_node(node)
        return ParseResult.ok(node)

    if tool_name == "update_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.update_node(node, args["annotations"])
        return ParseResult.ok(node)

    if tool_name == "split_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        children = tree.split_node(node, args.get("children", []))
        return ParseResult.ok(children[0] if children else node)

    if tool_name == "delete_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        tree.delete_node(node)
        return ParseResult.ok(node)

    if tool_name == "select_node":
        node = tree._get_node_by_id(args.get("node_id", ""))
        if node is None:
            return ParseResult.fail(f"node_id not found: {args.get('node_id')}")
        return ParseResult.ok(node)

    return ParseResult.fail(f"unknown tool: {tool_name}")
