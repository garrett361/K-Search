"""API compatibility test for tool calling with RITS."""

import json
import os
from typing import Any

import openai
import pytest

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "insert_node",
        "description": "Add a new action node to the tree",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {"type": "string", "description": "ID of parent node"},
                "title": {"type": "string", "description": "Action title"},
            },
            "required": ["parent_id", "title"],
        },
    },
]


@pytest.mark.timeout(180)
@pytest.mark.skipif(
    not os.getenv("RITS_API_KEY")
    or not os.getenv("RITS_BASE_URL")
    or not os.getenv("RITS_MODEL_NAME"),
    reason="RITS credentials not available",
)
class TestToolCallingAPI:
    """Integration tests for tool calling with RITS API."""

    def test_tools_produce_valid_tool_calls(self):
        """Verify tool schema works with responses API."""
        client = openai.OpenAI(
            base_url=os.environ["RITS_BASE_URL"],
            api_key="unused",
            default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
        )

        response = client.responses.create(
            model=os.environ["RITS_MODEL_NAME"],
            instructions="You are a tree editor. Use insert_node to add nodes.",
            input="Add a node called 'Optimize memory' under the root (id 0).",
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1
        tool_call = tool_calls[0]
        assert tool_call.name == "insert_node"

        args = json.loads(tool_call.arguments)
        assert "parent_id" in args
        assert "title" in args
