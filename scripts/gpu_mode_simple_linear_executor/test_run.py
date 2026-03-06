"""Tests for GPU mode executor script - prompt functions."""

from unittest.mock import MagicMock

from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

from scripts.gpu_mode_simple_linear_executor.run import (
    ACTION_PROMPT_TEMPLATE,
    create_action_prompt_fn,
    create_code_prompt_fn,
)


def test_action_prompt_first_round():
    """First round prompt includes task spec, no feedback."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    root = Node(status="closed")
    tree = Tree(root=root)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Optimize kernel X" in prompt
    assert "Task Specification" in prompt
    assert "Previous Best Result" not in prompt


def test_action_prompt_with_history():
    """Subsequent rounds include feedback from best round."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_task_def.feedback_provider.for_codegen.return_value = (
        "Previous attempt: loop tiling failed"
    )

    root = Node(status="closed")
    tree = Tree(root=root)

    best = Node(parent=root, status="closed")
    mock_round = MagicMock()
    mock_round.result.is_success.return_value = True
    mock_round.score = 0.8
    best.cycle = Cycle(rounds=[mock_round])
    tree.add_node(best)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Optimize kernel X" in prompt
    assert "Previous Best Result" in prompt
    assert "loop tiling failed" in prompt
    mock_task_def.feedback_provider.for_codegen.assert_called_once()


def test_action_prompt_no_successful_cycle():
    """No feedback if best node has no successful round."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    root = Node(status="closed")
    tree = Tree(root=root)

    failed = Node(parent=root, status="closed")
    failed.cycle = Cycle(rounds=[])
    tree.add_node(failed)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Previous Best Result" not in prompt


def test_code_prompt_includes_action():
    """Code prompt includes action title."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    node = Node(status="open", action=Action(title="try loop tiling"))

    prompt_fn = create_code_prompt_fn(mock_task_def)
    prompt = prompt_fn(node, mock_task_def)

    assert "Optimize kernel X" in prompt
    assert "try loop tiling" in prompt
    assert "Action:" in prompt


def test_code_prompt_no_action():
    """Code prompt for round 0 (no action) - direct generation."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    node = Node(status="open")

    prompt_fn = create_code_prompt_fn(mock_task_def)
    prompt = prompt_fn(node, mock_task_def)

    assert "Optimize kernel X" in prompt
    assert "Action:" not in prompt  # No action directive for round 0


def test_action_prompt_template_structure():
    """ACTION_PROMPT_TEMPLATE has expected placeholders."""
    assert "{task_spec}" in ACTION_PROMPT_TEMPLATE
    assert "{feedback_section}" in ACTION_PROMPT_TEMPLATE
    assert "one line" in ACTION_PROMPT_TEMPLATE.lower()
