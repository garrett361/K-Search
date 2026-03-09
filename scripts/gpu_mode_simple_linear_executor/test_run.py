"""Tests for GPU mode executor script - prompt functions."""

from unittest.mock import MagicMock

from k_search.modular.world.action import Action
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

from scripts.gpu_mode_simple_linear_executor.run import (
    ACTION_PROMPT_TEMPLATE,
    _extract_error_hint,
    create_action_prompt_fn,
    create_code_prompt_fn,
)


def test_action_prompt_first_round():
    """First round prompt includes task spec, no feedback."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_llm = MagicMock(return_value="analysis")

    root = Node(status="closed")
    tree = Tree(root=root)

    prompt_fn = create_action_prompt_fn(mock_task_def, mock_llm)
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
    mock_llm = MagicMock(return_value="analysis")

    root = Node(status="closed")
    tree = Tree(root=root)

    best = Node(parent=root, status="closed")
    mock_round = MagicMock()
    mock_round.result.succeeded.return_value = True
    mock_round.score = 0.8
    best.cycle = Cycle(rounds=[mock_round])
    tree.add_node(best)

    prompt_fn = create_action_prompt_fn(mock_task_def, mock_llm)
    prompt = prompt_fn(tree, None)

    assert "Optimize kernel X" in prompt
    assert "Previous Best Result" in prompt
    assert "loop tiling failed" in prompt
    mock_task_def.feedback_provider.for_codegen.assert_called_once()


def test_action_prompt_no_successful_cycle():
    """No feedback if best node has no successful round."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_llm = MagicMock(return_value="analysis")

    root = Node(status="closed")
    tree = Tree(root=root)

    failed = Node(parent=root, status="closed")
    failed.cycle = Cycle(rounds=[])
    tree.add_node(failed)

    prompt_fn = create_action_prompt_fn(mock_task_def, mock_llm)
    prompt = prompt_fn(tree, None)

    assert "Previous Best Result" not in prompt


def test_code_prompt_includes_action():
    """Code prompt includes action title."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_llm = MagicMock(return_value="analysis")

    node = Node(status="open", action=Action(title="try loop tiling"))
    tree = Tree(root=Node(status="closed"))

    prompt_fn = create_code_prompt_fn(mock_task_def, tree, mock_llm)
    prompt = prompt_fn(node, mock_task_def)

    assert "Optimize kernel X" in prompt
    assert "try loop tiling" in prompt
    assert "Action:" in prompt


def test_code_prompt_no_action():
    """Code prompt for round 0 (no action) - direct generation."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_llm = MagicMock(return_value="analysis")

    node = Node(status="open")
    tree = Tree(root=Node(status="closed"))

    prompt_fn = create_code_prompt_fn(mock_task_def, tree, mock_llm)
    prompt = prompt_fn(node, mock_task_def)

    assert "Optimize kernel X" in prompt
    assert "Action:" not in prompt  # No action directive for round 0


def test_action_prompt_template_structure():
    """ACTION_PROMPT_TEMPLATE has expected placeholders."""
    assert "{task_spec}" in ACTION_PROMPT_TEMPLATE
    assert "{feedback_section}" in ACTION_PROMPT_TEMPLATE
    assert "{last_round_section}" in ACTION_PROMPT_TEMPLATE
    assert "one line" in ACTION_PROMPT_TEMPLATE.lower()


def test_extract_error_hint_finds_error():
    """Extracts first error line from log."""
    log = """Running kernel...
CUDA error: misaligned address at line 42
Traceback follows"""
    hint = _extract_error_hint(log)
    assert hint == "CUDA error: misaligned address at line 42"


def test_action_prompt_includes_last_failure():
    """Action prompt includes last round failure info when analyze_failures=True."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_llm = MagicMock(return_value="Index was out of bounds due to wrong tile size")

    root = Node(status="closed")
    tree = Tree(root=root)

    failed = Node(
        parent=root, status="closed", action=Action(title="try vectorization")
    )
    mock_round = MagicMock()
    mock_round.result.succeeded.return_value = False
    mock_round.result.get_log.return_value = "Error: index out of bounds"
    mock_round.score = 0.0
    failed.cycle = Cycle(rounds=[mock_round])
    tree.add_node(failed)

    prompt_fn = create_action_prompt_fn(mock_task_def, mock_llm, analyze_failures=True)
    prompt = prompt_fn(tree, None)

    assert "Last Round (FAILED)" in prompt
    assert "try vectorization" in prompt
    assert "Lesson:" in prompt
    mock_llm.assert_called_once()
