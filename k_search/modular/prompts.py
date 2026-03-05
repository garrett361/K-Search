"""Prompt building utilities for search loop."""

from k_search.task_framework.llm_utils import strip_markdown_fences
from k_search.task_framework.protocols.task_definition import TaskDefinition
from k_search.modular.round import Round

__all__ = ["build_prompt", "strip_markdown_fences"]


def build_prompt(
    task: TaskDefinition,
    last_round: Round | None,
) -> str:
    """Build prompt for next code generation round.

    Args:
        task: Task definition with prompt text and feedback provider
        last_round: Previous round for feedback, or None for first round

    Returns:
        Complete prompt string for LLM
    """
    base = task.get_prompt_text()
    if last_round:
        feedback = task.feedback_provider.for_codegen(last_round)
        return f"{base}\n\n{feedback}"
    return base
