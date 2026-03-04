"""Prompt building utilities for search loop."""

from k_search.task_framework.protocols.task_definition import TaskDefinition
from k_search.task_framework.types import EvalOutcome
from k_search.task_framework.adapters.wrappers import GpuModeImplementation
from k_search.tasks.task_base import Solution, BuildSpec, SourceFile, SupportedLanguages


def build_prompt(
    task: TaskDefinition,
    last_outcome: EvalOutcome | None,
) -> str:
    """Build prompt for next code generation round.

    Args:
        task: Task definition with prompt text and feedback provider
        last_outcome: Previous round's outcome for feedback, or None for first round

    Returns:
        Complete prompt string for LLM
    """
    base = task.get_prompt_text()
    if last_outcome:
        feedback = task.feedback_provider.for_codegen(last_outcome)
        return f"{base}\n\n{feedback}"
    return base


def create_implementation(
    code: str,
    round_idx: int,
    task_name: str = "search_v2",
    language: str = "triton",
) -> GpuModeImplementation:
    """Create Implementation from generated code.

    Args:
        code: Generated kernel code
        round_idx: Current round index (used for naming)
        task_name: Name of the task
        language: Programming language (triton or cuda)

    Returns:
        GpuModeImplementation wrapping a Solution
    """
    lang_enum = SupportedLanguages(language)
    solution = Solution(
        name=f"{task_name}_r{round_idx}",
        definition=task_name,
        author="search_v2",
        spec=BuildSpec(
            language=lang_enum,
            target_hardware=[],
            entry_point="kernel.py::custom_kernel",
        ),
        sources=[SourceFile(path="kernel.py", content=code)],
    )
    return GpuModeImplementation(solution)
