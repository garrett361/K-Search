"""TaskDefinition composite protocol."""

from typing import Any, Protocol

from k_search.task_framework.protocols.input_generator import InputGenerator
from k_search.task_framework.protocols.reference_impl import ReferenceImpl
from k_search.task_framework.protocols.correctness import CorrectnessChecker
from k_search.task_framework.protocols.scorer import Scorer
from k_search.task_framework.protocols.feedback_provider import FeedbackProvider


class TaskDefinition(Protocol):
    """Complete task definition."""

    name: str

    input_generator: InputGenerator
    correctness_checker: CorrectnessChecker
    scorer: Scorer
    feedback_provider: FeedbackProvider
    reference_impl: ReferenceImpl | None

    def get_prompt_text(self, context: dict[str, Any] | None = None) -> str:
        """Task description for LLM."""
        ...

    def get_test_cases(self) -> list[dict[str, Any]]:
        """Parameter sets for evaluation."""
        ...
