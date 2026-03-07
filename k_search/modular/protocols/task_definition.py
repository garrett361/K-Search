"""TaskDefinition composite protocol."""

from typing import Any, Protocol

from k_search.modular.protocols.correctness import CorrectnessChecker
from k_search.modular.protocols.feedback_provider import FeedbackProvider
from k_search.modular.protocols.impl import Implementation
from k_search.modular.protocols.input_generator import InputGenerator
from k_search.modular.protocols.reference_impl import ReferenceImpl
from k_search.modular.protocols.scorer import Scorer


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

    def create_implementation(self, llm_output: str) -> Implementation:
        """Create Implementation from raw LLM output.

        Task handles: parsing (strip markdown), naming, language, entry points, build spec.
        """
        ...
