"""FeedbackProvider protocol."""

from typing import Any, Protocol

from k_search.task_framework.types import EvalOutcome


class FeedbackProvider(Protocol):
    """Routes evaluation feedback to different LLM consumers."""

    def for_codegen(self, outcomes: EvalOutcome | list[EvalOutcome]) -> str:
        """Format outcomes as feedback for codegen LLM."""
        ...

    def for_world_model(
        self, outcomes: EvalOutcome | list[EvalOutcome]
    ) -> list[dict[str, Any]]:
        """Format outcomes for world model. Returns one dict per outcome."""
        ...
