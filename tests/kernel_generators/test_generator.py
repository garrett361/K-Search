"""Integration tests for kernel generator reasoning API support."""

import os

import pytest

from k_search.kernel_generators.kernel_generator import KernelGenerator


def _endpoints_available() -> bool:
    """Check if endpoints config exists without raising."""
    try:
        from k_search.modular.llm import get_all_endpoints

        return bool(get_all_endpoints())
    except FileNotFoundError:
        return False


@pytest.mark.skipif(
    not _endpoints_available() or not os.getenv("RITS_API_KEY"),
    reason="No endpoints configured or RITS_API_KEY not set",
)
class TestRITSAPIIntegration:
    """Integration tests with actual RITS API calls."""

    @pytest.fixture
    def generator(self) -> KernelGenerator:
        from k_search.modular.llm import get_all_endpoints, get_endpoint

        model_name = next(iter(get_all_endpoints().keys()))
        return KernelGenerator(
            model_name=model_name,
            api_key=os.environ["RITS_API_KEY"],
            base_url=get_endpoint(model_name),
        )

    def test_reasoning_api_produces_reasoning_tokens(self, generator: KernelGenerator):
        """Verify responses.create() produces reasoning tokens."""
        prompt = "Write a Python function that computes factorial iteratively."
        response = generator.client.responses.create(
            model=generator.model_name, input=prompt, reasoning={"effort": "medium"}
        )

        assert response.output_text is not None
        assert len(response.output_text) > 0

        assert response.usage is not None
        assert response.usage.output_tokens_details is not None
        reasoning_tokens = getattr(
            response.usage.output_tokens_details, "reasoning_tokens", 0
        )
        assert reasoning_tokens > 0, (
            f"Reasoning API should produce reasoning_tokens > 0, got {reasoning_tokens}"
        )

    def test_chat_completions_no_reasoning_tokens(self, generator: KernelGenerator):
        """Verify chat.completions.create() does not produce reasoning tokens."""
        prompt = "Write a Python function that returns 42."
        response = generator.client.chat.completions.create(
            model=generator.model_name, messages=[{"role": "user", "content": prompt}]
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

        assert hasattr(response, "usage"), "Response should have usage statistics"
        reasoning_tokens = 0
        if hasattr(response.usage, "output_tokens_details"):
            reasoning_tokens = getattr(
                response.usage.output_tokens_details, "reasoning_tokens", 0
            )
        assert reasoning_tokens == 0, (
            f"Chat completions should not produce reasoning_tokens, got {reasoning_tokens}"
        )
