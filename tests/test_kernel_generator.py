"""Integration tests for kernel generator reasoning API support."""

import os

import pytest

from k_search.kernel_generators.kernel_generator import KernelGenerator


@pytest.mark.skipif(
    not os.getenv("RITS_API_KEY")
    or not os.getenv("RITS_BASE_URL")
    or not os.getenv("RITS_MODEL_NAME"),
    reason="RITS credentials not available",
)
class TestRITSAPIIntegration:
    """Integration tests with actual RITS API calls."""

    def test_reasoning_api_produces_reasoning_tokens(self):
        """Verify responses.create() produces reasoning tokens."""
        gen = KernelGenerator(
            model_name=os.environ["RITS_MODEL_NAME"],
            api_key=os.environ["RITS_API_KEY"],
            base_url=os.environ["RITS_BASE_URL"],
            use_reasoning_api=True,
        )

        prompt = "Write a Python function that computes factorial iteratively."
        response = gen.client.responses.create(
            model=gen.model_name, input=prompt, reasoning={"effort": "medium"}
        )

        assert response.output_text is not None
        assert len(response.output_text) > 0

        assert hasattr(response, "usage"), "Response should have usage statistics"
        assert hasattr(response.usage, "output_tokens_details"), (
            "Usage should have output_tokens_details"
        )
        reasoning_tokens = getattr(
            response.usage.output_tokens_details, "reasoning_tokens", 0
        )
        assert reasoning_tokens > 0, (
            f"Reasoning API should produce reasoning_tokens > 0, got {reasoning_tokens}"
        )

    def test_chat_completions_no_reasoning_tokens(self):
        """Verify chat.completions.create() does not produce reasoning tokens."""
        gen = KernelGenerator(
            model_name=os.environ["RITS_MODEL_NAME"],
            api_key=os.environ["RITS_API_KEY"],
            base_url=os.environ["RITS_BASE_URL"],
            use_reasoning_api=False,
        )

        prompt = "Write a Python function that returns 42."
        response = gen.client.chat.completions.create(
            model=gen.model_name, messages=[{"role": "user", "content": prompt}]
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
