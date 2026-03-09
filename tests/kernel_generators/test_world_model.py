"""Integration tests for world model LLM response parsing."""

import os

import pytest

from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.kernel_generators.world_model import (
    build_world_model_prompts,
    load_world_model_obj,
    render_open_action_nodes_block,
    try_parse_world_model_json,
)
from k_search.tasks.gpu_mode.causal_conv1d.spec import CAUSAL_CONV1D_SPEC_TEXT_TRITON


def _endpoints_available() -> bool:
    """Check if endpoints config exists without raising."""
    try:
        from k_search.modular.llm import get_all_endpoints

        return bool(get_all_endpoints())
    except FileNotFoundError:
        return False


@pytest.fixture
def real_init_prompt() -> str:
    """Generate a real world model init prompt using actual task definition."""
    prompts = build_world_model_prompts(
        definition_text=CAUSAL_CONV1D_SPEC_TEXT_TRITON,
        target_gpu="H100",
        language="triton",
        previous_world_model_json=None,
        current_code_excerpt=None,
        eval_result=None,
        chosen_action_text=None,
        prediction=None,
    )
    return prompts.init_prompt


@pytest.mark.timeout(180)
@pytest.mark.skipif(
    not _endpoints_available() or not os.getenv("RITS_API_KEY"),
    reason="No endpoints configured or RITS_API_KEY not set",
)
class TestWorldModelLLMIntegration:
    """Integration tests for world model responses from actual LLM calls."""

    @pytest.fixture
    def generator(self) -> KernelGenerator:
        from k_search.modular.llm import get_all_endpoints, get_endpoint

        model_name = next(iter(get_all_endpoints().keys()))
        return KernelGenerator(
            model_name=model_name,
            api_key=os.environ["RITS_API_KEY"],
            base_url=get_endpoint(model_name),
        )

    def test_chat_completion_parses_to_valid_world_model(
        self, generator: KernelGenerator, real_init_prompt: str
    ):
        """Chat completion response should parse into valid world model JSON."""
        response = generator.client.chat.completions.create(
            model=generator.model_name,
            messages=[{"role": "user", "content": real_init_prompt}],
        )

        raw = response.choices[0].message.content or ""
        assert len(raw) > 0, "Response should not be empty"

        parsed = try_parse_world_model_json(raw)
        assert parsed is not None, (
            f"Failed to parse world model JSON. Response start:\n{raw[:1000]}"
        )

        obj = load_world_model_obj(parsed)
        assert obj is not None, "Failed to load world model object"
        assert "decision_tree" in obj, f"Missing decision_tree. Keys: {list(obj.keys())}"

    def test_chat_completion_has_executable_action_nodes(
        self, generator: KernelGenerator, real_init_prompt: str
    ):
        """Parsed world model should have executable action nodes (not '(none)')."""
        response = generator.client.chat.completions.create(
            model=generator.model_name,
            messages=[{"role": "user", "content": real_init_prompt}],
        )

        raw = response.choices[0].message.content or ""
        parsed = try_parse_world_model_json(raw)
        assert parsed is not None, f"Failed to parse:\n{raw[:500]}"

        rendered = render_open_action_nodes_block(parsed)
        assert "(none)" not in rendered, (
            f"No executable action nodes found.\n"
            f"Rendered: {rendered}\n"
            f"Raw response start:\n{raw[:1500]}"
        )

    def test_action_nodes_have_required_fields(
        self, generator: KernelGenerator, real_init_prompt: str
    ):
        """Action nodes should have title, score_0_to_1, difficulty_1_to_5."""
        response = generator.client.chat.completions.create(
            model=generator.model_name,
            messages=[{"role": "user", "content": real_init_prompt}],
        )

        raw = response.choices[0].message.content or ""
        parsed = try_parse_world_model_json(raw)
        assert parsed is not None

        obj = load_world_model_obj(parsed)
        assert obj is not None
        nodes = obj["decision_tree"]["nodes"]

        action_nodes = [
            n
            for n in nodes
            if isinstance(n.get("action"), dict) and n["action"].get("title")
        ]
        assert len(action_nodes) >= 1, f"No action nodes with titles. Nodes:\n{nodes}"

        for node in action_nodes:
            action = node["action"]
            node_id = node.get("node_id", "unknown")
            assert action.get("title"), f"Node {node_id} missing action.title"
            assert "score_0_to_1" in action, f"Node {node_id} missing action.score_0_to_1"
            assert "difficulty_1_to_5" in action, (
                f"Node {node_id} missing action.difficulty_1_to_5"
            )

    def test_reasoning_api_parses_to_valid_world_model(
        self, generator: KernelGenerator, real_init_prompt: str
    ):
        """Reasoning API response should also parse into valid world model JSON."""
        response = generator.client.responses.create(
            model=generator.model_name,
            input=real_init_prompt,
            reasoning={"effort": "medium"},
        )

        raw = response.output_text or ""
        assert len(raw) > 0, "Reasoning response should not be empty"

        parsed = try_parse_world_model_json(raw)
        assert parsed is not None, f"Failed to parse from reasoning API:\n{raw[:1000]}"

        rendered = render_open_action_nodes_block(parsed)
        assert "(none)" not in rendered, (
            f"No executable action nodes from reasoning API.\n"
            f"Rendered: {rendered}\n"
            f"Raw start:\n{raw[:1500]}"
        )
