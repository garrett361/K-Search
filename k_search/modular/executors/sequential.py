"""Sequential executor - synchronous propose/select/execute/update loop."""

import logging
from collections.abc import Callable
from typing import Any

from k_search.modular.artifacts import NoOpArtifactStore
from k_search.modular.config import MetricsConfig
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.protocols import ArtifactStore, Evaluator, MetricsTracker
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.round import Round
from k_search.modular.world.tree import Tree

logger = logging.getLogger(__name__)

CodePromptFn = Callable[[Node, TaskDefinition], str]


class SequentialExecutor:
    """Reference executor - synchronous propose/select/execute/update loop."""

    def __init__(
        self,
        world_model: Any,
        task: TaskDefinition,
        evaluator: Evaluator,
        llm: Callable[[str], str],
        code_prompt_fn: CodePromptFn,
        tree: Tree,
        max_rounds: int,
        metrics_config: MetricsConfig | None = None,
        metrics_trackers: list[MetricsTracker] | None = None,
        artifact_stores: list[ArtifactStore] | None = None,
    ):
        self._world_model = world_model
        self._task = task
        self._evaluator = evaluator
        self._llm = llm
        self._code_prompt_fn = code_prompt_fn
        self._tree = tree
        self._max_rounds = max_rounds
        self._metrics_config = metrics_config or MetricsConfig()
        self._metrics_trackers = metrics_trackers or [NoOpMetricsTracker()]
        self._artifact_stores = artifact_stores or [NoOpArtifactStore()]

    def run(self) -> Node | None:
        """Execute search.

        Every round: propose action, then generate code from action.
        Round 0 proposes a generic initial action (no feedback yet).

        Termination: runs for max_rounds or until select() returns empty.
        TODO: termination responsibility (executor vs world model vs tree) not yet defined.
        """
        for round_idx in range(self._max_rounds):
            logger.info(f"Round {round_idx + 1}/{self._max_rounds}")
            logger.debug("=== ROUND %d START ===", round_idx + 1)

            proposed = self._world_model.propose(self._tree)
            logger.debug("World model proposed %d node(s)", len(proposed))

            for node in proposed:
                self._tree.add_node(node)

            nodes = self._world_model.select(self._tree)
            if not nodes:
                logger.info("No nodes to evaluate, stopping")
                break
            logger.debug("Selected %d node(s) for execution", len(nodes))

            for node in nodes:
                self._execute_node(node, round_idx)

            self._world_model.update(self._tree)
            logger.debug("=== ROUND %d END ===", round_idx + 1)

        return self._tree.get_best_node()

    def _execute_node(self, node: Node, round_idx: int) -> None:
        """Generate code for action and evaluate."""
        logger.debug("Node status: open -> in_progress")
        node.status = "in_progress"

        prompt = self._code_prompt_fn(node, self._task)
        code = self._llm(prompt)
        logger.debug("LLM code response:\n\n%s\n", code)

        impl = self._task.create_implementation(code)
        result = self._evaluator.evaluate(impl)
        score = self._task.scorer.score(result)

        logger.debug(
            "Evaluation result: success=%s, score=%.4f", result.is_success(), score
        )
        metrics = result.get_metrics()
        if metrics:
            logger.debug("Evaluation metrics:\n\n%s\n", metrics)

        chars_per_token = self._metrics_config.chars_per_token
        prompt_tokens = len(prompt) // chars_per_token
        completion_tokens = len(code) // chars_per_token

        round = Round(
            impl=impl,
            result=result,
            prompt=prompt,
            llm_response=code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_secs=0.0,
            score=score,
        )
        node.cycle = Cycle(rounds=[round])
        logger.debug("Node status: in_progress -> closed")
        node.status = "closed"

        for tracker in self._metrics_trackers:
            tracker.log({"score": score, "round_idx": round_idx}, step=round_idx)

        for store in self._artifact_stores:
            store.store(round, round_idx)

        logger.info(
            f"Round {round_idx + 1}: score={score:.4f}, success={result.is_success()}"
        )
