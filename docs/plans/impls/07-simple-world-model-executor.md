# SimpleWorldModel + SequentialExecutor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Context

Before implementing the full LLMWorldModel and PipelineExecutor, we validate the WorldModel and Executor protocols work end-to-end. This creates `SimpleWorldModel` and `SequentialExecutor` as the **reference implementation**:

1. Validates the propose/select/update protocol
2. Serves as a testbed for new features (land here first, propagate to complex cases)
3. Provides a working baseline to compare against

**Goal**: Prove protocol is viable with a two-step LLM pattern matching the current loop.

**Tech Stack:** Python 3.12, dataclasses, typing.Protocol

---

## Design

### Two-Step LLM Pattern

**Step 1 - propose()**: LLM generates action description
```
Input: task spec, feedback from previous rounds
LLM -> "try loop tiling with block size 32"
Output: Node with Action(title="try loop tiling with block size 32")
```

**Step 2 - execute**: LLM generates code from action
```
Input: task spec + node.action.title
Prompt: "{task_prompt}\n\nAction: {action.title}\n\nGenerate code:"
LLM -> implementation code
Output: evaluated Round attached to node
```

### Linear Tree (Linked List)

The tree is just a linked list for simplicity:

```
root -> node1 -> node2 -> node3 (latest)
```

- **propose()**: Add new node to end of list
- **select()**: Return the latest node (just added)
- No branching, no score-based selection

### Flow Per Round

```
propose() -> LLM generates action -> returns node (no tree mutation)
executor  -> adds node to tree (can filter here)
select()  -> returns [latest node]
execute   -> LLM generates code for action -> evaluate -> attach cycle
update()  -> pass
```

### Architecture

```
+-------------------------------------------------------------+
| Script (scripts/gpu_mode_simple_linear_executor/run.py)     |
|  - Defines prompt functions (action_prompt_fn, code_prompt_fn)
|  - Creates configs, metrics trackers, artifact stores       |
|  - Creates tree, task, evaluator, llm                       |
|  - Assembles executor with all dependencies                 |
|  - Calls executor.run()                                     |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
| SequentialExecutor                                          |
|  __init__(world_model, task, evaluator, llm,               |
|           code_prompt_fn, metrics_config, tree, max_rounds, |
|           metrics_trackers?, artifact_stores?)              |
|  run() -> Node | None                                       |
|    - propose -> add to tree -> select -> execute -> update  |
|    - logs metrics, stores artifacts                         |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
| SimpleWorldModel                                            |
|  __init__(llm, action_prompt_fn)                           |
|  propose(tree, context) -> list[Node]                      |
|  select(tree, context) -> list[Node]                       |
|  update(tree, context) -> None                             |
+-------------------------------------------------------------+
```

---

## Task 1: Create Executor protocol

**Files:**
- Create: `k_search/modular/protocols/executor.py`

```python
"""Executor protocol for search orchestration."""

from typing import Protocol

from k_search.modular.world.node import Node


class Executor(Protocol):
    """Search executor interface.

    Minimal protocol - all configuration passed via __init__.
    """

    def run(self) -> Node | None:
        """Execute search, return best node or None."""
        ...
```

**Commit:**
```bash
git add k_search/modular/protocols/executor.py
git commit -m "feat(protocols): add Executor protocol"
```

---

## Task 2: Update protocols __init__.py

**Files:**
- Modify: `k_search/modular/protocols/__init__.py`

Add import and export for `Executor`.

**Commit:**
```bash
git add k_search/modular/protocols/__init__.py
git commit -m "feat(protocols): export Executor"
```

---

## Task 3: Create world_models module

**Files:**
- Create: `k_search/modular/world_models/__init__.py`
- Create: `k_search/modular/world_models/simple.py`

**__init__.py:**
```python
"""World model implementations."""

from k_search.modular.world_models.simple import SimpleWorldModel

__all__ = ["SimpleWorldModel"]
```

**simple.py:**
```python
"""Simple world model for protocol validation."""

import logging
from collections.abc import Callable
from typing import Any

from k_search.modular.world.action import Action
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

logger = logging.getLogger(__name__)

ActionPromptFn = Callable[[Tree, dict[str, Any] | None], str]


class SimpleWorldModel:
    """Reference world model with two-step LLM pattern.

    - propose(): LLM generates action description, returns nodes (no tree mutation)
    - select(): Returns latest node (linear tree)
    - update(): No-op
    """

    def __init__(self, llm: Callable[[str], str], action_prompt_fn: ActionPromptFn):
        self._llm = llm
        self._action_prompt_fn = action_prompt_fn

    def propose(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Generate action via LLM, return node (don't add to tree)."""
        prompt = self._action_prompt_fn(tree, context)
        action_description = self._llm(prompt)

        parent = self._get_last_node(tree)

        node = Node(
            parent=parent,
            status="open",
            action=Action(title=action_description.strip()),
        )
        logger.info(f"Proposed action: {node.action.title[:50]}...")
        return [node]

    def select(self, tree: Tree, context: dict[str, Any] | None = None) -> list[Node]:
        """Return latest node (last in frontier)."""
        frontier = tree.get_frontier()
        return frontier[-1:] if frontier else []

    def update(self, tree: Tree, context: dict[str, Any] | None = None) -> None:
        """No-op for simple model."""
        pass

    def _get_last_node(self, tree: Tree) -> Node:
        """Get the last node in the linear chain."""
        node = tree.root
        while node.children:
            node = node.children[-1]
        return node
```

**Commit:**
```bash
git add k_search/modular/world_models/
git commit -m "feat(world_models): add SimpleWorldModel"
```

---

## Task 4: Create executors module

**Files:**
- Create: `k_search/modular/executors/__init__.py`
- Create: `k_search/modular/executors/sequential.py`

**__init__.py:**
```python
"""Executor implementations."""

from k_search.modular.executors.sequential import SequentialExecutor

__all__ = ["SequentialExecutor"]
```

**sequential.py:**
```python
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
        """Execute search."""
        for round_idx in range(self._max_rounds):
            logger.info(f"Round {round_idx + 1}/{self._max_rounds}")

            proposed = self._world_model.propose(self._tree)

            for node in proposed:
                self._tree.add_node(node)

            nodes = self._world_model.select(self._tree)
            if not nodes:
                logger.info("No nodes to evaluate, stopping")
                break

            for node in nodes:
                self._execute_node(node, round_idx)

            self._world_model.update(self._tree)

        return self._tree.get_best_node()

    def _execute_node(self, node: Node, round_idx: int) -> None:
        """Generate code for action and evaluate."""
        node.status = "in_progress"

        prompt = self._code_prompt_fn(node, self._task)
        code = self._llm(prompt)

        impl = self._task.create_impl(code)
        result = self._evaluator.evaluate(impl)
        score = self._task.scorer.score(result)

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
        node.status = "closed"

        for tracker in self._metrics_trackers:
            tracker.log({"score": score, "round_idx": round_idx}, step=round_idx)

        for store in self._artifact_stores:
            store.store(round, round_idx)

        logger.info(f"Round {round_idx + 1}: score={score:.4f}, success={result.succeeded()}")
```

**Commit:**
```bash
git add k_search/modular/executors/
git commit -m "feat(executors): add SequentialExecutor"
```

---

## Task 5: Write tests for SimpleWorldModel

**Files:**
- Create: `tests/modular/world_models/__init__.py`
- Create: `tests/modular/world_models/test_simple.py`

```python
"""Tests for SimpleWorldModel."""

from unittest.mock import MagicMock

from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel


def _simple_action_prompt_fn(tree, context):
    return "What to try next?"


def test_propose_creates_node_with_action():
    """propose() calls LLM and creates node with action (doesn't add to tree)."""
    mock_llm = MagicMock(return_value="try loop tiling")

    root = Node(status="closed")
    tree = Tree(root=root)
    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)

    nodes = model.propose(tree)

    assert len(nodes) == 1
    assert nodes[0].action.title == "try loop tiling"
    assert nodes[0].status == "open"
    assert nodes[0].parent is root
    assert nodes[0] not in root.children
    mock_llm.assert_called_once()


def test_propose_sets_parent_to_last_in_chain():
    """propose() sets parent to last node in linear chain."""
    mock_llm = MagicMock(return_value="action 2")

    root = Node(status="closed")
    tree = Tree(root=root)
    first = Node(parent=root, status="closed")
    tree.add_node(first)

    model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    nodes = model.propose(tree)

    assert nodes[0].parent is first


def test_propose_passes_context_to_prompt_fn():
    """propose() passes context to action_prompt_fn."""
    mock_llm = MagicMock(return_value="action")
    mock_prompt_fn = MagicMock(return_value="prompt")

    root = Node(status="closed")
    tree = Tree(root=root)
    model = SimpleWorldModel(mock_llm, mock_prompt_fn)

    context = {"round_idx": 5}
    model.propose(tree, context)

    mock_prompt_fn.assert_called_once_with(tree, context)


def test_select_returns_latest():
    """select() returns the latest open node."""
    root = Node(status="closed")
    tree = Tree(root=root)

    node1 = Node(parent=root, status="closed")
    node2 = Node(parent=root, status="open")
    tree.add_node(node1)
    tree.add_node(node2)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(tree)

    assert selected == [node2]


def test_select_empty_frontier():
    """select() returns empty list when no open nodes."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    selected = model.select(tree)

    assert selected == []


def test_update_is_noop():
    """update() does nothing."""
    root = Node(status="closed")
    tree = Tree(root=root)

    model = SimpleWorldModel(MagicMock(), _simple_action_prompt_fn)
    model.update(tree)
```

**Run tests:**
```bash
pytest tests/modular/world_models/test_simple.py -v --tb=short
```

**Commit:**
```bash
git add tests/modular/world_models/
git commit -m "test(world_models): add SimpleWorldModel tests"
```

---

## Task 6: Write tests for SequentialExecutor

**Files:**
- Create: `tests/modular/executors/__init__.py`
- Create: `tests/modular/executors/test_sequential.py`

```python
"""Tests for SequentialExecutor."""

from unittest.mock import MagicMock

from k_search.modular.executors.sequential import SequentialExecutor
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel


def _mock_task():
    """Create mock task."""
    task = MagicMock()
    task.get_prompt_text.return_value = "optimize kernel"
    task.create_impl.return_value = MagicMock(name="impl")
    task.scorer.score.return_value = 0.5
    return task


def _mock_evaluator(success=True):
    """Create mock evaluator."""
    evaluator = MagicMock()
    result = MagicMock()
    result.succeeded.return_value = success
    result.get_metrics.return_value = {}
    result.get_log.return_value = ""
    evaluator.evaluate.return_value = result
    return evaluator


def _simple_action_prompt_fn(tree, context):
    return "What to try next?"


def _simple_code_prompt_fn(node, task):
    action_title = node.action.title if node.action else "implement"
    return f"{task.get_prompt_text()}\n\nAction: {action_title}\n\nGenerate:"


def test_run_completes_rounds():
    """Executor runs for max_rounds."""
    root = Node(status="closed")
    tree = Tree(root=root)

    llm_responses = iter(["action 1", "code 1", "action 2", "code 2", "action 3", "code 3"])
    mock_llm = MagicMock(side_effect=lambda p: next(llm_responses))

    task = _mock_task()
    world_model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model, task, evaluator, mock_llm,
        _simple_code_prompt_fn, tree, max_rounds=3
    )
    executor.run()

    assert mock_llm.call_count == 6
    assert evaluator.evaluate.call_count == 3


def test_run_adds_proposed_nodes_to_tree():
    """Executor adds proposed nodes to tree."""
    root = Node(status="closed")
    tree = Tree(root=root)

    mock_llm = MagicMock(return_value="response")
    task = _mock_task()
    world_model = SimpleWorldModel(mock_llm, _simple_action_prompt_fn)
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model, task, evaluator, mock_llm,
        _simple_code_prompt_fn, tree, max_rounds=1
    )
    executor.run()

    assert len(root.children) == 1
    node = root.children[0]
    assert node.status == "closed"
    assert node.cycle is not None
    assert len(node.cycle.rounds) == 1


def test_run_stops_on_empty_select():
    """Executor stops when select returns empty."""
    root = Node(status="closed")
    tree = Tree(root=root)

    world_model = MagicMock()
    world_model.propose.return_value = []
    world_model.select.return_value = []

    mock_llm = MagicMock()
    task = _mock_task()
    evaluator = _mock_evaluator()

    executor = SequentialExecutor(
        world_model, task, evaluator, mock_llm,
        _simple_code_prompt_fn, tree, max_rounds=10
    )
    executor.run()

    assert evaluator.evaluate.call_count == 0
```

**Run tests:**
```bash
pytest tests/modular/executors/test_sequential.py -v --tb=short
```

**Commit:**
```bash
git add tests/modular/executors/
git commit -m "test(executors): add SequentialExecutor tests"
```

---

## Task 7: Create GPU Mode script with prompt functions and tests

**Files:**
- Create: `scripts/gpu_mode_simple_linear_executor/__init__.py`
- Create: `scripts/gpu_mode_simple_linear_executor/run.py`
- Create: `scripts/gpu_mode_simple_linear_executor/test_run.py`

**scripts/gpu_mode_simple_linear_executor/__init__.py:**
```python
"""GPU mode simple linear executor scripts."""
```

**scripts/gpu_mode_simple_linear_executor/run.py:**
```python
#!/usr/bin/env python3
"""GPU mode executor entry point - SimpleWorldModel + SequentialExecutor.

All code in one file for easy one-off scripting.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import openai

from k_search.modular.adapters import GpuModeEvaluator, GpuModeTriMulTaskDefinition
from k_search.modular.executors import SequentialExecutor
from k_search.modular.protocols.task_definition import TaskDefinition
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree
from k_search.modular.world_models.simple import SimpleWorldModel
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


ACTION_PROMPT_TEMPLATE = """You are proposing the next optimization action for a GPU kernel.

## Task Specification
{task_spec}
{feedback_section}
## Your Job
Propose ONE specific optimization action to try next.

Rules:
- Single-iteration implementable, SMALL change
- One concrete tweak (tiling OR memory OR scheduling - not multiple)
- Be specific (e.g., "use shared memory for input tile" not "optimize memory")

Respond with only the action title (one line, no explanation)."""


def create_action_prompt_fn(task_def: GpuModeTriMulTaskDefinition):
    """Create GPU mode specific action prompt function.

    Uses feedback from best round to inform next action proposal.
    """
    def action_prompt_fn(tree: Tree, context: dict[str, Any] | None) -> str:
        task_spec = task_def.get_prompt_text()

        feedback_section = ""
        best_node = tree.get_best_node()
        if best_node and best_node.cycle:
            best_round = best_node.cycle.best_round
            if best_round:
                feedback = task_def.feedback_provider.for_codegen(best_round)
                feedback_section = f"\n## Previous Best Result\n{feedback}\n"

        return ACTION_PROMPT_TEMPLATE.format(
            task_spec=task_spec,
            feedback_section=feedback_section,
        )

    return action_prompt_fn


def create_code_prompt_fn(task_def: GpuModeTriMulTaskDefinition):
    """Create GPU mode specific code generation prompt function."""
    def code_prompt_fn(node: Node, task: TaskDefinition) -> str:
        action_title = node.action.title if node.action else "implement solution"
        return f"{task_def.get_prompt_text()}\n\nAction: {action_title}\n\nGenerate the implementation:"

    return code_prompt_fn


def create_llm_call(client: openai.OpenAI, model_name: str):
    """Create LLM callable."""
    def llm_call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    return llm_call


def main():
    parser = argparse.ArgumentParser(description="Run GPU mode executor")
    parser.add_argument("--task", required=True, help="Task name (e.g., causal_conv1d)")
    parser.add_argument("--language", default="triton", choices=["triton", "cuda"])
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (or set LLM_API_KEY)")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("API key required (--api-key or LLM_API_KEY)")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent.parent
    task_dir = repo_root / "k_search" / "tasks" / "gpu_mode" / args.task
    if not task_dir.exists():
        logger.error(f"Task not found: {task_dir}")
        sys.exit(1)

    logger.info(f"Loading task: {args.task}")
    gpu_task = GpuModeTriMulTask(name=args.task, task_dir=task_dir)
    task_def = GpuModeTriMulTaskDefinition(gpu_task, language=args.language)
    evaluator = GpuModeEvaluator(gpu_task)

    client_kwargs = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    client = openai.OpenAI(**client_kwargs)
    llm = create_llm_call(client, args.model_name)

    action_prompt_fn = create_action_prompt_fn(task_def)
    code_prompt_fn = create_code_prompt_fn(task_def)

    world_model = SimpleWorldModel(llm, action_prompt_fn)
    tree = Tree(root=Node(status="closed"))

    executor = SequentialExecutor(
        world_model=world_model,
        task=task_def,
        evaluator=evaluator,
        llm=llm,
        code_prompt_fn=code_prompt_fn,
        tree=tree,
        max_rounds=args.max_rounds,
    )

    logger.info(f"Starting executor: max_rounds={args.max_rounds}, model={args.model_name}")
    best_node = executor.run()

    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    if best_node and best_node.cycle and best_node.cycle.best_round:
        best_round = best_node.cycle.best_round
        logger.info(f"Best score: {best_round.score:.4f}")
        metrics = best_round.result.get_metrics()
        logger.info(f"Speedup: {metrics.get('speedup_factor', 'N/A')}")
    else:
        logger.info("No successful solution found")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

**scripts/gpu_mode_simple_linear_executor/test_run.py:**
```python
"""Tests for GPU mode executor script - prompt functions."""

from unittest.mock import MagicMock

from k_search.modular.world.cycle import Cycle
from k_search.modular.world.node import Node
from k_search.modular.world.tree import Tree

from scripts.gpu_mode_simple_linear_executor.run import (
    ACTION_PROMPT_TEMPLATE,
    create_action_prompt_fn,
    create_code_prompt_fn,
)


def test_action_prompt_first_round():
    """First round prompt includes task spec, no feedback."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    root = Node(status="closed")
    tree = Tree(root=root)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Optimize kernel X" in prompt
    assert "Task Specification" in prompt
    assert "Previous Best Result" not in prompt


def test_action_prompt_with_history():
    """Subsequent rounds include feedback from best round."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"
    mock_task_def.feedback_provider.for_codegen.return_value = "Previous attempt: loop tiling failed"

    root = Node(status="closed")
    tree = Tree(root=root)

    best = Node(parent=root, status="closed")
    mock_round = MagicMock()
    mock_round.result.succeeded.return_value = True
    mock_round.score = 0.8
    best.cycle = Cycle(rounds=[mock_round])
    tree.add_node(best)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Optimize kernel X" in prompt
    assert "Previous Best Result" in prompt
    assert "loop tiling failed" in prompt
    mock_task_def.feedback_provider.for_codegen.assert_called_once()


def test_action_prompt_no_successful_cycle():
    """No feedback if best node has no successful round."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    root = Node(status="closed")
    tree = Tree(root=root)

    failed = Node(parent=root, status="closed")
    failed.cycle = Cycle(rounds=[])
    tree.add_node(failed)

    prompt_fn = create_action_prompt_fn(mock_task_def)
    prompt = prompt_fn(tree, None)

    assert "Previous Best Result" not in prompt


def test_code_prompt_includes_action():
    """Code prompt includes action title."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    from k_search.modular.world.action import Action
    node = Node(status="open", action=Action(title="try loop tiling"))

    prompt_fn = create_code_prompt_fn(mock_task_def)
    prompt = prompt_fn(node, mock_task_def)

    assert "Optimize kernel X" in prompt
    assert "try loop tiling" in prompt
    assert "Action:" in prompt


def test_code_prompt_no_action():
    """Code prompt handles missing action."""
    mock_task_def = MagicMock()
    mock_task_def.get_prompt_text.return_value = "Optimize kernel X"

    node = Node(status="open")

    prompt_fn = create_code_prompt_fn(mock_task_def)
    prompt = prompt_fn(node, mock_task_def)

    assert "implement solution" in prompt
```

**Run tests:**
```bash
pytest scripts/gpu_mode_simple_linear_executor/test_run.py -v --tb=short
```

**Commit:**
```bash
git add scripts/
git commit -m "feat(scripts): add gpu_mode_simple_linear_executor with tests"
```

---

## Task 8: Final verification

```bash
pytest tests/modular/world_models/ tests/modular/executors/ scripts/gpu_mode_simple_linear_executor/ -v --tb=short

ty check k_search/modular/world_models/ k_search/modular/executors/ k_search/modular/protocols/

ruff check k_search/modular/world_models/ k_search/modular/executors/
ruff format k_search/modular/world_models/ k_search/modular/executors/
```

---

## Files Summary

**Create:**
- `k_search/modular/protocols/executor.py`
- `k_search/modular/world_models/__init__.py`
- `k_search/modular/world_models/simple.py`
- `k_search/modular/executors/__init__.py`
- `k_search/modular/executors/sequential.py`
- `tests/modular/world_models/__init__.py`
- `tests/modular/world_models/test_simple.py`
- `tests/modular/executors/__init__.py`
- `tests/modular/executors/test_sequential.py`
- `scripts/gpu_mode_simple_linear_executor/__init__.py`
- `scripts/gpu_mode_simple_linear_executor/run.py`
- `scripts/gpu_mode_simple_linear_executor/test_run.py`

**Modify:**
- `k_search/modular/protocols/__init__.py`

**Total:** ~8 commits

## Notes

- `loop.py` remains unchanged - SequentialExecutor is parallel implementation
- Prompt functions are generic callables - script owns the specifics
- This is the reference implementation - new features land here first
