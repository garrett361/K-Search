# V1 Case Search Script Design

Self-contained script implementing V1 case search with world model cycles using existing modular abstractions.

## Location

```
scripts/gpu_mode_case_search/
├── run.py           # V1 case search implementation
└── test_run.py      # Smoke tests
```

## Requirements

- Full world model (action tree, cycle-based search)
- Uses existing modular protocols (`WorldModel`, `Executor`, `Evaluator`, `TaskDefinition`)
- V1-style prompt routing (first attempt / debug / improve)
- Import prompts from `world_model_prompts`
- Dual stagnation tracking (no improvement + no improvement over base)
- Minimal observability (console only)
- No changes to core modular code

## Components

### CycleConfig

```python
@dataclass
class CycleConfig:
    max_attempts_per_action: int = 10      # Max rounds per action node
    stagnation_window: int = 5              # End after N non-improving rounds
    stagnation_over_base_window: int = 5    # End after N rounds not beating base
```

### V1WorldModel

Wraps `WorldModelManager` to implement `WorldModel` protocol:

```python
class V1WorldModel:
    # Protocol methods (public)
    def propose(self, tree: Tree, context: dict | None = None) -> list[Node]
    def select(self, tree: Tree, context: dict | None = None) -> list[Node]
    def update(self, tree: Tree, context: dict | None = None) -> None

    # V1-specific helpers (private)
    def _get_prompt_section(self, max_chars: int = 6000) -> str
    def _sync_frontier_from_manager(self, tree: Tree) -> list[Node]
    def _find_node_by_v1_id(self, tree: Tree, node_id: str) -> Node | None
```

### V1PromptBuilder

Handles V1-style prompt routing based on cycle phase:

```python
class V1PromptBuilder:
    def build(
        self,
        node: Node,
        task: TaskDefinition,
        attempt: int,
        last_round: Round | None,
        has_passed: bool,
    ) -> str
```

Prompt selection logic:
- **Attempt 1**: `get_generate_code_from_action_prompt_from_text` or `get_generate_code_from_spec_with_action_prompt_from_text`
- **Subsequent, no PASSED**: `get_debug_generated_code_prompt_from_text` or `get_debug_and_improve_from_spec_prompt_from_text`
- **Subsequent, has PASSED**: `get_improve_generated_code_prompt_from_text` or `get_improve_from_spec_prompt_from_text`

### V1SequentialExecutor

Cycle-based execution with stagnation detection:

```python
class V1SequentialExecutor:
    def __init__(
        self,
        world_model: V1WorldModel,
        task: TaskDefinition,
        evaluator: Evaluator,
        llm: Callable[[str], str],
        prompt_builder: V1PromptBuilder,
        tree: Tree,
        max_rounds: int,
        cycle_config: CycleConfig | None = None,
    )

    def run(self) -> Node | None
    def _run_cycle(self, node: Node, rounds_remaining: int) -> Cycle
    def _get_base_score(self, node: Node) -> float
```

Main loop:
1. `world_model.propose(tree)` - generate new action nodes
2. `world_model.select(tree)` - select next action
3. `_run_cycle(node)` - multiple attempts with stagnation
4. `world_model.update(tree)` - attach success or mark too hard

Cycle loop:
1. Build prompt (attempt-aware routing)
2. Generate code via LLM
3. Evaluate implementation
4. Update best score, check stagnation
5. Repeat until stagnation or budget exhausted

## Deviations from V1

| V1 Feature | Script Approach | Rationale |
|------------|-----------------|-----------|
| SolutionDB | In-memory tracking | Simplifies script, can add later |
| W&B + artifacts | Console only | Faster iteration, add observability later |
| JSON tree | Modular Tree/Node | Uses existing protocols |
| Inline loop | V1SequentialExecutor | Cleaner structure, reusable |

## CLI Arguments

```bash
python scripts/gpu_mode_case_search/run.py \
    --task causal_conv1d \
    --language triton \
    --max-rounds 50 \
    --model-name gpt-5 \
    --base-url <url> \
    --api-key <key> \
    --max-attempts 10 \
    --stagnation-window 5
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    V1SequentialExecutor                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  for each cycle:                                           │  │
│  │    1. world_model.propose(tree)                            │  │
│  │    2. world_model.select(tree) → node                      │  │
│  │    3. _run_cycle(node) → Cycle                             │  │
│  │       ┌─────────────────────────────────────────────────┐  │  │
│  │       │  for each attempt:                               │  │  │
│  │       │    prompt = prompt_builder.build(...)            │  │  │
│  │       │    code = llm(prompt)                            │  │  │
│  │       │    impl = task.create_impl(code)                 │  │  │
│  │       │    result = evaluator.evaluate(impl)             │  │  │
│  │       │    score = task.scorer.score(result)             │  │  │
│  │       │    check stagnation...                           │  │  │
│  │       └─────────────────────────────────────────────────┘  │  │
│  │    4. world_model.update(tree)                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│  return tree.get_best_node()                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Testing Strategy

- Smoke test: mock LLM, verify cycle mechanics
- Integration test: run against real task with small round budget
