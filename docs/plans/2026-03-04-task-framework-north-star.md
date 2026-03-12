# Task Framework North-Star Architecture

Single-source visual reference consolidating `2026-03-04-task-framework-design.md`, `2026-03-04-implementation-protocol.md`, `2026-03-04-search-v2-design.md`, and `2026-03-05-tree-data-model-design.md`. Status: ✅ implemented, 🔲 planned.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SearchOrchestrator (modular) 🔲                           │
│                                                                              │
│  __init__(                                                                   │
│    task: TaskDefinition,                                                     │
│    evaluator: Evaluator,                                                     │
│    codegen_llm: Callable[[str], str],                                        │
│    world_model: WorldModel,                                                  │
│    formatter: StateFormatter,                                                │
│    config: SearchConfig                                                      │
│  )                                                                           │
│                                                                              │
│  run() -> Node | None                                                        │
│                                                                              │
│  Main loop:                                                                  │
│    1. node: Node = world_model.select(tree)                                  │
│    2. node.status = "in_progress"                                            │
│    3. cycle: Cycle = execute_cycle(node)  # multiple rounds                  │
│    4. node.cycle = cycle; node.status = "closed"                             │
│    5. world_model.update(tree)                                               │
│    6. new_node: Node = world_model.propose(tree)                             │
│    7. tree.add_node(new_node)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                      │
          │ uses               │ uses                 │ uses
          ▼                    ▼                      ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────────────────┐
│ WorldModel ✅    │  │ StateFormatter ✅│  │ TaskDefinition ✅ + Evaluator ✅ │
│ (Protocol)       │  │ (Protocol)      │  │ (from modular)                   │
│                  │  │                 │  │                                  │
│ propose(         │  │ format_tree(    │  │ See below                        │
│   tree: Tree,    │  │   tree: Tree    │  │                                  │
│   context: dict? │  │ ) -> str        │  │                                  │
│ ) -> Node        │  │                 │  │                                  │
│                  │  │ format_node(    │  │                                  │
│ select(          │  │   node: Node    │  │                                  │
│   tree: Tree,    │  │ ) -> str        │  │                                  │
│   context: dict? │  │                 │  │                                  │
│ ) -> Node        │  └─────────────────┘  │                                  │
│                  │                       │                                  │
│ update(          │  Implementations: 🔲  │                                  │
│   tree: Tree,    │  - LegacyJSONFmt      │                                  │
│   context: dict? │  - MarkdownFmt        │                                  │
│ ) -> None        │                       │                                  │
│                  │                       │                                  │
│ Impl: 🔲         │                       │                                  │
│ - LLMWorldModel  │                       │                                  │
└──────────────────┘                       └──────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        Tree (modular/world) ✅                               │
│  (Dataclass)                                                                 │
│                                                                              │
│  root: Node                                                                  │
│  annotations: dict[str, Any] | None                                          │
│                                                                              │
│  ┌───────────────────────────────┐  ┌───────────────────────────────────┐   │
│  │ Node (Dataclass) ✅           │  │ Action (Dataclass) ✅             │   │
│  │                               │  │                                   │   │
│  │ parent: Node | None           │  │ title: str                        │   │
│  │ children: list[Node]          │  │ annotations: dict | None          │   │
│  │ status: str                   │  └───────────────────────────────────┘   │
│  │ action: Action | None         │                                          │
│  │ cycle: Cycle | None           │  ┌───────────────────────────────────┐   │
│  │ annotations: dict | None      │  │ Cycle (Dataclass) ✅              │   │
│  └───────────────────────────────┘  │                                   │   │
│                                      │ rounds: list[Round]              │   │
│  get_frontier() -> list[Node]        │ best_round: Round | None (prop)  │   │
│  get_best_node() -> Node | None      │ succeeded: bool (property)       │   │
│  get_path_to_root(node) -> list[Node]│                                   │   │
│  add_node(node) -> None              └───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              TASK FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                           TaskDefinition ✅                                  │
│  (Protocol)                                                                  │
│                                                                              │
│  name: str                                                                   │
│  reference_impl: ReferenceImpl | None                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Atomic Protocols ✅                                                  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────┐  ┌───────────────────────────────┐    │   │
│  │  │ InputGenerator (Protocol) │  │ CorrectnessChecker (Protocol) │    │   │
│  │  │                           │  │                               │    │   │
│  │  │ generate(                 │  │ check(                        │    │   │
│  │  │   params: dict[str, Any], │  │   output: Any,                │    │   │
│  │  │   seed: int               │  │   reference: Any              │    │   │
│  │  │ ) -> Any                  │  │ ) -> CheckResult              │    │   │
│  │  └───────────────────────────┘  └───────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────┐  ┌───────────────────────────────┐    │   │
│  │  │ Scorer (Protocol)         │  │ FeedbackProvider (Protocol)   │    │   │
│  │  │                           │  │                               │    │   │
│  │  │ score(                    │  │ for_codegen(                  │    │   │
│  │  │   result: EvaluationResult│  │   outcomes: Round |           │    │   │
│  │  │ ) -> float                │  │            list[Round]        │    │   │
│  │  │                           │  │ ) -> str                      │    │   │
│  │  │                           │  │                               │    │   │
│  │  │                           │  │ for_world_model(              │    │   │
│  │  │                           │  │   outcomes: Round |           │    │   │
│  │  │                           │  │            list[Round]        │    │   │
│  │  │                           │  │ ) -> list[dict[str, Any]]     │    │   │
│  │  └───────────────────────────┘  └───────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  get_prompt_text(context: dict[str, Any] | None) -> str                      │
│  get_test_cases() -> list[dict[str, Any]]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ consumed by
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Executor (Protocol) 🔲                              │
│  Orchestration: sequential, parallel, pipelined                              │
│                                                                              │
│  __init__(evaluator: Evaluator, config: ExecutionConfig)                     │
│                                                                              │
│  execute(impl: Implementation) -> Round                                      │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ holds internally, delegates to
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Evaluator (Protocol) ✅                             │
│  Execution: loading, subprocess/in-process, timing                           │
│                                                                              │
│  evaluate(impl: Implementation) -> EvaluationResult                          │
│                                                                              │
│  Internal flow:                                                              │
│    1. input_data: Any = input_gen.generate(params: dict, seed: int)          │
│    2. expected: Any = _run(task.reference_impl, input_data)                  │
│    3. actual: Any = _run(impl: Implementation, input_data)                   │
│    4. check: CheckResult = checker.check(actual, expected)                   │
│    5. return EvaluationResult (wraps CheckResult + timing)                   │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       Implementation (Protocol) ✅                           │
│  Data container. No run() method. Evaluator knows how to execute it.         │
│                                                                              │
│  name: str                                                                   │
│  content: Any   # str | dict[str, str] | Path — task-specific                │
│                                                                              │
│  Both task.reference_impl and solution are Implementation:                   │
│    task.reference_impl: ReferenceImpl | None  (from TaskDefinition)          │
│    solution: Implementation                   (from LLM output)              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Types ✅                                     │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │ CheckResult (dataclass) ✅      │  │ EvaluationResult (Protocol) ✅  │   │
│  │                                 │  │                                 │   │
│  │ passed: bool                    │  │ succeeded() -> bool            │   │
│  │ message: str                    │  │ get_metrics() -> dict[str, Any] │   │
│  │ criteria: dict[str, Any] | None │  │ get_log() -> str                │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Round (dataclass) ✅           │  │ AnalysisResult (dataclass) ✅   │   │
│  │                                 │  │                                 │   │
│  │ impl: Implementation            │  │ summary: str                    │   │
│  │ result: EvaluationResult        │  │ metrics: dict[str, Any]         │   │
│  │ prompt: str                     │  │ raw_artifact: bytes | None      │   │
│  │ llm_response: str               │  │ strategic_guidance: str | None  │   │
│  │ prompt_tokens: int              │  └─────────────────────────────────┘   │
│  │ completion_tokens: int          │                                        │
│  │ duration_secs: float            │                                        │
│  │ score: float                    │                                        │
│  │ analysis: AnalysisResult | None │                                        │
│  └─────────────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Flow                                         │
│                                                                              │
│  EVALUATION PHASE (inside Executor -> Evaluator)                             │
│  ───────────────────────────────────────────────                             │
│  params: dict[str, Any], seed: int                                           │
│       │                                                                      │
│       ▼                                                                      │
│  InputGenerator.generate(params, seed) -> input_data: Any                    │
│       │                                                                      │
│       ▼                                                                      │
│  Evaluator._run(task.reference_impl, input_data) -> expected: Any            │
│  Evaluator._run(solution: Implementation, input_data) -> actual: Any         │
│       │                                                                      │
│       ▼                                                                      │
│  CorrectnessChecker.check(actual, expected) -> CheckResult                   │
│       │                                                                      │
│       ▼                                                                      │
│  Evaluator wraps CheckResult + timing -> EvaluationResult                    │
│       │                                                                      │
│       ▼                                                                      │
│  Executor wraps (impl: Implementation, result: EvaluationResult) -> Round│
│                                                                              │
│                                                                              │
│  POST-EVALUATION PHASE (consumed by run_search ✅ / SearchOrchestrator 🔲)   │
│  ────────────────────────────────────────────────────────────────────        │
│  EvaluationResult                                                            │
│       │                                                                      │
│       ├───────────────────────────────────────┐                              │
│       │                                       │                              │
│       ▼                                       ▼                              │
│  Scorer.score(                         Round(solution, result)         │
│    result: EvaluationResult                   │                              │
│  ) -> float                                   │                              │
│       │                                       ├────────────────────────┐     │
│       │                                       │                        │     │
│       ▼                                       ▼                        ▼     │
│  Tree.get_best_node() ✅          FeedbackProvider ✅     WorldModel ✅      │
│                                     for_codegen(...)        update(tree)     │
│                                     for_world_model(...)                     │
│                                       │                                      │
│                                       │                                      │
│                                       ├──► str (logs) ──► Codegen LLM        │
│                                       └──► list[dict] ──► Node.annotations   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    FeedbackProvider ✅ vs StateFormatter ✅                  │
│                                                                              │
│  FeedbackProvider.for_world_model(outcome) -> list[dict[str, Any]]           │
│    - Extracts metrics from ONE evaluation outcome                            │
│    - Stored in Node.annotations                                              │
│    - Called after each evaluation                                            │
│                                                                              │
│  StateFormatter.format_tree(tree) -> str                                     │
│    - Formats ENTIRE Tree for P_world prompt                                  │
│    - Used by WorldModel to see full search state                             │
│    - Called before each action selection                                     │
│                                                                              │
│  No overlap: one extracts metrics, one formats tree.                         │
└─────────────────────────────────────────────────────────────────────────────┘
```
