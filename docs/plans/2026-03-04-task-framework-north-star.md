# Task Framework North-Star Architecture

Single-source visual reference consolidating `2026-03-04-task-framework-design.md`, `2026-03-04-implementation-protocol.md`, and `2026-03-04-search-v2-design.md`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SearchOrchestrator (search_v2)                        │
│                                                                              │
│  __init__(                                                                   │
│    task: TaskDefinition,                                                     │
│    executor: Executor,                                                       │
│    codegen_llm: Callable[[str], str],                                        │
│    selector: ActionSelector,                                                 │
│    formatter: StateFormatter,                                                │
│    config: SearchConfig                                                      │
│  )                                                                           │
│                                                                              │
│  run() -> SolutionNode | None                                                │
│                                                                              │
│  Main loop:                                                                  │
│    1. actions: list[ActionNode] = selector.select(tree, k=1)                 │
│    2. prompt: str = formatter.format_tree(tree) + task.get_prompt_text()     │
│    3. code: str = codegen_llm(prompt)                                        │
│    4. impl: Implementation = parse_code_to_impl(code)                        │
│    5. outcome: EvalOutcome = executor.execute(impl)                          │
│    6. selector.update(tree, action, outcome)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                      │
          │ uses               │ uses                 │ uses
          ▼                    ▼                      ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────────────────┐
│ ActionSelector   │  │ StateFormatter  │  │ TaskDefinition + Executor        │
│ (Protocol)       │  │ (Protocol)      │  │ (from task_framework)            │
│                  │  │                 │  │                                  │
│ propose_actions( │  │ format_tree(    │  │ See below                        │
│   tree: SolTree, │  │   tree: SolTree,│  │                                  │
│   context: dict? │  │   context: dict?│  │                                  │
│ )->list[ActNode] │  │ ) -> str        │  │                                  │
│                  │  │                 │  │                                  │
│ select(          │  │ format_frontier(│  │                                  │
│   tree: SolTree, │  │   actions: list │  │                                  │
│   k: int         │  │     [ActionNode]│  │                                  │
│ )->list[ActNode] │  │ ) -> str        │  │                                  │
│                  │  │                 │  │                                  │
│ update(          │  └─────────────────┘  │                                  │
│   tree: SolTree, │                       │                                  │
│   action: ActNode│  Implementations:     │                                  │
│   outcome: Eval- │  - LegacyJSONFmt      │                                  │
│     Outcome      │  - MarkdownFmt        │                                  │
│ ) -> None        │                       │                                  │
│                  │                       │                                  │
│ Implementations: │                       │                                  │
│ - LLMWorldModel  │                       │                                  │
│ - SimpleRefine   │                       │                                  │
└──────────────────┘                       └──────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       SolutionTree (search_v2/model)                         │
│  (Dataclass)                                                                 │
│                                                                              │
│  solutions: dict[str, SolutionNode]                                          │
│  actions: dict[str, ActionNode]                                              │
│  root_id: str                                                                │
│  active_leaf_id: str                                                         │
│                                                                              │
│  ┌───────────────────────────────┐  ┌───────────────────────────────────┐   │
│  │ SolutionNode (Dataclass)      │  │ ActionNode (Dataclass)            │   │
│  │                               │  │                                   │   │
│  │ id: str                       │  │ id: str                           │   │
│  │ parent_id: str | None         │  │ parent_solution_id: str           │   │
│  │ solution_id: str | None       │  │ title: str                        │   │
│  │ solution_content: Any         │  │ description: str                  │   │
│  │ eval_result: dict[str,Any]|None│  │ difficulty: int                   │   │
│  │ status: str                   │  │ predicted_score: float            │   │
│  │ depth: int                    │  │ status: str                       │   │
│  │ child_action_ids: list[str]   │  │ result_solution_id: str | None    │   │
│  └───────────────────────────────┘  └───────────────────────────────────┘   │
│                                                                              │
│  get_frontier() -> list[ActionNode]                                          │
│  get_best_solution() -> SolutionNode | None                                  │
│  get_path_to_root(node_id: str) -> list[SolutionNode]                        │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              TASK FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                              TaskDefinition                                  │
│  (Protocol)                                                                  │
│                                                                              │
│  name: str                                                                   │
│  reference: Implementation | None                                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Atomic Protocols                                                     │   │
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
│  │  │   result: EvaluationResult│  │   outcomes: EvalOutcome |     │    │   │
│  │  │ ) -> float                │  │            list[EvalOutcome]  │    │   │
│  │  │                           │  │ ) -> str                      │    │   │
│  │  │                           │  │                               │    │   │
│  │  │                           │  │ for_world_model(              │    │   │
│  │  │                           │  │   outcomes: EvalOutcome |     │    │   │
│  │  │                           │  │            list[EvalOutcome]  │    │   │
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
│                          Executor (Protocol)                                 │
│  Orchestration: sequential, parallel, pipelined                              │
│                                                                              │
│  __init__(evaluator: Evaluator, config: ExecutionConfig)                     │
│                                                                              │
│  execute(impl: Implementation) -> EvalOutcome                                │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ holds internally, delegates to
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Evaluator (Protocol)                                │
│  Execution: loading, subprocess/in-process, timing                           │
│                                                                              │
│  evaluate(impl: Implementation) -> EvaluationResult                          │
│                                                                              │
│  Internal flow:                                                              │
│    1. input_data: Any = input_gen.generate(params: dict, seed: int)          │
│    2. expected: Any = _run(task.reference: Implementation, input_data)       │
│    3. actual: Any = _run(impl: Implementation, input_data)                   │
│    4. check: CheckResult = checker.check(actual, expected)                   │
│    5. return EvaluationResult (wraps CheckResult + timing)                   │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       Implementation (Protocol)                              │
│  Data container. No run() method. Evaluator knows how to execute it.         │
│                                                                              │
│  name: str                                                                   │
│  content: Any   # str | dict[str, str] | Path — task-specific                │
│                                                                              │
│  Both task.reference and solution are Implementation:                        │
│    task.reference: Implementation | None  (from TaskDefinition)              │
│    solution: Implementation               (from LLM output)                  │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Types                                        │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │ CheckResult (dataclass)         │  │ EvaluationResult (Protocol)     │   │
│  │                                 │  │                                 │   │
│  │ passed: bool                    │  │ is_success() -> bool            │   │
│  │ message: str                    │  │ get_metrics() -> dict[str, Any] │   │
│  │ criteria: dict[str, Any] | None │  │ get_log() -> str                │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────┐                                        │
│  │ EvalOutcome (dataclass)         │                                        │
│  │                                 │                                        │
│  │ solution: Implementation        │                                        │
│  │ result: EvaluationResult        │                                        │
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
│  Evaluator._run(task.reference: Implementation, input_data) -> expected: Any │
│  Evaluator._run(solution: Implementation, input_data) -> actual: Any         │
│       │                                                                      │
│       ▼                                                                      │
│  CorrectnessChecker.check(actual, expected) -> CheckResult                   │
│       │                                                                      │
│       ▼                                                                      │
│  Evaluator wraps CheckResult + timing -> EvaluationResult                    │
│       │                                                                      │
│       ▼                                                                      │
│  Executor wraps (impl: Implementation, result: EvaluationResult) -> EvalOutcome│
│                                                                              │
│                                                                              │
│  POST-EVALUATION PHASE (consumed by SearchOrchestrator and protocols)        │
│  ────────────────────────────────────────────────────────────────────        │
│  EvaluationResult                                                            │
│       │                                                                      │
│       ├───────────────────────────────────────┐                              │
│       │                                       │                              │
│       ▼                                       ▼                              │
│  Scorer.score(                         EvalOutcome(solution, result)         │
│    result: EvaluationResult                   │                              │
│  ) -> float                                   │                              │
│       │                                       ├────────────────────────┐     │
│       │                                       │                        │     │
│       ▼                                       ▼                        ▼     │
│  SolutionTree.                    FeedbackProvider.       ActionSelector.    │
│    get_best_solution()              for_codegen(...)        update(tree,     │
│                                     for_world_model(...)      action,        │
│                                       │                       outcome)       │
│                                       │                                      │
│                                       ├──► str (logs) ──► Codegen LLM        │
│                                       └──► list[dict] ──► SolutionNode.      │
│                                                              eval_result     │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    FeedbackProvider vs StateFormatter                        │
│                                                                              │
│  FeedbackProvider.for_world_model(outcome) -> list[dict[str, Any]]           │
│    - Extracts metrics from ONE evaluation outcome                            │
│    - Stored in SolutionNode.eval_result                                      │
│    - Called after each evaluation                                            │
│                                                                              │
│  StateFormatter.format_tree(tree) -> str                                     │
│    - Formats ENTIRE SolutionTree for P_world prompt                          │
│    - Used by ActionSelector to see full search state                         │
│    - Called before each action selection                                     │
│                                                                              │
│  No overlap: one extracts metrics, one formats tree.                         │
└─────────────────────────────────────────────────────────────────────────────┘
```
