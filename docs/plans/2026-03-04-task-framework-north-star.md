# Task Framework North-Star Architecture

Single-source visual reference consolidating `2026-03-04-task-framework-design.md` and `2026-03-04-implementation-protocol.md`.

```
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
│  execute(                                                                    │
│    impl: Implementation,                                                     │
│    evaluator: Evaluator                                                      │
│  ) -> EvalOutcome                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ delegates to
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
│  EVALUATION PHASE                                                            │
│  ────────────────                                                            │
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
│                                                                              │
│                                                                              │
│  POST-EVALUATION PHASE                                                       │
│  ─────────────────────                                                       │
│  EvaluationResult                                                            │
│       │                                                                      │
│       ├───────────────────────────────────────┐                              │
│       │                                       │                              │
│       ▼                                       ▼                              │
│  Scorer.score(                         EvalOutcome(solution, result)         │
│    result: EvaluationResult                   │                              │
│  ) -> float                                   │                              │
│                                               ▼                              │
│                                    FeedbackProvider.for_codegen(             │
│                                      outcomes: EvalOutcome | list            │
│                                    ) -> str                                  │
│                                                                              │
│                                    FeedbackProvider.for_world_model(         │
│                                      outcomes: EvalOutcome | list            │
│                                    ) -> list[dict[str, Any]]                 │
└─────────────────────────────────────────────────────────────────────────────┘
```
