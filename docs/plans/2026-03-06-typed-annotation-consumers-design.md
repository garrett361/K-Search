# Typed Annotation Consumers Design

Extends the typed annotations system (see `2026-03-05-typed-annotations-design.md`) so consumers can introspect annotation schemas for tool generation.

## Problem

With typed annotations, `Node[V1Ann]` validates annotations at construction. But when generating LLM tool prompts for node-modification operations, we need to:

1. Introspect the annotation schema from existing nodes
2. Generate tool parameter descriptions listing valid keys/types
3. Include these constraints in LLM prompts so the model knows what it can set

## Design Constraint

**Homogeneous trees only.** All nodes in a tree share the same annotation type. Multi-stage or ensemble searches requiring different annotation schemas must use separate trees.

Escape hatches if needed later:
- Separate trees per stage
- Union type `Tree[V1Ann | V2Ann]` with runtime discrimination
- Optional fields via `total=False` TypedDict

## Solution

Add `_annotation_schema()` classmethod to `_AnnotationMixin`, co-located with existing `_validate()`. Both use `__orig_bases__` introspection.

How the schema flows into tool definitions is **deferred** - depends on tool abstractions not yet designed. This design covers introspection only.

## Implementation

### `annotations.py` additions

```python
from typing import get_type_hints, get_args

class _AnnotationMixin(Generic[T]):
    # ... existing _adapter, __init_subclass__, _validate ...

    @classmethod
    def _annotation_schema(cls) -> dict[str, type]:
        """Return annotation TypedDict fields for this class."""
        for base in getattr(cls, "__orig_bases__", []):
            args = get_args(base)
            if args and args[0] is not AnnDict:
                return get_type_hints(args[0])
        return {}
```

## Flow

```
V1Ann (TypedDict)
    |
    v
V1Node(Node[V1Ann]) validates at construction
    |
    v
node._annotation_schema() -> {"overall_rating_0_10": int, "reasoning": str}
    |
    v
(Future: tool abstraction consumes schema to generate prompt constraints)
```

## File Changes

```
k_search/modular/world/
├── annotations.py   # add _annotation_schema() to mixin
└── node.py          # unchanged (inherits new method from mixin)
```

## Tests

```python
from typing_extensions import TypedDict
from k_search.modular.world.node import Node

class _TestAnn(TypedDict):
    score: int
    note: str

class _TypedNode(Node[_TestAnn]):
    pass


def test_annotation_schema_typed():
    schema = _TypedNode._annotation_schema()
    assert schema == {"score": int, "note": str}


def test_annotation_schema_untyped():
    schema = Node._annotation_schema()
    assert schema == {}
```

## Deferred

**Schema to tool description:** How `_annotation_schema()` flows into actual tool definitions depends on tool abstractions not yet designed. Possible approaches:
- `Tree.get_tools()` that introspects root node
- Standalone formatter functions
- Tool registry that queries node types

This is a separate design once tool abstractions are clearer.

## Future Refinement

Could factor out shared `__orig_bases__` logic into `_get_annotation_type()` if duplication bothers:

```python
@classmethod
def _get_annotation_type(cls) -> type | None:
    for base in getattr(cls, "__orig_bases__", []):
        args = get_args(base)
        if args and args[0] is not AnnDict:
            return args[0]
    return None
```

Then both `__init_subclass__` (adapter setup) and `_annotation_schema` use it. Not essential for initial implementation.

## References

- `2026-03-05-typed-annotations-design.md` - base typed annotations system
- `2026-03-05-tree-data-model-design.md` - Tree/Node data model
