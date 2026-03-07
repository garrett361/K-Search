# Typed Annotations Design

## Problem

`Node` and `Action` have `annotations: dict[str, Any]` fields for flexible metadata. Some contexts (v1 compatibility) require specific keys and types (e.g., `overall_rating_0_10: int`). We need:

1. **Static checking** - type checker catches wrong keys/types at dev time
2. **Runtime validation** - bad data fails fast at construction
3. **Loose default** - plain dicts work when unspecialized

## Solution

Combine `Generic[T]` with Pydantic's `TypeAdapter`:

- `Generic[T]` provides static typing (T flows to annotations field)
- `__orig_bases__` captures T at subclass definition (bypasses type erasure)
- `TypeAdapter(T)` validates dicts against TypedDict schemas at runtime

Dict in, dict out. TypedDict is just a typed dict at runtime.

## File Structure

```
k_search/modular/world/
├── annotations.py        # base types, mixin, docs
├── node.py               # inherits _AnnotationMixin
├── action.py             # inherits _AnnotationMixin

tests/modular/world/
├── test_annotations.py   # runtime validation
└── test_annotations.yml  # static type tests
```

Specialized schemas and subclasses (e.g., `V1Node`) are out of scope for this plan.

## Implementation

### `annotations.py`

```python
"""Typed annotation system with runtime validation.

Problem
-------
Node and Action have `annotations: dict` fields for flexible metadata.
Some contexts (v1 compatibility) require specific keys/types. We want:
  1. Static type checking - IDE/mypy catches wrong keys
  2. Runtime validation - bad data fails fast at construction
  3. Loose default - plain dicts still work when unspecialized

Solution
--------
Combine Generic[T] with Pydantic's TypeAdapter:
  - Generic[T] gives static typing (T flows through to annotations field)
  - __orig_bases__ captures T at subclass definition (bypasses erasure)
  - TypeAdapter(T) validates dicts against TypedDict schemas at runtime

Usage
-----
Define a schema as TypedDict:

    class V1Ann(TypedDict):
        overall_rating_0_10: int
        reasoning: str

Subclass with the type parameter:

    class V1Node(Node[V1Ann]):
        pass

V1Node validates annotations at construction:

    V1Node(annotations={"overall_rating_0_10": 8, "reasoning": "good"})  # ok
    V1Node(annotations={"wrong": 1})  # ValidationError

Base Node still accepts any dict:

    Node(annotations={"anything": "goes"})  # no validation
"""

from __future__ import annotations

from typing import ClassVar, Generic, get_args

from pydantic import TypeAdapter
from typing_extensions import TypeVar

AnnDict = dict[str, str | int | float | bool]
T = TypeVar("T", bound=AnnDict, default=AnnDict)


class _AnnotationMixin(Generic[T]):
    """Mixin providing annotation validation via __orig_bases__."""

    annotations: T | None
    _adapter: ClassVar[TypeAdapter | None] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", []):
            origin = getattr(base, "__origin__", None)
            if origin is not None and hasattr(origin, "_adapter"):
                args = get_args(base)
                if args and args[0] is not AnnDict:
                    cls._adapter = TypeAdapter(args[0])
                break

    def _validate(self):
        if self.annotations is not None and self._adapter is not None:
            self._adapter.validate_python(self.annotations)
```

### `node.py` changes

```python
from k_search.modular.world.annotations import _AnnotationMixin, T

@dataclass
class Node(_AnnotationMixin[T]):
    _id: str = ""
    annotations: T | None = None
    # ... existing fields ...

    def __post_init__(self):
        self._validate()
```

### `action.py` changes

```python
from k_search.modular.world.annotations import _AnnotationMixin, T

@dataclass
class Action(_AnnotationMixin[T]):
    title: str
    annotations: T | None = None

    def __post_init__(self):
        self._validate()
```

## Tests

### Runtime validation tests

```python
from pydantic import ValidationError
from typing_extensions import TypedDict
import pytest

from k_search.modular.world.node import Node
from k_search.modular.world.action import Action


class _TestAnn(TypedDict):
    score: int
    note: str


class _StrictNode(Node[_TestAnn]):
    pass


class _StrictAction(Action[_TestAnn]):
    pass


def test_loose_node_accepts_any():
    node = Node(annotations={"x": 1, "y": "z"})
    assert node.annotations["x"] == 1


def test_loose_action_accepts_any():
    action = Action(title="test", annotations={"foo": "bar"})
    assert action.annotations["foo"] == "bar"


def test_strict_node_valid():
    node = _StrictNode(annotations={"score": 8, "note": "ok"})
    assert node.annotations["score"] == 8


def test_strict_node_missing_key():
    with pytest.raises(ValidationError):
        _StrictNode(annotations={"score": 8})


def test_strict_node_wrong_type():
    with pytest.raises(ValidationError):
        _StrictNode(annotations={"score": "bad", "note": "ok"})


def test_strict_action_valid():
    action = _StrictAction(title="test", annotations={"score": 5, "note": "ok"})
    assert action.annotations["note"] == "ok"


def test_none_annotations_ok():
    node = _StrictNode(annotations=None)
    assert node.annotations is None
```

### Static type tests

Uses `pytest-mypy-plugins` to verify type checker catches errors.

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout", "ty", "ruff", "pytest-mypy-plugins"]
```

`tests/modular/world/test_annotations.yml`:
```yaml
- case: strict_node_infers_annotation_type
  main: |
    from typing import assert_type
    from typing_extensions import TypedDict
    from k_search.modular.world.node import Node

    class Ann(TypedDict):
        score: int

    class StrictNode(Node[Ann]):
        pass

    node = StrictNode(annotations={"score": 1})
    assert_type(node.annotations, Ann | None)

- case: strict_node_rejects_wrong_key
  main: |
    from typing_extensions import TypedDict
    from k_search.modular.world.node import Node

    class Ann(TypedDict):
        score: int

    class StrictNode(Node[Ann]):
        pass

    StrictNode(annotations={"wrong": 1})  # E: TypedDict "Ann" has no key "wrong"

- case: loose_node_accepts_any
  main: |
    from k_search.modular.world.node import Node

    Node(annotations={"anything": 123})  # no error
```

## Dependencies

Add to `pyproject.toml`:

```toml
requires-python = ">=3.12"
dependencies = [
    "typing_extensions>=4.0",
    "pydantic>=2.0",
    # ... existing ...
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout", "ty", "ruff", "pytest-mypy-plugins"]
```