# Trivial Test Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove trivial test functions that only check hasattr, isinstance, or "is not None" without testing actual behavior.

**Architecture:** Surgical deletion of specific test functions. Files with all trivial tests are deleted entirely. Tests with meaningful assertions are preserved.

**Tech Stack:** Python, pytest

---

### Task 1: Delete trivial test file - test_parse_result.py

**Files:**
- Delete: `tests/modular/world/test_parse_result.py`

**Step 1: Verify tests are trivial**

Run: `cat tests/modular/world/test_parse_result.py`
Expected: Two tests that only check `result.success`, `result.value`, `result.error` - basic dataclass container tests.

**Step 2: Delete the file**

```bash
rm tests/modular/world/test_parse_result.py
```

**Step 3: Run tests to verify no breakage**

Run: `pytest tests/modular/world/ -q --tb=short`
Expected: PASS (remaining tests in world/ still work)

**Step 4: Commit**

```bash
git add -u tests/modular/world/
git commit -m "test(modular): remove trivial test_parse_result.py

Both tests only verified dataclass container values without testing
actual behavior."
```

---

### Task 2: Delete trivial tests from test_metrics.py

**Files:**
- Modify: `tests/modular/test_metrics.py:42-90`

**Step 1: Verify class is trivial**

The `TestCreateMetricsTrackers` class (lines 42-90) contains 6 tests that all just check `isinstance()`. The tests in `TestWandbMetricsTracker`, `TestBuildRoundMetrics`, and `TestLocalMetricsTracker` test actual behavior - keep those.

**Step 2: Delete TestCreateMetricsTrackers class**

Delete lines 42-90 (the entire `TestCreateMetricsTrackers` class).

```python
# DELETE this entire class:
class TestCreateMetricsTrackers:
    def test_returns_noop_by_default(self):
        ...
    def test_returns_noop_when_wandb_false(self):
        ...
    def test_returns_wandb_tracker_when_wandb_true(self):
        ...
    def test_returns_local_tracker_when_local_and_output_dir(self, tmp_path):
        ...
    def test_returns_both_trackers_when_wandb_and_local(self, tmp_path):
        ...
    def test_noop_when_local_false_and_wandb_false(self):
        ...
```

**Step 3: Clean up unused imports if any**

Check if `NoOpMetricsTracker` import is still needed after deletion.

**Step 4: Run tests**

Run: `pytest tests/modular/test_metrics.py -q --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/modular/test_metrics.py
git commit -m "test(modular): remove trivial TestCreateMetricsTrackers

All 6 tests only checked isinstance() without testing actual behavior.
Keep TestWandbMetricsTracker, TestBuildRoundMetrics, TestLocalMetricsTracker
which test real functionality."
```

---

### Task 3: Delete trivial tests from test_artifacts.py

**Files:**
- Modify: `tests/modular/test_artifacts.py:57-67`

**Step 1: Delete TestArtifactConfig and TestNoOpArtifactStore classes**

Delete `TestArtifactConfig` (lines 57-60) - only checks isinstance(Path).
Delete `TestNoOpArtifactStore` (lines 63-66) - only checks "doesn't raise".

```python
# DELETE these classes:
class TestArtifactConfig:
    def test_string_to_path_coercion(self):
        config = ArtifactConfig(output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)


class TestNoOpArtifactStore:
    def test_store_does_not_raise(self):
        store = NoOpArtifactStore()
        store.store(make_round_mock(), round_idx=0)
```

**Step 2: Run tests**

Run: `pytest tests/modular/test_artifacts.py -q --tb=short`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/modular/test_artifacts.py
git commit -m "test(modular): remove trivial artifact tests

- TestArtifactConfig: only checked isinstance(Path)
- TestNoOpArtifactStore: only checked 'doesn't raise'"
```

---

### Task 4: Delete trivial test from test_gpu_mode_task_definition.py

**Files:**
- Modify: `tests/modular/test_gpu_mode_task_definition.py:28-38`

**Step 1: Delete test_has_required_components**

Delete the `test_has_required_components` method (lines 28-38) which only asserts `is not None`.

```python
# DELETE this method:
    def test_has_required_components(self):
        from k_search.modular.adapters.gpu_mode import GpuModeTriMulTaskDefinition

        task = GpuModeTriMulTask(task_dir=CAUSAL_CONV1D_DIR)
        task_def = GpuModeTriMulTaskDefinition(task)

        assert task_def.input_generator is not None
        assert task_def.correctness_checker is not None
        assert task_def.scorer is not None
        assert task_def.feedback_provider is not None
        assert task_def.reference_impl is not None
```

**Step 2: Run tests**

Run: `pytest tests/modular/test_gpu_mode_task_definition.py -q --tb=short -m cuda`
Expected: PASS (or skip if no GPU)

**Step 3: Commit**

```bash
git add tests/modular/test_gpu_mode_task_definition.py
git commit -m "test(modular): remove trivial test_has_required_components

Only asserted 'is not None' for 5 attributes without testing behavior."
```

---

### Task 5: Delete trivial tests from test_gpu_mode_wrappers.py

**Files:**
- Modify: `tests/modular/test_gpu_mode_wrappers.py:45-72`

**Step 1: Delete backwards_compat tests**

Delete these three methods that only test trivial property pass-through:
- `test_backwards_compat_is_passed` (lines 45-54)
- `test_backwards_compat_to_dict` (lines 56-64)
- `test_backwards_compat_score` (lines 66-72)

```python
# DELETE these methods:
    def test_backwards_compat_is_passed(self):
        ...
        assert wrapper.is_passed() is True
        assert wrapper.latency_ms == 1.0
        assert wrapper.status == "passed"

    def test_backwards_compat_to_dict(self):
        ...
        d = wrapper.to_dict(include_log_excerpt=True)
        assert d["status"] == "passed"
        assert d["latency_ms"] == 1.0

    def test_backwards_compat_score(self):
        ...
        assert wrapper.score() == 0.5
```

**Step 2: Run tests**

Run: `pytest tests/modular/test_gpu_mode_wrappers.py -q --tb=short`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/modular/test_gpu_mode_wrappers.py
git commit -m "test(modular): remove trivial backwards_compat tests

Three tests only verified property pass-through without testing
meaningful behavior."
```

---

### Task 6: Delete trivial test from test_causal_conv1d.py

**Files:**
- Modify: `tests/tasks/gpu_mode/test_causal_conv1d.py:35-39`

**Step 1: Delete test_spec_text_loads**

Delete the `test_spec_text_loads` method (lines 36-39) which only checks type and length.

```python
# DELETE this method:
    def test_spec_text_loads(self):
        """Verify spec text loads and is non-empty."""
        assert isinstance(CAUSAL_CONV1D_SPEC_TEXT_TRITON, str)
        assert len(CAUSAL_CONV1D_SPEC_TEXT_TRITON) > 500
```

Keep `test_spec_contains_interface`, `test_spec_mentions_silu`, `test_spec_has_test_case`, `test_spec_contains_baseline_submission` which test actual content.

**Step 2: Run tests**

Run: `pytest tests/tasks/gpu_mode/test_causal_conv1d.py::TestSpec -q --tb=short`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/tasks/gpu_mode/test_causal_conv1d.py
git commit -m "test(v1): remove trivial test_spec_text_loads

Only checked isinstance(str) and len > 500. Keep tests that verify
actual spec content."
```

---

### Task 7: Remove hasattr assertions from test_generator.py

**Files:**
- Modify: `tests/kernel_generators/test_generator.py:35-38`

**Step 1: Review the test**

The test `test_reasoning_api_produces_reasoning_tokens` has hasattr assertions on lines 35-38, but the final assertion `assert reasoning_tokens > 0` is meaningful. Remove only the hasattr checks.

**Step 2: Delete hasattr assertions**

Delete lines 35-38:

```python
# DELETE these lines:
        assert hasattr(response, "usage"), "Response should have usage statistics"
        assert hasattr(response.usage, "output_tokens_details"), (
            "Usage should have output_tokens_details"
        )
```

Keep the `reasoning_tokens = getattr(...)` and `assert reasoning_tokens > 0` logic.

**Step 3: Run tests**

Run: `pytest tests/kernel_generators/test_generator.py -q --tb=short`
Expected: PASS (or skip if no RITS credentials)

**Step 4: Commit**

```bash
git add tests/kernel_generators/test_generator.py
git commit -m "test(v1): remove hasattr assertions from test_generator

Keep the meaningful assertion that reasoning_tokens > 0."
```

---

### Task 8: Final verification

**Step 1: Run full test suite**

Run: `pytest tests/ -q --tb=short --ignore=tests/modular/test_e2e_causal_conv1d.py --ignore=tests/modular/test_e2e_search.py -m "not cuda"`
Expected: All tests PASS

**Step 2: Run ruff check**

Run: `ruff check tests/`
Expected: No errors

**Step 3: Run ruff format**

Run: `ruff format tests/`
Expected: Files reformatted if needed

**Step 4: Final commit if formatting changed**

```bash
git add tests/
git commit -m "style: format tests after cleanup" || echo "No formatting changes"
```

---

## Summary

| File | Action | Tests Removed |
|------|--------|---------------|
| `tests/modular/world/test_parse_result.py` | Delete file | 2 |
| `tests/modular/test_metrics.py` | Delete class | 6 |
| `tests/modular/test_artifacts.py` | Delete 2 classes | 2 |
| `tests/modular/test_gpu_mode_task_definition.py` | Delete method | 1 |
| `tests/modular/test_gpu_mode_wrappers.py` | Delete 3 methods | 3 |
| `tests/tasks/gpu_mode/test_causal_conv1d.py` | Delete method | 1 |
| `tests/kernel_generators/test_generator.py` | Delete assertions | 0 (partial) |

**Total: 15 trivial tests removed**
