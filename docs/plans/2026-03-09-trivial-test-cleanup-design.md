# Trivial Test Cleanup Design

## Goal

Remove individual test functions that only verify attribute existence, isinstance, or "is not None" without testing actual behavior. This aligns with the project's stated principle: "don't write trivial tests, make sure they test something important and not simple like attribute existence."

## Approach

Surgical deletion of specific trivial test functions while keeping test files intact. Files that become empty after deletion will be removed entirely.

## Tests to Delete

### `tests/kernel_generators/test_generator.py`
- Lines 35-44: `hasattr` assertions within `test_reasoning_api_generates_code`
- Reason: Pure attribute existence checks (`hasattr(response, "usage")`, `hasattr(response.usage, "output_tokens_details")`)

### `tests/modular/test_gpu_mode_task_definition.py`
- `test_has_required_components` (lines 28-38)
- Reason: Only "is not None" checks for `input_generator`, `correctness_checker`, `scorer`, `feedback_provider`, `reference_impl`

### `tests/modular/test_metrics.py`
- All 6 tests (lines 43-90): `test_returns_noop_by_default`, `test_returns_noop_when_wandb_false`, `test_returns_wandb_tracker_when_wandb_true`, `test_returns_local_tracker_when_local_and_output_dir`, `test_returns_both_trackers_when_wandb_and_local`, `test_noop_when_local_false_and_wandb_false`
- Reason: All are `isinstance` checks without behavior testing
- Action: Delete entire file (all tests trivial)

### `tests/modular/test_artifacts.py`
- `test_string_to_path_coercion` (lines 57-61)
- `test_store_does_not_raise` (lines 63-67)
- Reason: isinstance check + "doesn't raise" without behavior assertion

### `tests/modular/test_gpu_mode_wrappers.py`
- `test_backwards_compat_is_passed` (lines 45-54)
- `test_backwards_compat_to_dict` (lines 56-64)
- `test_backwards_compat_score` (lines 66-72)
- Reason: Trivial property pass-through tests

### `tests/tasks/gpu_mode/test_causal_conv1d.py`
- `test_spec_text_loads` (lines 35-44)
- Reason: Only checks string type and length (`len > 500`)

### `tests/modular/world/test_parse_result.py`
- `test_parse_result_ok` (lines 6-10)
- `test_parse_result_fail` (lines 13-17)
- Reason: Basic container value tests
- Action: Delete entire file (all tests trivial)

### `tests/modular/test_config.py`
- Assertions on lines 24-27 within `test_get_system_info_returns_expected_keys`
- Reason: Key existence + "is not None" checks

## Files to Delete Entirely

- `tests/modular/test_metrics.py` - all tests are trivial
- `tests/modular/world/test_parse_result.py` - all tests are trivial

## What We Keep

- All E2E tests (`test_e2e_search.py`, `test_e2e_causal_conv1d.py`)
- Tests with actual behavior assertions (mocking, state changes, error conditions)
- GPU tests that verify kernel correctness
- Tests that assert outputs, state changes, raised exceptions, or side effects

## Out of Scope

- Directory restructuring (planned for future PR)
- Writing replacement tests
- Orphaned directory cleanup (`tests/search_v2/`, `tests/task_framework/`)

## Verification

After cleanup:
1. Run `pytest tests/ -q --tb=short` to ensure remaining tests pass
2. Run `ruff check tests/` to verify no lint issues
3. Confirm no import errors from deleted files
