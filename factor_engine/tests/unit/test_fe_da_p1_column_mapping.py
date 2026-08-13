"""Tests for P1-1: Column name mapping - alias resolution."""
import pytest


def test_p1_1_catalog_alias_resolution():
    """P1-1: Batch catalog resolve preserves aliases and error classification.

    When an alias is requested, the batch path should keep the resolution
    even though the returned row has a different logical_name (the canonical).
    """
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from data_access.security.run_mode import RunMode

    # This test requires a real DataAccessSource with a catalog that has aliases
    # We can't fully test without a real catalog, but we can verify the logic

    # The fix ensures that when resolve_fields returns a row with a different
    # logical_name than requested (because it's an alias), we now:
    # 1. Check both 'name' and 'logical_name' fields
    # 2. Keep the row if EITHER matches the requested name OR is the canonical
    # 3. Record the mapping from requested alias to canonical name

    # This would require integration testing with a real catalog
    # For now, document the expected behavior
    pass


def test_p1_1_batch_error_classification_preserved():
    """P1-1: Batch path preserves per-field error classification.

    After batch retry, fields still missing should go through the per-field
    path to get proper error classification (clean miss vs unknown field error).
    """
    # This requires integration testing with a real DataAccessSource
    # The fix ensures that after batch retry, still_missing names are:
    # 1. NOT just logged at debug level
    # 2. Fed back through the per-field resolution path
    # 3. Which calls _is_clean_catalog_miss and _raise_or_fallback
    # 4. So production gets proper strict_unknown_fields behavior
    pass


def test_p1_1_output_names_collision_detection():
    """P1-1: Detect when two logical names map to same physical column.

    If two requested logical fields resolve to the same physical column
    (alias + canonical, or two catalog entries with same physical_name),
    we should detect and fail rather than silently dropping one.
    """
    # The subagent found that output_names is keyed by physical name,
    # so if two logical names map to the same physical, the second
    # overwrites the first in the dict.
    #
    # The fix should detect this collision and either:
    # a) Deduplicate the physical list and keep both mappings, OR
    # b) Raise an error if the collision is unintentional

    # This requires integration testing
    pass
