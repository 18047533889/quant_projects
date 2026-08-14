"""
Pytest configuration for factor_preprocess tests.

Provides fixtures for test isolation, particularly for global registry cleanup.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_transform_registry():
    """
    Reset global transform registry before each test.

    Ensures test isolation by resetting the default registry to a fresh
    state before each test, preventing state leakage between test cases.
    """
    from factor_preprocess.registry import transforms

    # Reset to None to force recreation on next access
    transforms._default_registry = None

    yield

    # Clean up after test
    transforms._default_registry = None
