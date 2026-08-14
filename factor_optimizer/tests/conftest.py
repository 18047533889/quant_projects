"""
Pytest configuration for factor_optimizer tests.

Provides fixtures for test isolation, particularly for global registry cleanup.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_mutation_registry():
    """
    Reset global mutation registry before each test.

    Ensures test isolation by resetting the global registry to None,
    forcing recreation with default mutations on next access.
    """
    from factor_optimizer.grammar import registry

    # Reset to None to force recreation on next access
    registry._GLOBAL_REGISTRY = None

    yield

    # Clean up after test
    registry._GLOBAL_REGISTRY = None
