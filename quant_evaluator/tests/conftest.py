"""
Pytest configuration for quant_evaluator tests.

Provides fixtures for test isolation, particularly for global registry cleanup.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_backend_registry():
    """
    Reset global backend registry before each test.

    Ensures test isolation by clearing any registrations made during
    previous tests, preventing state leakage between test cases.
    """
    from quant_evaluator.backends import registry

    # Reset to fresh BackendRegistry instance
    registry._global_registry = registry.BackendRegistry()

    yield

    # Clean up after test
    registry._global_registry = registry.BackendRegistry()
