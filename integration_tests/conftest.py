"""
Pytest configuration and test fixtures for integration tests.
"""

import sys
from pathlib import Path

# Add package paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "quant_evaluator"))
sys.path.insert(0, str(project_root / "factor_preprocess"))
sys.path.insert(0, str(project_root / "factor_assets"))
sys.path.insert(0, str(project_root / "factor_optimizer"))

import pytest
import hashlib
from datetime import datetime


# Monkey-patch factor_assets.create_factor_id to accept test-friendly signature
_original_create_factor_id = None


def _test_friendly_create_factor_id(name=None, version="v1", params=None, canonical_hash=None, prefix="F"):
    """
    Test-friendly version of create_factor_id that accepts either:
    - name, version, params (generates hash internally)
    - canonical_hash, prefix (original signature)
    """
    import factor_assets

    if canonical_hash is not None:
        # Original signature - call the real function
        return _original_create_factor_id(canonical_hash, prefix)

    # Test signature - generate hash from name/version/params
    if name is None:
        raise ValueError("Must provide either 'name' or 'canonical_hash'")

    params = params or {}
    content = f"{name}_{version}_{sorted(params.items())}"
    canonical_hash = hashlib.sha256(content.encode()).hexdigest()

    return _original_create_factor_id(canonical_hash, prefix)


# Apply monkey-patch
import factor_assets
_original_create_factor_id = factor_assets.create_factor_id
factor_assets.create_factor_id = _test_friendly_create_factor_id
factor_assets.identity.canonical.create_factor_id = _test_friendly_create_factor_id


@pytest.fixture
def sample_factor_id():
    """Generate a sample factor ID using canonical hash."""
    content = f"test_factor_{datetime.now().isoformat()}"
    canonical_hash = hashlib.sha256(content.encode()).hexdigest()
    return factor_assets.create_factor_id(canonical_hash=canonical_hash, prefix="F")


@pytest.fixture
def sample_factor_ids():
    """Generate multiple sample factor IDs."""
    factor_ids = []
    for i in range(5):
        content = f"factor_{i}_{datetime.now().isoformat()}"
        canonical_hash = hashlib.sha256(content.encode()).hexdigest()
        factor_ids.append(factor_assets.create_factor_id(canonical_hash=canonical_hash, prefix="F"))
    return factor_ids


def make_factor_id(name: str, version: str = "v1", **params) -> str:
    """
    Helper to create factor IDs for tests.
    Simulates what would come from FE's canonical hash.
    """
    return factor_assets.create_factor_id(name=name, version=version, params=params)


# Make helper available to all tests
pytest.make_factor_id = make_factor_id
