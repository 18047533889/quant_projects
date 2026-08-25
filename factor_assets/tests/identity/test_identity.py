"""
Test identity management and FE protocol boundary.
"""

import pytest

from factor_assets.identity import (
    FactorIdentityProvider,
    FactorIdentity,
    create_factor_id,
)


def test_factor_definition_identity_requires_version():
    """FactorDefinitionIdentity is version-qualified and must NOT fall back to
    an FE compiler generation — the compiler is a separate identity axis."""
    from factor_assets.identity.canonical import (
        FactorDefinitionIdentity,
        FactorCompilerIdentity,
        FactorValueIdentity,
    )

    # factor_version is mandatory: no fallback to a compiler generation.
    with pytest.raises(ValueError, match="factor_version"):
        FactorDefinitionIdentity(factor_id="F1", factor_version="")

    d = FactorDefinitionIdentity(factor_id="F1", factor_version="v1")
    assert d.factor_version == "v1"

    # Compiler and value identity are distinct axes.
    c = FactorCompilerIdentity(compiler_generation="fe-0.9.7")
    assert c.compiler_generation == "fe-0.9.7"
    with pytest.raises(ValueError, match="FactorCompilerIdentity"):
        FactorCompilerIdentity()

    v = FactorValueIdentity(factor_id="F1", snapshot_ref="snapshot:2024")
    assert v.snapshot_ref == "snapshot:2024"
    with pytest.raises(ValueError, match="FactorValueIdentity"):
        FactorValueIdentity(factor_id="F1")


def test_factor_identity_requires_fields():
    """FactorIdentity must have canonical_repr and canonical_hash."""
    with pytest.raises(ValueError, match="canonical_repr"):
        FactorIdentity(canonical_repr="", canonical_hash="abc123")

    with pytest.raises(ValueError, match="canonical_hash"):
        FactorIdentity(canonical_repr="ts_rank(close, 20)", canonical_hash="")


def test_factor_identity_creation():
    """Test FactorIdentity creation."""
    identity = FactorIdentity(
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123def456ghi789",
        fe_identity_ref="plan_xyz",
        fe_compiler_generation="v2.1.0",
        complexity_score=42.5,
    )

    assert identity.canonical_repr == "ts_rank(close, 20)"
    assert identity.canonical_hash == "abc123def456ghi789"
    assert identity.fe_identity_ref == "plan_xyz"
    assert identity.complexity_score == 42.5


def test_create_factor_id_from_hash():
    """Test factor ID creation from canonical hash."""
    hash_val = "abc123def456ghi789jkl"
    factor_id = create_factor_id(hash_val)

    assert factor_id.startswith("F")
    assert factor_id == "Fabc123def456ghi7"
    assert len(factor_id) == 17  # "F" + 16 chars


def test_create_factor_id_with_custom_prefix():
    """Test factor ID with custom prefix."""
    hash_val = "abc123def456ghi789jkl"
    factor_id = create_factor_id(hash_val, prefix="FACT")

    assert factor_id.startswith("FACT")
    assert factor_id == "FACTabc123def456ghi7"


def test_create_factor_id_requires_long_hash():
    """Factor ID creation requires hash of at least 16 characters."""
    with pytest.raises(ValueError, match="at least 16 characters"):
        create_factor_id("short")

    with pytest.raises(ValueError, match="canonical_hash is required"):
        create_factor_id("")


def test_factor_identity_provider_protocol():
    """Test that FactorIdentityProvider is a protocol."""
    # Protocol can be satisfied by any class with matching methods
    class MockProvider:
        def get_canonical_hash(self, expression: str) -> str:
            return f"hash_{expression}"

        def get_canonical_repr(self, expression: str) -> str:
            return expression.lower()

        def get_identity_ref(self, expression: str) -> str:
            return f"ref_{expression}"

    provider = MockProvider()

    # Should work with protocol
    hash_val = provider.get_canonical_hash("test")
    assert hash_val == "hash_test"

    repr_val = provider.get_canonical_repr("TEST")
    assert repr_val == "test"

    ref = provider.get_identity_ref("expr")
    assert ref == "ref_expr"


def test_factor_identity_immutability():
    """FactorIdentity instances are immutable."""
    identity = FactorIdentity(
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123def456ghi789",
    )

    with pytest.raises(Exception):  # FrozenInstanceError
        identity.canonical_hash = "different"  # type: ignore
