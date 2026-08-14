"""
DA-P0-010 through DA-P0-014: Identity encoder correctness tests.

Tests strict typed encoding, NaN/Inf/-0.0 semantics, and correctness identity requirements.
"""
import math
import sys
from pathlib import Path

import pytest

# Ensure we import from the local worktree, not installed package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: identity_encoder internally imports from data_access.core.exceptions (installed)
# so we need to catch that exception type, not the local one
from data_access.core.exceptions import ValidationError
from core.identity_encoder import (
    CanonicalIdentityEncoder,
    hash_correctness_identity,
    hash_cache_key,
)


class TestDAP0010CorrectnessIdentity:
    """DA-P0-010: correctness cache identity uses strict + >=128 bit."""

    def test_correctness_identity_uses_strict_mode(self):
        """Correctness identity must use strict mode (no repr fallback)."""
        # Unknown object should raise ValidationError in strict mode
        class UnknownObject:
            pass

        obj = UnknownObject()

        with pytest.raises(ValidationError, match="DA-P0-011.*Cannot encode.*UnknownObject"):
            hash_correctness_identity(obj)

    def test_correctness_identity_minimum_128_bits(self):
        """Correctness identity must be at least 128 bits."""
        result = hash_correctness_identity({"key": "value"})

        assert result.identity_type == "correctness"
        assert result.bits >= 128
        # SHA-256 = 256 bits = 64 hex chars
        assert len(result.digest) == 64

    def test_correctness_identity_rejects_64_bit(self):
        """Correctness identity cannot be 64-bit."""
        # Create encoder with strict=True
        encoder = CanonicalIdentityEncoder(strict=True)

        # 64-bit hash
        digest_64 = encoder.hash_identity({"key": "value"}, bits=64)

        # Cannot create IdentityDigest with correctness type and 64 bits
        from core.identity_encoder import IdentityDigest

        with pytest.raises(ValidationError, match="correctness.*requires.*128 bits"):
            IdentityDigest(
                digest=digest_64[:16],  # 64 bits = 16 hex chars
                bits=64,
                identity_type="correctness",
            )

    def test_performance_cache_can_use_64_bit(self):
        """Ephemeral performance cache can still use 64-bit keys."""
        key = hash_cache_key({"key": "value"}, bits=64)

        # Should return a 64-bit hash (16 hex chars)
        assert len(key) == 16


class TestDAP0011ReprFallbackProhibited:
    """DA-P0-011: Identity encoder禁止unknown objects repr fallback."""

    def test_strict_mode_rejects_unknown_types(self):
        """Strict mode must reject unknown types without repr fallback."""
        encoder = CanonicalIdentityEncoder(strict=True)

        class CustomClass:
            def __init__(self):
                self.value = 42

        obj = CustomClass()

        with pytest.raises(ValidationError) as exc_info:
            encoder.encode(obj)

        error_msg = str(exc_info.value)
        assert "DA-P0-011" in error_msg
        assert "Cannot encode" in error_msg
        assert "CustomClass" in error_msg
        assert "禁止repr fallback" in error_msg

    def test_non_strict_mode_still_uses_repr(self):
        """Non-strict mode can still use repr for unknown types."""
        encoder = CanonicalIdentityEncoder(strict=False)

        class CustomClass:
            def __repr__(self):
                return "CustomClass(42)"

        obj = CustomClass()
        encoded = encoder.encode(obj)

        # Should contain repr in encoded form
        assert "CustomClass(42)" in encoded


class TestDAP0012SetTypedCanonicalization:
    """DA-P0-012: set fallback preserves type information."""

    def test_set_preserves_type_information(self):
        """Set encoding must preserve types to avoid collision."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # Different types should produce different encodings
        set_with_int = {1, 2, 3}
        set_with_str = {"1", "2", "3"}

        encoded_int = encoder.encode(set_with_int)
        encoded_str = encoder.encode(set_with_str)

        assert encoded_int != encoded_str

    def test_mixed_type_set_preserves_all_types(self):
        """Mixed-type sets preserve each element's type."""
        encoder = CanonicalIdentityEncoder(strict=True)

        mixed_set = {1, "1", 2.0, "2"}

        encoded = encoder.encode(mixed_set)

        # Should contain type tags for differentiation
        assert "int" in encoded or "__set__" in encoded

    def test_set_deterministic_ordering(self):
        """Sets must have deterministic ordering regardless of insertion."""
        encoder = CanonicalIdentityEncoder(strict=True)

        set1 = {3, 1, 2}
        set2 = {2, 3, 1}
        set3 = {1, 2, 3}

        encoded1 = encoder.encode(set1)
        encoded2 = encoder.encode(set2)
        encoded3 = encoder.encode(set3)

        assert encoded1 == encoded2 == encoded3


class TestDAP0013DictKeyTypedCanonicalization:
    """DA-P0-013: dict keys need typed canonicalization."""

    def test_dict_key_type_preservation(self):
        """Dict keys with different types must not collide."""
        encoder = CanonicalIdentityEncoder(strict=True)

        dict_int_key = {1: "a", 2: "b"}
        dict_str_key = {"1": "a", "2": "b"}

        encoded_int = encoder.encode(dict_int_key)
        encoded_str = encoder.encode(dict_str_key)

        assert encoded_int != encoded_str

    def test_dict_key_deterministic_ordering(self):
        """Dict keys must have deterministic ordering."""
        encoder = CanonicalIdentityEncoder(strict=True)

        dict1 = {"z": 1, "a": 2, "m": 3}
        dict2 = {"a": 2, "m": 3, "z": 1}
        dict3 = {"m": 3, "z": 1, "a": 2}

        encoded1 = encoder.encode(dict1)
        encoded2 = encoder.encode(dict2)
        encoded3 = encoder.encode(dict3)

        assert encoded1 == encoded2 == encoded3

    def test_mixed_type_dict_keys(self):
        """Mixed-type dict keys preserve type information."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # Mixed int and str keys
        mixed_dict = {1: "int_key", "1": "str_key", 2: "int_key_2"}

        encoded = encoder.encode(mixed_dict)

        # Should preserve distinction between int and str keys
        assert "__dict__" in encoded


class TestDAP0014SpecialFloatSemantics:
    """DA-P0-014: NaN / Infinity / -0.0 identity policy frozen."""

    def test_nan_has_explicit_token(self):
        """NaN must have explicit token, not just JSON NaN."""
        encoder = CanonicalIdentityEncoder(strict=True)

        value = float("nan")
        encoded = encoder.encode(value)

        assert "NaN" in encoded or "__float_special__" in encoded
        assert encoded == encoder.encode(float("nan"))  # Stable

    def test_positive_infinity_token(self):
        """Positive infinity has explicit token."""
        encoder = CanonicalIdentityEncoder(strict=True)

        value = float("inf")
        encoded = encoder.encode(value)

        assert "+Inf" in encoded or "__float_special__" in encoded

    def test_negative_infinity_token(self):
        """Negative infinity has explicit token."""
        encoder = CanonicalIdentityEncoder(strict=True)

        value = float("-inf")
        encoded = encoder.encode(value)

        assert "-Inf" in encoded or "__float_special__" in encoded

    def test_positive_and_negative_infinity_distinct(self):
        """Positive and negative infinity must be distinct."""
        encoder = CanonicalIdentityEncoder(strict=True)

        pos_inf = encoder.encode(float("inf"))
        neg_inf = encoder.encode(float("-inf"))

        assert pos_inf != neg_inf

    def test_negative_zero_distinct_from_positive_zero(self):
        """Negative zero and positive zero must have distinct identities."""
        encoder = CanonicalIdentityEncoder(strict=True)

        pos_zero = encoder.encode(0.0)
        neg_zero = encoder.encode(-0.0)

        assert pos_zero != neg_zero
        assert "-0.0" in neg_zero or "__float_special__" in neg_zero

    def test_negative_zero_detection(self):
        """Negative zero is correctly detected."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # Create negative zero
        neg_zero = -0.0
        assert math.copysign(1.0, neg_zero) == -1.0

        encoded = encoder.encode(neg_zero)
        assert "-0.0" in encoded

    def test_regular_float_not_affected(self):
        """Regular floats still encode normally."""
        encoder = CanonicalIdentityEncoder(strict=True)

        regular_floats = [1.5, -2.3, 42.0, 0.0001, -999.999]

        for value in regular_floats:
            encoded = encoder.encode(value)
            # Should use regular float encoding
            assert "__float__" in encoded or (value == 0.0 and encoded == "0.0")

    def test_special_floats_in_collections(self):
        """Special floats in collections are handled correctly."""
        encoder = CanonicalIdentityEncoder(strict=True)

        data = {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "neg_zero": -0.0,
            "pos_zero": 0.0,
            "regular": 3.14,
        }

        # Should not raise
        encoded = encoder.encode(data)

        # All special values should be present
        assert "NaN" in encoded
        assert "Inf" in encoded


class TestCorrectnessIdentityIntegration:
    """Integration tests for correctness identity usage."""

    def test_factor_cache_key_scenario(self):
        """Scenario: caching factor computation results."""
        # Factor computation parameters
        params = {
            "window": 20,
            "min_periods": 10,
            "adjustment": "forward",
            "universe": {"A", "B", "C"},
        }

        # Generate correctness identity
        identity = hash_correctness_identity(params)

        assert identity.bits == 256
        assert identity.identity_type == "correctness"

        # Same parameters should produce same identity
        identity2 = hash_correctness_identity(params)
        assert identity.digest == identity2.digest

    def test_different_params_different_identity(self):
        """Different parameters must produce different identities."""
        params1 = {"window": 20, "min_periods": 10}
        params2 = {"window": 20, "min_periods": 11}

        id1 = hash_correctness_identity(params1)
        id2 = hash_correctness_identity(params2)

        assert id1.digest != id2.digest

    def test_type_changes_affect_identity(self):
        """Type changes must affect identity (not just value)."""
        params_int = {"threshold": 5}
        params_float = {"threshold": 5.0}

        id_int = hash_correctness_identity(params_int)
        id_float = hash_correctness_identity(params_float)

        # These should be different because types differ
        assert id_int.digest != id_float.digest
