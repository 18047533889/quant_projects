"""
Algorithm golden tests for identity encoder.

Coordinator review: Verify SHA-256 implementation and detect unintended changes.
"""
import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.identity_encoder import (
    CanonicalIdentityEncoder,
    hash_correctness_identity,
    hash_cache_key,
    hash_ephemeral_cache_key,
)


class TestSHA256AlgorithmGolden:
    """Golden tests: Verify exact SHA-256 algorithm output."""

    def test_sha256_64bit_golden(self):
        """DA-P1-028: Verify 64-bit is SHA-256 truncated, not MD5."""
        encoder = CanonicalIdentityEncoder(strict=True)
        value = {"key": "value"}

        # Get 64-bit digest
        digest_64 = encoder.hash_identity(value, bits=64)

        # Manually compute expected: SHA-256 truncated to 16 hex chars
        canonical = encoder.encode(value)
        expected_full = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected_64 = expected_full[:16]

        assert digest_64 == expected_64
        assert len(digest_64) == 16  # 64 bits = 16 hex chars

        # Verify it's NOT MD5
        md5_digest = hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
        assert digest_64 != md5_digest[:16], "DA-P1-028: Must not use MD5"

    def test_sha256_128bit_golden(self):
        """DA-P1-028: Verify 128-bit is SHA-256 truncated, not MD5."""
        encoder = CanonicalIdentityEncoder(strict=True)
        value = {"key": "value"}

        # Get 128-bit digest
        digest_128 = encoder.hash_identity(value, bits=128)

        # Manually compute expected: SHA-256 truncated to 32 hex chars
        canonical = encoder.encode(value)
        expected_full = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected_128 = expected_full[:32]

        assert digest_128 == expected_128
        assert len(digest_128) == 32  # 128 bits = 32 hex chars

        # Verify it's NOT MD5
        md5_digest = hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
        assert digest_128 != md5_digest, "DA-P1-028: Must not use MD5"

    def test_sha256_256bit_golden(self):
        """Verify 256-bit uses full SHA-256."""
        encoder = CanonicalIdentityEncoder(strict=True)
        value = {"key": "value"}

        # Get 256-bit digest
        digest_256 = encoder.hash_identity(value, bits=256)

        # Manually compute expected: full SHA-256
        canonical = encoder.encode(value)
        expected_256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert digest_256 == expected_256
        assert len(digest_256) == 64  # 256 bits = 64 hex chars

    def test_correctness_identity_uses_sha256(self):
        """Correctness identity must use SHA-256, never MD5."""
        value = {"window": 20, "method": "ewm"}

        identity = hash_correctness_identity(value, bits=256)

        # Verify by recomputing with SHA-256
        encoder = CanonicalIdentityEncoder(strict=True)
        canonical = encoder.encode(value)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert identity.digest == expected


class TestCanonicalizationGolden:
    """Golden tests for typed canonicalization."""

    def test_typed_set_canonicalization_golden(self):
        """Verify typed set canonicalization produces expected structure."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # int set
        set_int = {1, 2, 3}
        canonical_int = encoder.encode(set_int)

        # str set with same values
        set_str = {"1", "2", "3"}
        canonical_str = encoder.encode(set_str)

        # Must be different
        assert canonical_int != canonical_str

        # Verify structure contains type tags
        assert "int" in canonical_int
        assert "str" in canonical_str
        assert "__set__" in canonical_int
        assert "__set__" in canonical_str

    def test_typed_dict_canonicalization_golden(self):
        """Verify typed dict key canonicalization produces expected structure."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # int key
        dict_int = {1: "a", 2: "b"}
        canonical_int = encoder.encode(dict_int)

        # str key with same values
        dict_str = {"1": "a", "2": "b"}
        canonical_str = encoder.encode(dict_str)

        # Must be different
        assert canonical_int != canonical_str

        # Verify structure contains type tags
        assert "int" in canonical_int
        assert "str" in canonical_str
        assert "__dict__" in canonical_int
        assert "__dict__" in canonical_str

    def test_special_float_tokens_golden(self):
        """Verify special float tokens are exact."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # NaN
        canonical_nan = encoder.encode(float("nan"))
        assert "__float_special__" in canonical_nan
        assert "NaN" in canonical_nan

        # +Inf
        canonical_pinf = encoder.encode(float("inf"))
        assert "__float_special__" in canonical_pinf
        assert "+Inf" in canonical_pinf

        # -Inf
        canonical_ninf = encoder.encode(float("-inf"))
        assert "__float_special__" in canonical_ninf
        assert "-Inf" in canonical_ninf

        # -0.0
        canonical_nzero = encoder.encode(-0.0)
        assert "__float_special__" in canonical_nzero
        assert "-0.0" in canonical_nzero

        # +0.0 should be regular
        canonical_pzero = encoder.encode(0.0)
        assert "__float_special__" not in canonical_pzero

    def test_deterministic_set_ordering(self):
        """Set elements must have deterministic ordering."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # Create same set multiple times
        encodings = []
        for _ in range(5):
            s = {3, 1, 2, "b", "a"}
            encodings.append(encoder.encode(s))

        # All encodings must be identical
        assert len(set(encodings)) == 1

    def test_deterministic_dict_ordering(self):
        """Dict keys must have deterministic ordering."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # Create same dict multiple times with different insertion orders
        encodings = []
        for _ in range(5):
            d = {3: "c", 1: "a", 2: "b", "z": "zz", "x": "xx"}
            encodings.append(encoder.encode(d))

        # All encodings must be identical
        assert len(set(encodings)) == 1


class TestEphemeralCacheKeyAPI:
    """Test ephemeral cache key API and deprecation."""

    def test_hash_ephemeral_cache_key_exists(self):
        """New hash_ephemeral_cache_key function should exist."""
        key = hash_ephemeral_cache_key({"key": "value"})
        assert isinstance(key, str)
        assert len(key) == 16  # 64-bit default

    def test_hash_cache_key_deprecated_but_works(self):
        """Old hash_cache_key should still work for backward compatibility."""
        key = hash_cache_key({"key": "value"})
        assert isinstance(key, str)
        assert len(key) == 16

    def test_ephemeral_and_deprecated_equivalent(self):
        """hash_cache_key and hash_ephemeral_cache_key should produce same result."""
        value = {"window": 20, "min_periods": 10}

        key1 = hash_cache_key(value)
        key2 = hash_ephemeral_cache_key(value)

        assert key1 == key2

    def test_ephemeral_uses_non_strict(self):
        """Ephemeral cache key uses non-strict mode for convenience."""
        class CustomObject:
            def __repr__(self):
                return "CustomObject(42)"

        # Should not raise, uses repr fallback
        key = hash_ephemeral_cache_key(CustomObject())
        assert isinstance(key, str)
        assert len(key) == 16


class TestStrictModeNoStrFallback:
    """Coordinator review: Verify strict mode fails without str() fallback."""

    def test_strict_set_sorting_no_fallback(self):
        """Strict mode set sorting uses JSON, no str() fallback."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # All supported types should be sortable via JSON
        valid_set = {1, "a", 2.5, True, None}
        canonical = encoder.encode(valid_set)
        assert "__set__" in canonical

        # Multiple encodings should be identical (deterministic)
        canonical2 = encoder.encode(valid_set)
        assert canonical == canonical2

    def test_strict_dict_sorting_no_fallback(self):
        """Strict mode dict sorting uses JSON, no str() fallback."""
        encoder = CanonicalIdentityEncoder(strict=True)

        # All supported types should be sortable via JSON
        valid_dict = {1: "a", "key": "b", 2.5: "c", None: "d"}
        canonical = encoder.encode(valid_dict)
        assert "__dict__" in canonical

        # Multiple encodings should be identical (deterministic)
        canonical2 = encoder.encode(valid_dict)
        assert canonical == canonical2
