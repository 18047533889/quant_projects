# -*- coding: utf-8 -*-
"""DA-ID-P0-002: Hash identity migration tests.

Verifies that correctness-critical identity functions use full 256-bit SHA-256
instead of 64-bit truncation.
"""
import hashlib

import pytest


def test_data_knowledge_identity_digest_is_256_bit():
    """DA-ID-P0-002: FactorId digest must use full 256-bit SHA-256."""
    from factor_engine.semantic.data_knowledge_identity import DataKnowledgeIdentity

    identity = DataKnowledgeIdentity(
        dataset_id="test_dataset",
        snapshot_id="snap_123",
        market="A",
        calendar_id="trade_cal",
    )
    digest = identity.digest()

    # Full SHA-256 hex digest is 64 characters (256 bits / 4 bits per hex char)
    assert len(digest) == 64, f"Expected 64-char (256-bit) digest, got {len(digest)}"

    # Verify it's valid hex
    int(digest, 16)

    # Verify it matches manual SHA-256
    expected = hashlib.sha256(identity.to_key().encode("utf-8")).hexdigest()
    assert digest == expected


def test_compute_payload_hash_is_256_bit():
    """DA-ID-P0-002: Evidence payload hash must use full 256-bit SHA-256."""
    from factor_engine.backend.evidence_provenance import compute_payload_hash

    payload = {"operator": "ts_mean", "window": 5, "backend": "polars"}
    hash_value = compute_payload_hash(payload)

    # Full SHA-256 hex digest is 64 characters
    assert len(hash_value) == 64, f"Expected 64-char (256-bit) hash, got {len(hash_value)}"

    # Verify it's valid hex
    int(hash_value, 16)


def test_compute_implementation_hash_is_256_bit():
    """DA-ID-P0-002: Implementation hash must use full 256-bit SHA-256."""
    from factor_engine.backend.evidence_provenance import compute_implementation_hash

    source_code = """
def calculate(x, window):
    return x.rolling(window).mean()
"""
    hash_value = compute_implementation_hash(source_code)

    # Full SHA-256 hex digest is 64 characters
    assert len(hash_value) == 64, f"Expected 64-char (256-bit) hash, got {len(hash_value)}"

    # Verify it's valid hex
    int(hash_value, 16)


def test_different_identities_produce_different_digests():
    """Verify that distinct DataKnowledgeIdentity instances produce distinct digests."""
    from factor_engine.semantic.data_knowledge_identity import DataKnowledgeIdentity

    id1 = DataKnowledgeIdentity(dataset_id="ds1", market="A")
    id2 = DataKnowledgeIdentity(dataset_id="ds2", market="A")
    id3 = DataKnowledgeIdentity(dataset_id="ds1", market="US")

    digests = {id1.digest(), id2.digest(), id3.digest()}
    assert len(digests) == 3, "All three identities should have unique digests"


def test_collision_probability_documentation():
    """Document birthday paradox collision probability for 64-bit vs 256-bit.

    64-bit hash:
    - 2^64 = 18,446,744,073,709,551,616 possible values
    - Birthday paradox: 50% collision at sqrt(2^64) ≈ 2^32 ≈ 4.3 billion factors
    - Production system could realistically generate millions of factor variants
    - UNACCEPTABLE for FactorId namespace

    256-bit hash:
    - 2^256 ≈ 1.16 × 10^77 possible values
    - Birthday paradox: 50% collision at sqrt(2^256) ≈ 2^128 ≈ 3.4 × 10^38
    - Negligible collision probability for any realistic workload
    - ACCEPTABLE for correctness-critical identity
    """
    # This test exists for documentation only
    assert 2**32 < 5_000_000_000  # 64-bit collision threshold < 5B
    assert 2**128 > 10**38  # 256-bit collision threshold is astronomical


def test_legacy_16_char_digest_rejected():
    """Verify that 16-character (64-bit) digests are now rejected/migrated."""
    from factor_engine.semantic.data_knowledge_identity import DataKnowledgeIdentity

    identity = DataKnowledgeIdentity(dataset_id="test", market="A")
    digest = identity.digest()

    # New digests must be 64 chars, not 16
    assert len(digest) != 16, "64-bit truncated digests are no longer allowed"
    assert len(digest) == 64, "Must use full 256-bit digest"


def test_evidence_hash_binding_uniqueness():
    """Verify that distinct evidence payloads produce distinct hashes."""
    from factor_engine.backend.evidence_provenance import compute_payload_hash

    payload1 = {"op": "ts_mean", "window": 5}
    payload2 = {"op": "ts_mean", "window": 10}
    payload3 = {"op": "ts_std", "window": 5}

    hash1 = compute_payload_hash(payload1)
    hash2 = compute_payload_hash(payload2)
    hash3 = compute_payload_hash(payload3)

    # All should be unique
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3

    # All should be 256-bit
    assert len(hash1) == len(hash2) == len(hash3) == 64


def test_implementation_source_changes_detected():
    """Verify implementation hash changes when source code changes."""
    from factor_engine.backend.evidence_provenance import compute_implementation_hash

    source_v1 = "def calc(x): return x.mean()"
    source_v2 = "def calc(x): return x.sum()"

    hash_v1 = compute_implementation_hash(source_v1)
    hash_v2 = compute_implementation_hash(source_v2)

    assert hash_v1 != hash_v2, "Different implementations must have different hashes"
    assert len(hash_v1) == 64
    assert len(hash_v2) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
