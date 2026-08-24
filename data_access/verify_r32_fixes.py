#!/usr/bin/env python3
"""Verification script for R32 P0-107, P0-048, and service import fixes."""

import os
import sys

def verify_identity_ordering():
    """P0-107: Identity ordering regression fix."""
    from r30._shared import stable_digest

    # Lists and tuples preserve order
    list_abc = stable_digest(["a", "b", "c"])
    list_cba = stable_digest(["c", "b", "a"])
    assert list_abc != list_cba, "Lists should preserve order!"

    tuple_abc = stable_digest(("a", "b", "c"))
    tuple_cba = stable_digest(("c", "b", "a"))
    assert tuple_abc != tuple_cba, "Tuples should preserve order!"

    # Sets are sorted (no meaningful order)
    set_abc = stable_digest({"a", "b", "c"})
    set_cba = stable_digest({"c", "b", "a"})
    assert set_abc == set_cba, "Sets should be order-independent!"

    print("✓ P0-107: Identity ordering fix verified")
    return True

def verify_credential_family_atomicity():
    """P0-048: Credential security vulnerability fix."""
    from factor_engine.security.credentials import EnvCredentialProvider, ValidationError

    # Save original env
    orig_env = os.environ.copy()

    try:
        # Test 1: Complete COS family works
        os.environ.clear()
        os.environ["COS_SECRET_ID"] = "test-cos-id"
        os.environ["COS_SECRET_KEY"] = "test-cos-key"
        cred = EnvCredentialProvider().resolve()
        assert cred.access_key_id == "test-cos-id"
        assert cred.secret_access_key == "test-cos-key"

        # Test 2: Mixed families are rejected
        os.environ.clear()
        os.environ["COS_SECRET_ID"] = "test-cos-id"  # COS family (partial)
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-aws-secret"  # AWS family (partial)

        try:
            EnvCredentialProvider().resolve()
            assert False, "Should have raised ValidationError for mixed families"
        except ValidationError as e:
            assert "family" in str(e).lower(), "Error should mention 'family'"

        print("✓ P0-048: Credential family atomicity fix verified")
        return True
    finally:
        # Restore original env
        os.environ.clear()
        os.environ.update(orig_env)

def verify_service_import():
    """Service import fix: duplicate asynccontextmanager."""
    try:
        from factor_engine.service.app import create_app
        print("✓ Service import fix verified")
        return True
    except Exception as e:
        print(f"✗ Service import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("R32 Critical Blockers Verification")
    print("=" * 60)

    results = []

    try:
        results.append(("P0-107 Identity Ordering", verify_identity_ordering()))
    except Exception as e:
        print(f"✗ P0-107 failed: {e}")
        results.append(("P0-107 Identity Ordering", False))

    try:
        results.append(("P0-048 Credential Security", verify_credential_family_atomicity()))
    except Exception as e:
        print(f"✗ P0-048 failed: {e}")
        results.append(("P0-048 Credential Security", False))

    try:
        results.append(("Service Import", verify_service_import()))
    except Exception as e:
        print(f"✗ Service import failed: {e}")
        results.append(("Service Import", False))

    print("=" * 60)
    print(f"Results: {sum(r[1] for r in results)}/{len(results)} fixes verified")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    return all(r[1] for r in results)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
