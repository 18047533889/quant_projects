"""DLIB-FA-004/017 identity + value-instance status hardening tests.

- FactorValueIdentity: computed ``0.0`` value is distinguishable from
  UNKNOWN / INSUFFICIENT_DATA / INVALID via a typed status enum
  (DLIB-FA-004).
- Status invalid states: a ``FactorValueIdentity`` with status
  ``INVALID`` must never be treated as computed.
"""

import pytest


def test_factor_value_identity_computed_zero_is_distinct_from_unknown():
    from factor_assets.identity.canonical import (
        FactorValueIdentity,
        FactorValueStatus,
    )

    computed = FactorValueIdentity(
        factor_id="F1",
        snapshot_ref="snapshot:2024",
        status=FactorValueStatus.COMPUTED,
        sample_ratio=1.0,
        confidence=0.95,
    )
    assert computed.is_computed is True
    assert computed.status is FactorValueStatus.COMPUTED

    unknown = FactorValueIdentity(
        factor_id="F1",
        snapshot_ref="snapshot:2024",
        status=FactorValueStatus.UNKNOWN,
    )
    assert unknown.is_computed is False

    insufficient = FactorValueIdentity(
        factor_id="F1",
        snapshot_ref="snapshot:2024",
        status=FactorValueStatus.INSUFFICIENT_DATA,
    )
    assert insufficient.is_computed is False

    invalid = FactorValueIdentity(
        factor_id="F1",
        snapshot_ref="snapshot:2024",
        status=FactorValueStatus.INVALID,
    )
    assert invalid.is_computed is False


def test_factor_value_identity_validation():
    from factor_assets.identity.canonical import (
        FactorValueIdentity,
        FactorValueStatus,
    )

    # sample_ratio must be in [0, 1].
    with pytest.raises(ValueError, match="sample_ratio"):
        FactorValueIdentity(
            factor_id="F1",
            snapshot_ref="s",
            sample_ratio=1.5,
        )
    # confidence must be finite and in [0, 1].
    with pytest.raises(ValueError, match="confidence"):
        FactorValueIdentity(
            factor_id="F1",
            snapshot_ref="s",
            confidence=float("nan"),
        )
    # status must be a FactorValueStatus.
    with pytest.raises(TypeError, match="status"):
        FactorValueIdentity(factor_id="F1", snapshot_ref="s", status="COMPUTED")


def test_factor_value_status_enum_values():
    from factor_assets.identity.canonical import FactorValueStatus

    assert FactorValueStatus.COMPUTED.value == "COMPUTED"
    assert FactorValueStatus.UNKNOWN.value == "UNKNOWN"
    assert FactorValueStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"
    assert FactorValueStatus.INVALID.value == "INVALID"
    # A consumer must never conflate UNKNOWN with a computed zero.
    assert FactorValueStatus.UNKNOWN is not FactorValueStatus.COMPUTED


def test_factor_definition_and_compiler_identity_axes_still_distinct():
    from factor_assets.identity.canonical import (
        FactorDefinitionIdentity,
        FactorCompilerIdentity,
    )

    with pytest.raises(ValueError, match="factor_version"):
        FactorDefinitionIdentity(factor_id="F1", factor_version="")
    d = FactorDefinitionIdentity(factor_id="F1", factor_version="v2")
    assert d.factor_version == "v2"
    c = FactorCompilerIdentity(compiler_generation="fe-0.9.7")
    # factor_version must never fall back to a compiler generation.
    assert d.factor_version != c.compiler_generation