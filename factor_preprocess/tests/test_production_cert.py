"""FP-CERT-IMMUTABLE: production certification requires verified causality.

A caller could previously bypass certification by registering with
``admission="PRODUCTION", causal_safe=True, causal_verified=False``:
``register`` only checked ``causal_safe`` and ``validate_production`` never
looked at ``causal_verified``.  These tests pin the fail-closed behaviour at
both the register boundary and the validation boundary.
"""
import pytest

from factor_preprocess.registry.transforms import (
    TransformCategory,
    TransformRegistry,
)


def _dummy(x):
    return x


def _registry_with(name="dummy", **kwargs):
    registry = TransformRegistry()
    registry.register(name, _dummy, TransformCategory.CROSS_SECTIONAL, **kwargs)
    return registry


class TestRegisterBoundary:
    def test_production_with_unverified_causality_is_rejected(self):
        with pytest.raises(ValueError, match="causal_verified"):
            _registry_with(
                causal_safe=True,
                causal_verified=False,
                admission="PRODUCTION",
            )

    def test_production_with_asserted_causal_safe_only_is_rejected(self):
        # The original bypass: causal_safe=True without certification.
        with pytest.raises(ValueError, match="causal_verified"):
            _registry_with(causal_safe=True, admission="PRODUCTION")

    def test_production_with_verified_causality_is_accepted(self):
        registry = _registry_with(
            causal_safe=True,
            causal_verified=True,
            admission="PRODUCTION",
        )
        metadata = registry.validate_production("dummy")
        assert metadata.admission == "PRODUCTION"
        assert metadata.causal_verified is True

    def test_derived_production_admission_requires_verification(self):
        # Omitting admission derives CAUSAL_CERTIFIED only when verified.
        registry = _registry_with(causal_safe=True, causal_verified=True)
        metadata = registry.get("dummy")
        assert metadata.admission == "CAUSAL_CERTIFIED"
        with pytest.raises(ValueError, match="CAUSAL_CERTIFIED"):
            registry.validate_production("dummy")

    def test_causal_certified_without_verification_is_rejected(self):
        # CAUSAL_CERTIFIED itself must not be claimable without verification.
        with pytest.raises(ValueError, match="causal_safe=True"):
            _registry_with(
                causal_safe=False,
                causal_verified=False,
                admission="CAUSAL_CERTIFIED",
            )


class TestValidateProductionBoundary:
    def test_validation_rejects_unverified_production_metadata(self):
        # If a PRODUCTION-admitted metadata somehow reaches the registry
        # without causal_verified (e.g. a mutated snapshot path or a future
        # registration bug), validate_production must still fail closed.
        registry = _registry_with(
            causal_safe=True,
            causal_verified=True,
            admission="PRODUCTION",
        )
        # Simulate a registry-internal inconsistency.
        registry._transforms["dummy"].causal_verified = False

        with pytest.raises(ValueError, match="not causal_verified"):
            registry.validate_production("dummy")

    def test_validation_rejects_production_without_causal_safe(self):
        registry = _registry_with(
            causal_safe=False,
            causal_verified=True,
            admission="UNVERIFIED",
        )
        with pytest.raises(ValueError, match="not registered|UNVERIFIED"):
            registry.validate_production("dummy")

    def test_unverified_admission_rejected_for_production(self):
        registry = _registry_with()
        metadata = registry.get("dummy")
        assert metadata.admission == "UNVERIFIED"
        assert metadata.causal_verified is False
        with pytest.raises(ValueError, match="UNVERIFIED"):
            registry.validate_production("dummy")


class TestDefaultRegistryCertification:
    def test_all_default_production_transforms_are_causally_verified(self):
        from factor_preprocess.registry.transforms import get_default_registry

        registry = get_default_registry()
        production = [
            meta for meta in registry.all_transforms()
            if meta.admission == "PRODUCTION"
        ]
        assert production, "default registry should admit production transforms"
        for meta in production:
            assert meta.causal_verified is True, (
                f"{meta.name} is PRODUCTION but not causal_verified"
            )

    def test_default_production_transforms_pass_validation(self):
        from factor_preprocess.registry.transforms import get_default_registry

        registry = get_default_registry()
        for meta in registry.all_transforms():
            if meta.admission == "PRODUCTION":
                validated = registry.validate_production(meta.name)
                assert validated.causal_verified is True
