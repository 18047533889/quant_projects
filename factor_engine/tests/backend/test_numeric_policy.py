import numpy as np
import pytest

from factor_engine.backend.numeric_policy import (
    NumericPolicy,
    QuantizationCertificate,
    ToleranceProfile,
    quantization_metrics,
)
from factor_engine.storage.materialize.materializer import storage_precision_policy_for


def test_numeric_policy_separates_compute_accumulation_and_output():
    policy = NumericPolicy.default(output_dtype="float32")
    assert policy.compute_dtype == "float64"
    assert policy.accumulation_dtype == "float64"
    assert policy.output_dtype == "float32"
    assert policy.identity_hash() == NumericPolicy.default(output_dtype="float32").identity_hash()
    assert policy.identity_hash() != NumericPolicy.default(output_dtype="float64").identity_hash()


def test_quantization_certificate_covers_numeric_and_rank_metrics():
    values = np.linspace(-1.0, 1.0, 1001, dtype=np.float64)
    cert = QuantizationCertificate.certify("alpha", values)
    metrics = cert.metrics
    assert metrics.max_abs_error >= 0.0
    assert metrics.max_relative_error >= 0.0
    assert metrics.rank_correlation == pytest.approx(1.0)
    assert metrics.top_decile_overlap == pytest.approx(1.0)
    assert metrics.sign_flip_rate == 0.0
    assert cert.validate_for(factor_id="alpha")
    assert not cert.validate_for(factor_id="other")


def test_quantization_certificate_can_fail_rank_and_ic_gate():
    values = np.arange(20, dtype=np.float64)
    tolerance = ToleranceProfile(min_rank_correlation=1.0, max_ic_delta=0.0)
    cert = QuantizationCertificate.certify(
        "alpha", values, forward_returns=values[::-1], tolerance_profile=tolerance
    )
    assert cert.metrics.rank_correlation == pytest.approx(1.0)
    assert cert.production_eligible
    tampered = cert.to_dict()
    tampered["production_eligible"] = False
    assert not QuantizationCertificate.from_dict(tampered).validate_for(factor_id="alpha")


def test_production_float32_publication_is_fail_closed():
    with pytest.raises(ValueError, match="QuantizationCertificate"):
        storage_precision_policy_for("float32", production=True, factor_id="alpha")

    cert = QuantizationCertificate.certify("alpha", np.linspace(-1, 1, 1001))
    dtype, policy = storage_precision_policy_for(
        "float32", production=True, factor_id="alpha",
        lineage_extra={"quantization_certificate": cert.to_dict()},
    )
    assert dtype == "float32"
    assert policy == "float32_certified"


def test_research_float32_keeps_legacy_compatibility():
    assert storage_precision_policy_for("float32", production=False, factor_id="alpha")[1] == "float32_legacy"
