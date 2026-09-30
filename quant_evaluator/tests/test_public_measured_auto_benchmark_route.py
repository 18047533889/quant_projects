import pytest

from quant_evaluator.scripts.benchmark_public_measured_auto import measured_route_matches


def receipt():
    return {
        "auto_backend_reason": "measured_auto_exact_candidate",
        "auto_backend_policy": "measured_auto_registry_exact_v1",
        "auto_backend_profile": "exact_content_process_local",
        "backend_used": "cuda",
        "metric_backends": {"pearson_ic": "cuda", "pearson_ic_ir": "cuda"},
    }


def test_exact_measured_route_receipt():
    assert measured_route_matches(receipt(), ("pearson_ic", "pearson_ic_ir"))


@pytest.mark.parametrize("changes", [
    {"auto_backend_reason": "certified_batch_real_cos_f32_pearson_chain"},
    {"auto_backend_policy": "static_policy"},
    {"auto_backend_profile": "static_profile"},
    {"backend_used": "cpu"},
    {"metric_backends": {"pearson_ic": "cuda", "pearson_ic_ir": "cpu"}},
    {"metric_backends": {"pearson_ic": "cuda"}},
])
def test_parity_alone_cannot_certify_measured_adoption(changes):
    updated = {**receipt(), **changes}
    assert not measured_route_matches(updated, ("pearson_ic", "pearson_ic_ir"))
