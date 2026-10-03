"""Cross-check the standalone QE scan against the closure's QE identity."""
from types import SimpleNamespace

import pytest

from quant_evaluator.runtime import source_qualified_router as router
from quant_evaluator.runtime.source_identity import ProcessSourceIdentity
from test_source_qualified_router_oct03 import live


_COMPONENTS = ("quant_evaluator", "data_access", "factor_optimizer",
               "factor_preprocess")


def _capture(live):
    source, metadata, policy = live
    return router.capture_source_route_context(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint="9" * 64, requested_tile_size=2,
        maximum_effective_tile_size=2, effective_tile_size=2, policy=policy)


def _receipt(qe_digest):
    digests = tuple((name, qe_digest if name == "quant_evaluator" else char * 64)
                    for name, char in zip(_COMPONENTS, "abcd"))
    return SimpleNamespace(
        schema=router.DEPENDENCY_IDENTITY_SCHEMA, digest="3" * 64,
        component_digests=digests,
    )


def test_public_context_capture_rejects_mixed_qe_source_identity(live, monkeypatch):
    """A standalone A scan plus closure B must never form one executable identity."""
    monkeypatch.setattr(router, "capture_source_dependency_identity",
                        lambda: _receipt("0" * 64))

    with pytest.raises(router.SourceQualificationError,
                       match="current_qe_dependency_identity_mismatch"):
        _capture(live)


def test_public_context_capture_accepts_matching_qe_component_binding(live, monkeypatch):
    monkeypatch.setattr(router, "capture_source_dependency_identity",
                        lambda: _receipt("f" * 64))

    context = _capture(live)

    assert context.executable_source_sha256


def test_public_context_capture_checks_real_strict_qe_tree_against_binding(
        live, monkeypatch, tmp_path):
    source_root = tmp_path / "quant_evaluator"
    runtime_root = source_root / "runtime"
    runtime_root.mkdir(parents=True)
    route_file = runtime_root / "source_qualified_router.py"
    route_file.write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(router, "__file__", str(route_file))
    monkeypatch.setattr(router, "ProcessSourceIdentity", ProcessSourceIdentity)
    monkeypatch.setattr(router, "capture_source_dependency_identity",
                        lambda: _receipt("0" * 64))

    with pytest.raises(router.SourceQualificationError,
                       match="current_qe_dependency_identity_mismatch"):
        _capture(live)


@pytest.mark.parametrize("component_digests", [
    (("data_access", "a" * 64), ("factor_optimizer", "b" * 64),
     ("factor_preprocess", "c" * 64)),
    (("quant_evaluator", "f" * 64), ("quant_evaluator", "f" * 64),
     ("data_access", "a" * 64), ("factor_optimizer", "b" * 64),
     ("factor_preprocess", "c" * 64)),
    (("quant_evaluator", "not-a-digest"), ("data_access", "a" * 64),
     ("factor_optimizer", "b" * 64), ("factor_preprocess", "c" * 64)),
])
def test_public_context_capture_rejects_invalid_component_binding(
        live, monkeypatch, component_digests):
    monkeypatch.setattr(router, "capture_source_dependency_identity", lambda:
                        SimpleNamespace(schema=router.DEPENDENCY_IDENTITY_SCHEMA,
                                        digest="3" * 64,
                                        component_digests=component_digests))

    with pytest.raises(router.SourceQualificationError,
                       match="current_dependency_component_binding_invalid"):
        _capture(live)
