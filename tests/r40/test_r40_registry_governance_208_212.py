# -*- coding: utf-8 -*-
"""R40 #208/#209/#211/#212: registry mutation token, explicit backend override
spec, canonical manifest, and production semantic_version gate.

Tests use a fresh ``OperatorRegistry`` subclass so the shared bootstrap registry
is never mutated."""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator
from factor_engine.cleaned_operators.registry import (
    BackendOverrideSpec,
    CanonicalOperatorManifest,
    OperatorRegistry,
    _BOOTSTRAP_TOKEN,
    _merge_param_names,
)


# 模块导入时（任何测试触发 layer-governance seal 之前）捕获未封印的原始
# ``OperatorRegistry.register`` 函数。被 ``_seal_registry()`` 替换成
# ``strict_register`` 后，它的前置重复检查会先于 declared-override manifest
# 逻辑触发——governance 测试必须针对原始 registry 逻辑（R40 #208-212 就在其中），
# 否则在全套件里（service_security 等先触发 bootstrap/seal）会误报 duplicate。
_ORIGINAL_REGISTER_FUNC = OperatorRegistry.register.__func__


class _TestRegistry(OperatorRegistry):
    """Isolated registry for governance tests."""

    _operators = {}
    _aliases = {}
    _catalog = {}
    _lifecycle = OperatorRegistry.Lifecycle.BUILDING
    _version = 0
    _first_registered = {}
    _overwrite_log = []
    _override_chain = {}
    _canonical_manifests = {}
    _frozen = None
    _mutation_token = _BOOTSTRAP_TOKEN
    _DECLARED_OVERRIDE_MANIFEST = {}
    _DECLARED_OVERRIDE_CONTRACT_HASHES = {}


def _reset_registry_state():
    _TestRegistry._operators = {}
    _TestRegistry._aliases = {}
    _TestRegistry._catalog = {}
    _TestRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING
    _TestRegistry._version = 0
    _TestRegistry._first_registered = {}
    _TestRegistry._overwrite_log = []
    _TestRegistry._override_chain = {}
    _TestRegistry._canonical_manifests = {}
    _TestRegistry._frozen = None
    _TestRegistry._mutation_token = _BOOTSTRAP_TOKEN
    _TestRegistry._DECLARED_OVERRIDE_MANIFEST = {}
    _TestRegistry._DECLARED_OVERRIDE_CONTRACT_HASHES = {}
    # 绑定原始 register（即使已被 seal）——strict_register 的重复检查会绕过
    # declared-override / semantic_version 逻辑，治理测试需要原始语义。
    _TestRegistry.register = classmethod(_ORIGINAL_REGISTER_FUNC)


@pytest.fixture(autouse=True)
def _reset_test_registry():
    """Reset the isolated registry class state before each test."""
    _reset_registry_state()
    yield
    _reset_registry_state()


def _make_op(canonical, param_names=("x",), source="test_src"):
    class _Op(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical, category="test", description="t",
            param_names=list(param_names),
        )

        def _calculate_series(self, *a, **k):
            return None

    return _Op(), param_names, source


# ---------------------------------------------------------------------------
# #208: mutation requires the bootstrap token; freeze makes dicts immutable.
# ---------------------------------------------------------------------------
class TestRegistryMutationToken:
    def test_registry_mutation_requires_token(self):
        reg = _TestRegistry
        op, names, src = _make_op("tok_op")
        # register during BUILDING with token held -> OK
        reg.register(op, canonical="tok_op", backend="pandas_numpy", source=src, status="research")
        assert reg.get("tok_op", mode="any") is not None
        # finalize invalidates the token
        reg.finalize()
        with pytest.raises((RuntimeError, PermissionError)):
            reg.register(op, canonical="tok_op", backend="polars", source=src, status="research")
        # thaw restores the token -> mutation again allowed
        reg.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
        reg.register(op, canonical="tok_op", backend="polars", source=src, status="research")
        reg.finalize()
        reg.freeze()

    def test_frozen_dicts_immutable(self):
        reg = _TestRegistry
        op, names, src = _make_op("frozen_op")
        reg.register(op, canonical="frozen_op", backend="pandas_numpy", source=src, status="research")
        reg.finalize()
        reg.freeze()
        # direct dict mutation is impossible on the immutable snapshot
        with pytest.raises(TypeError):
            reg._operators["frozen_op"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            reg._aliases["x"] = "y"  # type: ignore[index]
        # public read still resolves (research mode bypasses surface gate)
        assert reg.get("frozen_op", mode="any") is not None


# ---------------------------------------------------------------------------
# #209: explicit BackendOverrideSpec pins are order-independent.
# ---------------------------------------------------------------------------
class TestBackendOverrideSpec:
    def test_override_spec_records_contract_hash(self):
        from factor_engine.cleaned_operators.registry import _contract_hash

        reg = _TestRegistry
        op, names, src = _make_op("ovr_op", source="src_a")
        reg.register(op, canonical="ovr_op", backend="pandas_numpy", source="src_a", status="research")
        real_hash = _contract_hash(reg._operators["ovr_op"]["pandas_numpy"])

        spec = BackendOverrideSpec(
            canonical="ovr_op", backend="pandas_numpy",
            expected_old_source="src_a", expected_old_contract_hash=real_hash,
            new_source="src_b",
        )
        reg.register_declared_override("ovr_op", "pandas_numpy", "src_a", "src_b", "r40 test", spec=spec)
        assert reg._DECLARED_OVERRIDE_CONTRACT_HASHES[("ovr_op", "pandas_numpy")] == real_hash

        # Registering the declared new source succeeds.
        op_b, _, _ = _make_op("ovr_op", source="src_b")
        reg.register(op_b, canonical="ovr_op", backend="pandas_numpy", source="src_b",
                     status="research", expected_old_source="src_a", replacement_reason="r40")

    def test_override_spec_wrong_contract_hash_rejected(self):
        from factor_engine.cleaned_operators.registry import _contract_hash

        reg = _TestRegistry
        op, _, _ = _make_op("ovr3_op", source="src_a")
        reg.register(op, canonical="ovr3_op", backend="pandas_numpy", source="src_a", status="research")
        real_hash = _contract_hash(reg._operators["ovr3_op"]["pandas_numpy"])
        spec = BackendOverrideSpec(
            canonical="ovr3_op", backend="pandas_numpy",
            expected_old_source="src_a", expected_old_contract_hash="wrong-hash",
            new_source="src_b",
        )
        reg.register_declared_override("ovr3_op", "pandas_numpy", "src_a", "src_b", "r40", spec=spec)
        op_b, _, _ = _make_op("ovr3_op", source="src_b")
        with pytest.raises(ValueError, match="contract hash"):
            reg.register(op_b, canonical="ovr3_op", backend="pandas_numpy", source="src_b",
                         status="research", expected_old_source="src_a", replacement_reason="r40")

    def test_override_spec_wrong_new_source_rejected(self):
        reg = _TestRegistry
        op, _, _ = _make_op("ovr2_op", source="src_a")
        reg.register(op, canonical="ovr2_op", backend="pandas_numpy", source="src_a", status="research")
        spec = BackendOverrideSpec(
            canonical="ovr2_op", backend="pandas_numpy",
            expected_old_source="src_a", expected_old_contract_hash="",
            new_source="src_b",
        )
        reg.register_declared_override("ovr2_op", "pandas_numpy", "src_a", "src_b", "r40", spec=spec)
        op_c, _, _ = _make_op("ovr2_op", source="src_c")
        with pytest.raises(ValueError, match="new source"):
            reg.register(op_c, canonical="ovr2_op", backend="pandas_numpy", source="src_c",
                         status="research", expected_old_source="src_a", replacement_reason="r40")


# ---------------------------------------------------------------------------
# #211: CanonicalOperatorManifest makes param_names import-order independent.
# ---------------------------------------------------------------------------
class TestCanonicalManifest:
    def test_canonical_contract_independent_of_import_order(self):
        manifest = CanonicalOperatorManifest(
            canonical="manifest_op", param_names=("a", "b"), output_type="series",
        )
        # Register the manifest as the declared authority (direct dict to avoid
        # touching the shared bootstrap registry's lifecycle).
        OperatorRegistry._canonical_manifests["manifest_op"] = manifest
        try:
            # Regardless of which backend registers first, the manifest wins.
            assert _merge_param_names(["z", "w"], ["a", "b"], canonical="manifest_op") == ["a", "b"]
            assert _merge_param_names(None, ["a", "b"], canonical="manifest_op") == ["a", "b"]
            assert _merge_param_names(["x"], [], canonical="manifest_op") == ["a", "b"]
        finally:
            OperatorRegistry._canonical_manifests.pop("manifest_op", None)

    def test_first_registered_semantics_without_manifest(self):
        # No manifest -> historical first-registered wins.
        assert _merge_param_names(["z"], ["a", "b"]) == ["z"]


# ---------------------------------------------------------------------------
# #212: production registration requires explicit semantic_version.
# ---------------------------------------------------------------------------
class TestSemanticVersionGate:
    def test_missing_semantic_version_rejected(self):
        # A production-status operator WITHOUT an explicit semantic_version is
        # flagged by the production gate (no silent "1.0" default).
        reg = _TestRegistry
        op, names, src = _make_op("sv_op")
        reg.register(op, canonical="sv_op", backend="pandas_numpy",
                     source=src, status="production")
        missing = reg.assert_production_semantic_versions_declared(["sv_op"])
        assert "sv_op" in missing

    def test_assert_production_semantic_versions_declared(self):
        reg = _TestRegistry
        op, names, src = _make_op("sv2_op")
        reg.register(op, canonical="sv2_op", backend="pandas_numpy",
                     source=src, status="production", semantic_version="2.0.0")
        missing = reg.assert_production_semantic_versions_declared(["sv2_op"])
        assert missing == []
