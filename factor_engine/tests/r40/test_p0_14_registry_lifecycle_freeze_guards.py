# -*- coding: utf-8 -*-
"""P0-14: OperatorRegistry lifecycle (BUILDING → FINALIZED → FROZEN) must be
respected by every register/replace/alias path.

These tests are the freeze-guard contract for the CLOSER P0-14 work:

* ``register_operator`` / ``OperatorRegistry.register`` must refuse to mutate
  after freeze.
* A direct write into the frozen live dicts (``_operators`` / ``_aliases`` /
  ``_catalog`` — including NESTED catalog entries and ``backend_meta``) must
  raise ``TypeError`` (MappingProxyType), never silently mutate.
* Aliasing a new name after freeze must fail loudly.
* ``snapshot()`` / ``catalog()`` return detached deep copies — mutating the
  returned object must never leak into live registry state.

Uses a fresh ``OperatorRegistry`` subclass so the shared bootstrap registry is
never mutated (same isolation pattern as the R40 governance tests)."""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.registry import (
    OperatorRegistry,
    _BOOTSTRAP_TOKEN,
)


class _TestRegistry(OperatorRegistry):
    """Isolated registry for lifecycle freeze-guard tests."""

    _operators = {}
    _aliases = {}
    _catalog = {}
    _lifecycle = OperatorRegistry.Lifecycle.BUILDING
    _version = 0
    _first_registered = {}
    _overwrite_log = []
    _pending_aliases = []
    _override_chain = {}
    _canonical_manifests = {}
    _frozen = None
    _mutation_token = _BOOTSTRAP_TOKEN
    _DECLARED_OVERRIDE_MANIFEST = {}
    _DECLARED_OVERRIDE_CONTRACT_HASHES = {}


def _reset_registry_state() -> None:
    _TestRegistry._operators = {}
    _TestRegistry._aliases = {}
    _TestRegistry._catalog = {}
    _TestRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING
    _TestRegistry._version = 0
    _TestRegistry._first_registered = {}
    _TestRegistry._overwrite_log = []
    _TestRegistry._pending_aliases = []
    _TestRegistry._override_chain = {}
    _TestRegistry._canonical_manifests = {}
    _TestRegistry._frozen = None
    _TestRegistry._mutation_token = _BOOTSTRAP_TOKEN
    _TestRegistry._DECLARED_OVERRIDE_MANIFEST = {}
    _TestRegistry._DECLARED_OVERRIDE_CONTRACT_HASHES = {}


@pytest.fixture(autouse=True)
def _reset_test_registry():
    _reset_registry_state()
    yield
    _reset_registry_state()


def _make_op(canonical, param_names=("x",), source="test_src"):
    from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator

    class _Op(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical, category="test", description="t",
            param_names=list(param_names),
        )

        def _calculate_series(self, *a, **k):
            return None

    return _Op(), param_names, source


def _bootstrap_two_ops(reg) -> None:
    """Register two canonicals (with an alias) and freeze."""
    op_a, _, src_a = _make_op("p014_a", source="src_a")
    reg.register(op_a, canonical="p014_a", backend="pandas_numpy",
                 source=src_a, status="research")
    op_b, _, src_b = _make_op("p014_b", source="src_b")
    reg.register(op_b, canonical="p014_b", backend="pandas_numpy",
                 source=src_b, status="research", aliases=["p014_alias_b"])
    reg.finalize()
    reg.freeze()


# ---------------------------------------------------------------------------
# Guard #1: register_operator after freeze must refuse.
# ---------------------------------------------------------------------------
class TestRegisterGuard:
    def test_register_operator_refuses_after_freeze(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        assert reg.lifecycle() == "frozen"

        with pytest.raises((RuntimeError, PermissionError)):
            op, _, src = _make_op("p014_z", source="src_z")
            reg.register(op, canonical="p014_z", backend="pandas_numpy",
                         source=src, status="research")

    def test_decorator_register_refuses_after_freeze(self):
        from factor_engine.cleaned_operators.base import register_operator

        reg = _TestRegistry
        _bootstrap_two_ops(reg)

        with pytest.raises(Exception):
            @register_operator(name="p014_dec_z", category="misc",
                               canonical="p014_dec_z")
            class _Z:
                pass

    def test_register_alias_refuses_after_freeze(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        with pytest.raises((RuntimeError, PermissionError)):
            reg.register_alias("p014_alias_z", "p014_a")

    def test_replace_backend_refuses_after_freeze(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        op, _, src = _make_op("p014_a", source="src_c")
        with pytest.raises((RuntimeError, PermissionError)):
            reg.register(op, canonical="p014_a", backend="pandas_numpy",
                         source=src, status="research", replace=True,
                         replacement_reason="p0-14 probe")


# ---------------------------------------------------------------------------
# Guard #2: direct writes into frozen live dicts must raise TypeError.
# ---------------------------------------------------------------------------
class TestDictWriteGuard:
    def test_operators_top_level_write_raises(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        with pytest.raises(TypeError):
            reg._operators["p014_z"] = {}  # type: ignore[index]

    def test_operators_backend_slot_write_raises(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        impls = reg._operators["p014_a"]
        with pytest.raises(TypeError):
            impls["polars"] = None  # type: ignore[index]

    def test_aliases_write_raises(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        with pytest.raises(TypeError):
            reg._aliases["p014_z"] = "p014_a"  # type: ignore[index]

    def test_catalog_top_level_write_raises(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        with pytest.raises(TypeError):
            reg._catalog["p014_z"] = {}  # type: ignore[index]

    def test_catalog_entry_write_raises(self):
        """Nested catalog entries are frozen leaves — a module holding a
        reference from ``_catalog.get(canon)`` cannot mutate live state."""
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        entry = reg._catalog.get("p014_a")
        with pytest.raises(TypeError):
            entry["surface"] = "zz_probe"  # type: ignore[index]

    def test_backend_meta_write_raises(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        bm = reg._catalog.get("p014_a").get("backend_meta")  # type: ignore[union-attr]
        with pytest.raises(TypeError):
            bm["pandas_numpy"]["zz_probe"] = 1  # type: ignore[index]


# ---------------------------------------------------------------------------
# Guard #3: public snapshot / catalog reads are detached copies.
# ---------------------------------------------------------------------------
class TestDetachedCopyGuard:
    def test_catalog_export_is_detached(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        exported = reg.catalog()
        assert exported.get("p014_a") is not reg._catalog.get("p014_a")
        exported["p014_a"]["zz_probe"] = 1  # type: ignore[index]
        assert "zz_probe" not in reg._catalog.get("p014_a")  # type: ignore[union-attr]

    def test_snapshot_is_detached(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        snap = reg.snapshot()
        snap["catalog"].get("p014_a")["zz_probe"] = 1  # type: ignore[index]
        assert "zz_probe" not in reg._catalog.get("p014_a")  # type: ignore[union-attr]
        assert "zz_probe" not in reg._frozen["catalog"].get("p014_a")  # type: ignore[index]

    def test_live_registry_unmutated_after_guards(self):
        reg = _TestRegistry
        _bootstrap_two_ops(reg)
        # Exhaust every probe that must NOT mutate live state.
        try:
            reg.register(*[], canonical="p014_z", backend="pandas_numpy",
                         source="s", status="research")
        except Exception:
            pass
        try:
            reg._catalog["p014_z"] = {}
        except Exception:
            pass
        try:
            reg._aliases["p014_z"] = "p014_a"
        except Exception:
            pass
        assert "p014_z" not in reg._operators
        assert "p014_z" not in reg._catalog
        assert "p014_z" not in reg._aliases
        assert reg.get("p014_a", mode="any") is not None
        assert reg.resolve_canonical("p014_alias_b") == "p014_b"
