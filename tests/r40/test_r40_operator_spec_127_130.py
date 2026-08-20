# -*- coding: utf-8 -*-
"""R40 #127/#128/#129/#130: OperatorSpec real contract values (frequency,
output_type, supports_panel, dual_backend_target)."""
from __future__ import annotations

import pytest
from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(autouse=True)
def _no_load_all(monkeypatch, writable_global_registry):
    """These tests register a minimal operator; avoid the full bootstrap.

    ``infer_polars_long_tier`` (and other capability helpers) call
    ``cleaned_operators.load_all`` at call time via ``from cleaned_operators
    import load_all``, so patching ``cleaned_operators.load_all`` short-circuits
    the slow/failing full bootstrap.  ``writable_global_registry`` 保证在全套件
    中（前序测试已把全局 registry 冻结时）这些 scratch 注册仍可写。
    """
    import cleaned_operators as co

    monkeypatch.setattr(co, "load_all", lambda *a, **k: None)
    yield
    # cleanup: remove test operators
    for canon in ("r40_spec_grain", "r40_spec_scalar", "r40_spec_panel", "r40_spec_scalar_only"):
        try:
            OperatorRegistry.unregister(canon)
        except Exception:
            pass


def _register(canonical, *, param_names, return_type="series", input_grain=None, output_grain=None,
              panel_params=None):
    # Real kernel signature: panel params have NO default (positional inputs),
    # scalar knobs have a default — matches _infer_panel_params's signature rule.
    panels = list(panel_params) if panel_params is not None else list(param_names)
    sig_parts = []
    for p in param_names:
        sig_parts.append(p if p in panels else f"{p}=20")
    sig_params = ", ".join(sig_parts)

    class _Op(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical, category="test", description="t",
            param_names=list(param_names), return_type=return_type,
            input_grain=input_grain, output_grain=output_grain,
        )

        def _calculate_series(self, *args, **kwargs):
            return args[0] if args else None

    # Attach a real named signature so _infer_panel_params sees positional params.
    def _make_fn():
        src = "def _calculate_series(self, %s):\n    return args\n" % sig_params
        ns = {}
        exec(compile(src, "<sig>", "exec"), ns)  # noqa: S102 - controlled test signature
        return ns["_calculate_series"]

    _Op._calculate_series = _make_fn()

    register_operator(
        name=canonical, category="test", business_category="test",
        canonical=canonical, source="r40_spec", status="research",
    )(_Op)
    return canonical


class TestManifestFrequency:
    def test_manifest_frequency_from_real_contract(self):
        from cleaned_operators.operator_spec import build_operator_spec, spec_to_manifest_entry

        _register("r40_spec_grain", param_names=["x", "window"],
                  input_grain="minute", output_grain="daily")
        spec = build_operator_spec("r40_spec_grain")
        assert spec is not None
        entry = spec_to_manifest_entry(spec)
        # Real grain contract, not the hardcoded "any".
        assert entry["frequency"] == "daily"
        assert spec.frequency == "daily"


class TestManifestOutputType:
    def test_manifest_output_type_matches_contract(self):
        from cleaned_operators.operator_spec import build_operator_spec, spec_to_manifest_entry

        _register("r40_spec_scalar", param_names=["x"], return_type="scalar")
        spec = build_operator_spec("r40_spec_scalar")
        entry = spec_to_manifest_entry(spec)
        assert entry["output_type"] == "scalar"
        assert spec.output_type == "scalar"


class TestSupportsPanel:
    def test_supports_panel_from_real_capability(self):
        from cleaned_operators.operator_spec import build_operator_spec

        # panel_params inference from kernel signature: x has no default -> panel.
        _register("r40_spec_panel", param_names=["x", "window"], panel_params=["x"])
        spec = build_operator_spec("r40_spec_panel")
        assert spec.panel_params == ("x",)
        assert spec.supports_panel is True

    def test_supports_panel_false_for_scalar_only(self):
        from cleaned_operators.operator_spec import build_operator_spec

        _register("r40_spec_scalar_only", param_names=["window"], panel_params=[])
        spec = build_operator_spec("r40_spec_scalar_only")
        # No panel input inferred (only a defaulted scalar knob) -> not panel-based.
        assert spec.panel_params == ()
        assert spec.supports_panel is False


class TestDualBackendTarget:
    def test_dual_backend_target_from_backend_evidence(self, monkeypatch):
        # dual_backend_target must be derived from certified backend evidence, NOT
        # from execution_kind.  A research-status op with 2 eligible backends is
        # still NOT a dual-backend target because allow_in_production is False.
        from cleaned_operators.operator_spec import build_operator_spec
        import backend.operator_capability as oc

        _register("r40_spec_panel", param_names=["x", "window"])

        monkeypatch.setattr(
            oc, "production_eligible_backends",
            lambda c, **k: ("pandas_numpy", "polars") if c == "r40_spec_panel" else (),
        )
        spec = build_operator_spec("r40_spec_panel")
        # execution_kind for a plain primitive would be "primitive" — the OLD rule
        # would have returned True; the evidence rule correctly returns False.
        assert spec.dual_backend_target is False

    def test_dual_backend_target_requires_two_eligible_backends(self, monkeypatch):
        from cleaned_operators.operator_spec import build_operator_spec
        import backend.operator_capability as oc

        _register("r40_spec_panel", param_names=["x", "window"])
        monkeypatch.setattr(
            oc, "production_eligible_backends",
            lambda c, **k: ("pandas_numpy",) if c == "r40_spec_panel" else (),
        )
        spec = build_operator_spec("r40_spec_panel")
        assert spec.dual_backend_target is False


def _register_production(canonical):
    from cleaned_operators.registry import OperatorRegistry

    class _Op(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical, category="test", description="t",
            param_names=["x", "window"], return_type="series",
        )

        def _calculate_series(self, *args, **kwargs):
            return args[0] if args else None

    OperatorRegistry.register(
        _Op(), canonical=canonical, backend="pandas_numpy",
        source="r40_spec", status="production", semantic_version="1.0.0",
    )
    return canonical
