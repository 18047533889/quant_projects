# -*- coding: utf-8 -*-
"""R23-245..250 / R23-301: single registration authority for period operators.

``quarter_from_cumulative`` / ``ttm_from_quarterly`` / ``ttm_from_cumulative`` /
``yoy_by_period`` (and the strict period_change/lag/average/cagr) must resolve to
the STRICT fiscal implementation with an import-order-invariant provenance — the
old row-walking ``period_helpers`` registration must NOT first-occupy the
canonical and be silently overridden.
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

_PERIOD_CANONICALS = (
    "period_lag",
    "period_change",
    "period_average",
    "period_cagr",
    "quarter_from_cumulative",
    "ttm_from_quarterly",
    "ttm_from_cumulative",
    "yoy_by_period",
)


def test_strict_period_ops_are_the_first_registered_authority():
    for name in _PERIOD_CANONICALS:
        fr = OperatorRegistry.first_registered(name)
        assert fr.get("first_registered_source") == "operator_overhaul_audited", (
            f"{name}: strict fiscal impl must be the FIRST-registered authority "
            f"(got {fr.get('first_registered_source')!r}) — the old row-walking "
            "period_helpers registration must not first-occupy the canonical"
        )
        assert fr.get("first_registered_status") == "production", name


def test_period_ops_have_strict_backend_sources():
    for name in _PERIOD_CANONICALS:
        bm = OperatorRegistry._catalog.get(name, {}).get("backend_meta", {})
        src = (bm.get("pandas_numpy", {}) or {}).get("source", "")
        assert src == "operator_overhaul_audited", name


def test_period_ops_have_multi_backend_parity():
    # The strict fiscal kernel is backend-symmetric (pandas + polars + SQL).
    for name in ("period_change", "ttm_from_quarterly"):
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends, name
        assert "polars" in backends, name


def test_legacy_ttm_quarter_yoy_are_deprecated_compat():
    # The legacy row-walking spellings (ttm / quarter / yoy) were REMOVED from
    # the production catalog; their semantic successors (fin_ttm and the strict
    # period conversions) are marked compatibility-only, never a production
    # authority.
    for name in ("ttm", "quarter", "yoy"):
        assert name not in OperatorRegistry._catalog, (
            f"{name} must not be a registered canonical (row-walking legacy removed)"
        )
    entry = OperatorRegistry._catalog.get("fin_ttm", {})
    assert entry.get("compatibility_only") is True, "fin_ttm must be compat-only"
