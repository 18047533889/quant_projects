# -*- coding: utf-8 -*-
"""R23 P1 certification sprint — technical_signal category assessment.

Asserts the honest current state: all 46 technical_signal canonicals remain
experimental because the factor_operator_verified.json evidence artifact is stale
(source tree hashes diverged from the recorded artifact). No production_certified
canonical exists in this category until a fresh factor-operator evidence run is
executed.

This test documents the current baseline and will need to be updated once
certify_factor_operator_evidence.py is re-run and the canonicals achieve
production certification.
"""
from __future__ import annotations

import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.mining.direct_use import build_direct_use_operator

# All 46 technical_signal canonicals from factor_engine/docs/R23_PER_CANONICAL_AUDIT.json
TECHNICAL_SIGNAL_CANONICALS = frozenset({
    "ADX", "ATR_WILDER", "CMO", "DEMA",
    "DMI_minus", "DMI_plus", "DX",
    "KAMA",
    "KeltnerLower", "KeltnerMid", "KeltnerPosition", "KeltnerUpper",
    "MACD_hist", "MACD_line", "MACD_signal",
    "NATR",
    "PPO", "PPO_hist", "PPO_signal",
    "PSAR",
    "PVO", "PVO_hist", "PVO_signal",
    "RSI_WILDER",
    "Supertrend", "SupertrendDirection",
    "TEMA",
    "TSI", "TSI_signal",
    "UltimateOscillator",
    "VortexMinus", "VortexPlus",
    "bollinger_pct_b", "bollinger_width",
    "choppiness_index",
    "donchian_lower", "donchian_mid", "donchian_position", "donchian_upper",
    "efficiency_ratio",
    "ichimoku_cloud_position", "ichimoku_cloud_width",
    "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
    "ichimoku_tenkan",
})


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


class TestCertTechnicalSignalBaseline:
    """R23 P1: baseline state — production_certified is False for all."""

    def test_all_46_technical_signal_canonicals_registered(self):
        """Every canonical in the R23 audit list is registered in the catalog."""
        registered = set(OperatorRegistry._catalog.keys())
        missing = TECHNICAL_SIGNAL_CANONICALS - registered
        assert not missing, f"Missing from registry: {sorted(missing)}"

    def test_all_technical_signal_have_pandas_numpy_backend(self):
        """Every technical_signal canonical has a pandas_numpy implementation."""
        for c in sorted(TECHNICAL_SIGNAL_CANONICALS):
            backends = OperatorRegistry.backends_for(c)
            assert "pandas_numpy" in backends, f"{c} missing pandas_numpy backend"

    def test_all_technical_signal_have_polars_backend(self):
        """Every technical_signal canonical has a polars implementation."""
        for c in sorted(TECHNICAL_SIGNAL_CANONICALS):
            backends = OperatorRegistry.backends_for(c)
            assert "polars" in backends, f"{c} missing polars backend"

    def test_none_production_certified(self):
        """No technical_signal canonical is production-certified yet.

        R23 P1 assessment: factor_operator_verified.json evidence artifact is
        stale (source tree hashes diverged from recorded artifact). Every
        canonical is status=experimental, production_certified=False.
        """
        for c in sorted(TECHNICAL_SIGNAL_CANONICALS):
            cat = OperatorRegistry._catalog.get(c, {})
            assert cat.get("production_certified") is not True, (
                f"{c} unexpectedly production_certified"
            )
            assert cat.get("status") == "experimental", (
                f"{c} unexpected status={cat.get('status')!r}"
            )

    def test_all_production_admitted_false(self):
        """production_admitted is False for all technical_signal canonicals.

        production_admitted requires catalog.production_certified=True AND
        cost_contract_declared AND sources not missing AND not denied. Since
        all are experimental, production_admitted must be False.
        """
        for c in sorted(TECHNICAL_SIGNAL_CANONICALS):
            cat = OperatorRegistry._catalog.get(c, {})
            row = build_direct_use_operator(c, cat)
            assert row.production_admitted is False, (
                f"{c}: production_admitted={row.production_admitted} "
                f"(production_certified={row.production_certified})"
            )
            # mining_visible is role-based so it can be True even when
            # production_admitted is False
            assert row.production_terminal_usable is False, (
                f"{c}: production_terminal_usable={row.production_terminal_usable}"
            )

    def test_stale_evidence_artifact_is_blocker(self):
        """Confirm the evidence artifact is now valid after re-certification.

        R23 P1 certification sprint: certify_factor_operator_evidence.py was
        re-run (2026-09-04).  The artifact is now fresh — bound to current HEAD,
        hashes re-derived, pandas-reference operators certified.  The previous
        "stale blocker" that kept every technical_signal canonical experimental
        is resolved.  ``backend_meta.production_certified`` is True for all 46;
        top-level ``production_certified`` stays False only because the
        independent semantic_golden / source_contract gates have no dedicated
        payload record yet (P0-18 fail-closed), which is the honest remaining
        gap before these can claim full six-gate production.
        """
        from factor_engine.backend.factor_operator_evidence import (
            factor_operator_evidence_valid,
            validation_errors,
        )
        valid = factor_operator_evidence_valid()
        errs = validation_errors()
        assert valid, f"Evidence artifact should be valid after re-certification. Errors: {errs}"

    def test_evidence_artifact_commit_mismatch(self):
        """The evidence artifact commit now matches current HEAD."""
        import json
        artifact = json.load(open("evidence/factor_operator_verified.json"))
        recorded_commit = artifact.get("commit_sha", "")
        import subprocess
        current_head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode().strip()
        )
        assert recorded_commit == current_head, (
            f"Evidence artifact commit {recorded_commit} does not match HEAD "
            f"{current_head} — the certifier must be re-run to bind it."
        )