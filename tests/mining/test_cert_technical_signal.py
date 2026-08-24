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

# All 46 technical_signal canonicals from docs/R23_PER_CANONICAL_AUDIT.json
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
        """Confirm the evidence artifact is stale (root cause)."""
        from factor_engine.backend.factor_operator_evidence import (
            factor_operator_evidence_valid,
            validation_errors,
        )
        valid = factor_operator_evidence_valid()
        errs = validation_errors()
        assert not valid, (
            "Evidence artifact should be stale (test environment). "
            f"If it's valid now, the certification sprint can proceed. Errors: {errs}"
        )
        assert any("hashes are stale" in e for e in errs), (
            f"Expected stale-hash error, got: {errs}"
        )

    def test_evidence_artifact_commit_mismatch(self):
        """The evidence artifact commit does not match current HEAD."""
        import json
        artifact = json.load(open("evidence/factor_operator_verified.json"))
        recorded_commit = artifact.get("commit_sha", "")
        import subprocess
        current_head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode().strip()
        )
        assert recorded_commit != current_head, (
            f"Evidence artifact commit {recorded_commit} matches HEAD — "
            "evidence may be valid. If so, the certification blocker is resolved."
        )