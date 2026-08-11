# -*- coding: utf-8 -*-
"""SampleAdequacyContract wiring audit — dead-contract-field hard gate + wiring
helpers (Model Layer Major Redesign §5).

Audit item (R4x wiring closure):
1. ``sample_adequacy_met`` used ``contract.X is not None and telemetry.X is
   not None`` for the optional fields — a contract that REQUIRES a field while
   the telemetry did not supply it was silently skipped -> PASS (false green).
   Fixed to three-state: not-required -> skip; required + supplied -> compare;
   required + not supplied (None) -> FAIL with
   ``"<field> required but telemetry not supplied"``.
2. New wiring helpers in ``modeling/sample_policy.py``:
   ``resolve_sample_contract`` (learner -> default contract) and
   ``adequacy_failures`` (telemetry -> failures, every field passed for real).
"""
from __future__ import annotations

import pytest

from modeling.contracts import SampleAdequacyContract, sample_adequacy_met
from modeling.learners.base import BaseLearner, LearnerSpec
from modeling.learners.pcr import PCRLearner
from modeling.sample_policy import (
    SampleTelemetry,
    adequacy_failures,
    adequacy_report,
    resolve_sample_contract,
)


# --------------------------------------------------------------------------- #
# 1. three-state hard gate on optional contract fields
# --------------------------------------------------------------------------- #
def _base_contract(**optional) -> SampleAdequacyContract:
    kw = dict(
        min_raw_obs=10,
        min_effective_obs=5,
        min_unique_dates=3,
        min_unique_stocks=2,
        min_obs_per_parameter=2.0,
    )
    kw.update(optional)
    return SampleAdequacyContract(**kw)


def _met_kwargs(**overrides) -> dict:
    kw = dict(
        contract=_base_contract(),
        raw_obs=100,
        effective_obs=80,
        unique_dates=50,
        unique_stocks=30,
        free_parameter_count=5,
    )
    kw.update(overrides)
    return kw


def test_required_field_missing_telemetry_fails():
    """Contract requires min_regime_obs but telemetry supplies none -> FAIL,
    message names the field."""
    kw = _met_kwargs(contract=_base_contract(min_regime_obs=10))
    ok, fails = sample_adequacy_met(**kw)
    assert not ok
    assert any("regime_obs" in f and "required but telemetry not supplied" in f for f in fails)


def test_required_field_insufficient_fails():
    """Contract requires min_regime_obs and telemetry supplies too little -> FAIL."""
    kw = _met_kwargs(contract=_base_contract(min_regime_obs=10), regime_obs=5)
    ok, fails = sample_adequacy_met(**kw)
    assert not ok
    assert any("regime_obs 5 < 10" in f for f in fails)


def test_required_field_sufficient_passes():
    """Contract requires min_regime_obs and telemetry supplies enough -> PASS."""
    kw = _met_kwargs(contract=_base_contract(min_regime_obs=10), regime_obs=20)
    ok, fails = sample_adequacy_met(**kw)
    assert ok and fails == []


def test_optional_field_not_required_skips_when_missing():
    """Contract does NOT require min_regime_obs -> missing telemetry still PASS."""
    kw = _met_kwargs()  # contract has no min_regime_obs
    ok, fails = sample_adequacy_met(**kw)
    assert ok and fails == []


@pytest.mark.parametrize(
    "contract_kw, telemetry_kw, field",
    [
        ({"min_regime_obs": 10}, {}, "regime_obs"),
        ({"min_expert_obs": 10}, {}, "expert_obs"),
        ({"min_state_transitions": 5}, {}, "state_transitions"),
        ({"min_cross_section_peers": 5}, {}, "cross_section_peers"),
        ({"max_missing_fraction": 0.3}, {}, "missing_fraction"),
        ({"min_date_coverage": 0.5}, {}, "date_coverage"),
    ],
)
def test_every_optional_field_fails_closed_when_not_supplied(
    contract_kw, telemetry_kw, field
):
    """Every optional contract field REQUIRED but telemetry not supplied -> FAIL
    naming that field."""
    kw = _met_kwargs(contract=_base_contract(**contract_kw), **telemetry_kw)
    ok, fails = sample_adequacy_met(**kw)
    assert not ok
    assert any(field in f and "required but telemetry not supplied" in f for f in fails)


def test_supplied_optional_fields_compare_normally():
    """Supplied optional fields compare: each one just-under its floor fails."""
    contract = _base_contract(
        min_regime_obs=20,
        min_expert_obs=15,
        min_state_transitions=5,
        min_cross_section_peers=10,
        max_missing_fraction=0.3,
        min_date_coverage=0.5,
    )
    ok, fails = sample_adequacy_met(
        contract=contract,
        raw_obs=200,
        effective_obs=100,
        unique_dates=20,
        unique_stocks=10,
        free_parameter_count=3,
        regime_obs=19,
        expert_obs=14,
        state_transitions=4,
        cross_section_peers=9,
        missing_fraction=0.4,
        date_coverage=0.4,
    )
    assert not ok
    joined = "; ".join(fails)
    for token in ("regime_obs", "expert_obs", "state_transitions",
                  "cross_section_peers", "missing_fraction", "date_coverage"):
        assert token in joined


def test_obs_per_parameter_uses_free_parameter_count():
    """obs/param = effective_obs / max(1, free_parameters)."""
    contract = _base_contract(min_obs_per_parameter=20.0)
    ok, fails = sample_adequacy_met(
        contract=contract,
        raw_obs=100,
        effective_obs=100,
        unique_dates=50,
        unique_stocks=30,
        free_parameter_count=10,  # 100/10 = 10 < 20 -> fail
    )
    assert not ok
    assert any("obs/param 10.00 < 20.0" in f for f in fails)
    ok, fails = sample_adequacy_met(
        contract=contract,
        raw_obs=100,
        effective_obs=100,
        unique_dates=50,
        unique_stocks=30,
        free_parameter_count=4,  # 100/4 = 25 >= 20 -> pass
    )
    assert ok and fails == []


# --------------------------------------------------------------------------- #
# 2. wiring helper: resolve_sample_contract
# --------------------------------------------------------------------------- #
def test_resolve_sample_contract_class_hits_linear():
    contract = resolve_sample_contract(PCRLearner)
    assert contract is not None
    assert contract.min_raw_obs == 50_000
    assert contract.min_effective_obs == 10_000


def test_resolve_sample_contract_instance_hits_linear():
    spec = LearnerSpec(learner_name="predictive_pcr", family="pcr")
    contract = resolve_sample_contract(PCRLearner(spec))
    assert contract is not None
    assert contract.min_obs_per_parameter == 200.0


class _UnfamilyLearner(BaseLearner):
    name = "unfamily"
    family = ""
    sample_contract_family = ""

    def fit(self, X, y, *, weights=None, aux=None):  # pragma: no cover
        raise NotImplementedError

    def predict(self, frozen, X):  # pragma: no cover
        raise NotImplementedError


class _FamilyOnlyLearner(BaseLearner):
    name = "family_only"
    family = "moe"

    def fit(self, X, y, *, weights=None, aux=None):  # pragma: no cover
        raise NotImplementedError

    def predict(self, frozen, X):  # pragma: no cover
        raise NotImplementedError


def test_resolve_sample_contract_none_for_unfamilied_learner():
    assert resolve_sample_contract(_UnfamilyLearner) is None
    spec = LearnerSpec(learner_name="unfamily", family="")
    assert resolve_sample_contract(_UnfamilyLearner(spec)) is None


def test_resolve_sample_contract_falls_back_to_family_key():
    # family = "moe" resolves to the moe default contract.
    contract = resolve_sample_contract(_FamilyOnlyLearner)
    assert contract is not None
    assert contract.min_expert_obs == 1_000


# --------------------------------------------------------------------------- #
# 3. wiring helper: adequacy_failures (SampleTelemetry + dict)
# --------------------------------------------------------------------------- #
def test_adequacy_failures_sample_telemetry_required_field_missing():
    tele = SampleTelemetry(raw_obs=100, effective_obs=80, unique_dates=50,
                           unique_stocks=30)
    contract = _base_contract(min_regime_obs=10)
    failures = adequacy_failures(contract, tele, free_parameters=5)
    assert any("regime_obs required but telemetry not supplied" in f for f in failures)


def test_adequacy_failures_sample_telemetry_regime_supplied():
    tele = SampleTelemetry(raw_obs=100, effective_obs=80, unique_dates=50,
                           unique_stocks=30, regime_obs=20, expert_obs=30)
    contract = _base_contract(min_regime_obs=10, min_expert_obs=15)
    failures = adequacy_failures(contract, tele, free_parameters=5)
    assert failures == []


def test_adequacy_failures_dict_style_keys():
    tele = {
        "raw_obs": 100,
        "effective_obs": 80,
        "n_unique_dates": 50,
        "n_unique_stocks": 30,
        "median_stocks_per_date": 40.0,
        "date_coverage": 0.6,
        "missing_fraction": 0.1,
    }
    contract = _base_contract(min_cross_section_peers=20, max_missing_fraction=0.3,
                              min_date_coverage=0.5)
    failures = adequacy_failures(contract, tele, free_parameters=5)
    assert failures == []


def test_adequacy_failures_dict_missing_required_field_fails_closed():
    tele = {
        "raw_obs": 100,
        "effective_obs": 80,
        "n_unique_dates": 50,
        "n_unique_stocks": 30,
    }
    contract = _base_contract(max_missing_fraction=0.3)
    failures = adequacy_failures(contract, tele, free_parameters=5)
    assert any("missing_fraction required but telemetry not supplied" in f for f in failures)


def test_adequacy_failures_dict_obs_per_parameter_uses_free_params():
    tele = {"raw_obs": 100, "effective_obs": 100, "unique_dates": 50, "unique_stocks": 30}
    contract = _base_contract(min_obs_per_parameter=20.0)
    failures = adequacy_failures(contract, tele, free_parameters=10)  # 100/10 = 10
    assert any("obs/param" in f for f in failures)


def test_adequacy_report_delegates_with_regime_field():
    tele = SampleTelemetry(raw_obs=100, effective_obs=80, unique_dates=50,
                           unique_stocks=30, regime_obs=20)
    ok, fails = adequacy_report(_base_contract(min_regime_obs=10), tele)
    assert ok and fails == []
    tele_missing = SampleTelemetry(raw_obs=100, effective_obs=80, unique_dates=50,
                                   unique_stocks=30)
    ok, fails = adequacy_report(_base_contract(min_regime_obs=10), tele_missing)
    assert not ok
    assert any("regime_obs required but telemetry not supplied" in f for f in fails)
