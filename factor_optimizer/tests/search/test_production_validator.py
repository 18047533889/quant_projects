"""Tests for P0-11 production enforcement and validator identity wiring.

The package currently declares itself RESEARCH_ONLY (``PRODUCTION_CAPABILITY``
is ``supported=False``), so ANY PRODUCTION-mode construction is fail-closed at
the capability gate.  These tests split the concerns honestly:

- One test asserts the REAL ``require_production_capability`` raises
  ``CapabilityError`` (the package is not production-ready yet).
- The remaining tests monkeypatch ``require_production_capability`` to a no-op
  so they can exercise the trial_validator / identity / _validate_trial
  enforcement logic in isolation -- without pretending the package can be
  constructed in production today.
"""

import pytest

import factor_optimizer.capabilities
import factor_optimizer.search.runner as runner_module
from factor_optimizer import (
    CapabilityError,
    ExecutionMode,
    MutationGrammarValidator,
    TrialValidatorIdentity,
)
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.search.runner import SearchConfig, SearchRunner
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.trial import Trial, TrialStatus


def _protocol():
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}),
        lambda trial, fid: {
            "score": 0.5,
            "cost": 1.0,
            "evidence_ref": "ev-1",
        },
    )


def _production_config():
    return SearchConfig(
        budget=SearchBudget(max_trials=3, max_evaluations=3),
        execution_mode=ExecutionMode.PRODUCTION,
    )


def _noop_capability(monkeypatch):
    # runner.py imported `require_production_capability` into its own module
    # namespace at import time, so both SearchConfig.__post_init__ and
    # SearchRunner.__init__ resolve it from factor_optimizer.search.runner.
    monkeypatch.setattr(
        runner_module, "require_production_capability", lambda: None
    )


def _validating_trial():
    return Trial(
        trial_id="t1",
        mutation_id="m1",
        status=TrialStatus.VALIDATING,
        parent_factor_ids=["f1"],
        metadata={
            "mutation_type": "parameter_tune",
            "params": {
                "operator_name": "op",
                "parameter_name": "window",
                "new_value": 5.0,
            },
        },
    )


def _callable_grammar_validator():
    """A MutationGrammarValidator exposed as a callable (identity preserved)."""
    inner = MutationGrammarValidator()

    def validate(trial):
        return inner.validate(trial)

    validate.identity = inner.identity
    return validate


def test_production_capability_fails_closed():
    """The real capability gate raises CapabilityError under RESEARCH_ONLY."""
    assert factor_optimizer.capabilities.PRODUCTION_CAPABILITY.status.value == (
        "research_only"
    )
    assert factor_optimizer.capabilities.PRODUCTION_CAPABILITY.supported is False
    with pytest.raises(CapabilityError):
        factor_optimizer.capabilities.require_production_capability()


def test_production_mode_rejects_missing_trial_validator(monkeypatch):
    """PRODUCTION without a trial_validator raises TypeError."""
    _noop_capability(monkeypatch)
    with pytest.raises(TypeError, match="production search requires a trial_validator"):
        SearchRunner(
            config=_production_config(),
            proposal_fn=lambda: _validating_trial(),
            evaluation_fn=_protocol(),
        )


def test_production_mode_rejects_validator_without_identity(monkeypatch):
    """PRODUCTION with a plain callable validator (no identity) raises TypeError."""
    _noop_capability(monkeypatch)

    def plain_validator(trial):
        return {"is_legal": True}

    with pytest.raises(TypeError, match="identity"):
        SearchRunner(
            config=_production_config(),
            proposal_fn=lambda: _validating_trial(),
            evaluation_fn=_protocol(),
            trial_validator=plain_validator,
        )


def test_production_mode_accepts_validator_with_identity(monkeypatch):
    """PRODUCTION with a real TrialValidatorIdentity constructs."""
    _noop_capability(monkeypatch)
    validator = MutationGrammarValidator()
    runner = SearchRunner(
        config=_production_config(),
        proposal_fn=lambda: _validating_trial(),
        evaluation_fn=_protocol(),
        trial_validator=validator,
    )
    assert runner.trial_validator is validator


def test_real_capability_blocks_production_even_with_validator():
    """Without the monkeypatch, PRODUCTION construction always raises CapabilityError."""
    validator = MutationGrammarValidator()
    with pytest.raises(CapabilityError):
        SearchRunner(
            config=_production_config(),
            proposal_fn=lambda: _validating_trial(),
            evaluation_fn=_protocol(),
            trial_validator=validator,
        )


def test_validate_trial_rejects_research_only_in_production(monkeypatch):
    """_validate_trial in PRODUCTION turns a research_only verdict into illegal."""
    _noop_capability(monkeypatch)

    def research_only_validator(trial):
        return {"is_legal": True, "research_only": True}

    # The validator must carry a real identity to survive construction (the
    # identity check is separate from the research_only rejection), so attach
    # one to the callable.
    research_only_validator.identity = MutationGrammarValidator().identity

    runner = SearchRunner(
        config=_production_config(),
        proposal_fn=lambda: _validating_trial(),
        evaluation_fn=_protocol(),
        trial_validator=research_only_validator,
    )
    result = runner._validate_trial(_validating_trial())
    assert result["is_legal"] is False
    assert "research_only" in result["errors"][0]


def test_validate_trial_pins_identity(monkeypatch):
    """_validate_trial pins the validator identity onto the legality result."""
    _noop_capability(monkeypatch)
    validator = _callable_grammar_validator()
    runner = SearchRunner(
        config=_production_config(),
        proposal_fn=lambda: _validating_trial(),
        evaluation_fn=_protocol(),
        trial_validator=validator,
    )
    result = runner._validate_trial(_validating_trial())
    assert isinstance(result, dict)
    assert result["is_legal"] is True
    assert result["validator_id"] == validator.identity.validator_id
    assert result["validator_version"] == validator.identity.validator_version
    assert result["implementation_hash"] == validator.identity.implementation_hash


def test_identity_contract_roundtrip():
    """TrialValidatorIdentity fields are non-empty and hash is hex >= 16 chars."""
    identity = MutationGrammarValidator().identity
    assert isinstance(identity, TrialValidatorIdentity)
    assert identity.validator_id == "factor_optimizer.mutation_grammar"
    assert identity.validator_version == "0.1.0"
    assert len(identity.implementation_hash) >= 16
    d = identity.as_dict()
    assert set(d) == {"validator_id", "validator_version", "implementation_hash"}
