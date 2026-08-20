# -*- coding: utf-8 -*-
"""R20-EXEC-CONTRACT-STATEFUL: execution_contract statefulness extension for
the R20 recursive derivative canonicals.

Before this slice, the R20 derivative canonicals (atr_pct family, keltner_*
derivatives, ema/dema/tema_distance_pct, ma_slope_pct, ema_crossover,
supertrend_* derivatives, bvc_imbalance_ma, psar_* derivatives,
kama_distance_pct) were tagged ``stateful``/``full_replay`` only in
technical/indicators_v2.py's ``_RECURSIVE_EWM`` registration tags — which
``runtime.execution_contract.execution_contract()`` NEVER consults.  They
resolved ``state_model='stateless'`` while genuinely depending on recursive
full history: an incremental/chunked execution would silently drop the
recursive warmup.

The fix declares them via ``declare_stateful`` in
``cleaned_operators/stateful_contract_migration.py``'s new
``_R20_DERIVATIVE_RECURSIVE`` tuple (the documented single authority — never
an edit to the legacy ``_STATEFUL_CANONICALS`` seed).

Contract under test:
* every ``_R20_DERIVATIVE_RECURSIVE`` member resolves
  ``state_model='recursive'`` / ``chunking='required_full_history'`` /
  ``legacy_seed_fallback=False`` (declared, not seed-inherited);
* none of them is in the legacy ``_STATEFUL_CANONICALS`` seed (the declaration
  is the ONLY authority — proving they were previously silently stateless);
* genuinely stateless operators (``ts_mean``, ``cs_zscore``) stay stateless;
* the migration idempotence guard (``_APPLIED``) does not silently re-declare;
* the declared set contains the union of the three migration tuples (subset
  direction — other modules declare too; catches a member failing to declare);
"""
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators import stateful_contract_migration as _scm
from cleaned_operators.stateful_contract_migration import (
    _CHECKPOINT_RECURSIVE,
    _FULL_HISTORY_RECURSIVE,
    _R20_DERIVATIVE_RECURSIVE,
    apply_stateful_contract_migration,
)
from runtime.execution_contract import (
    ExecutionContractResolutionError,
    _STATEFUL_CANONICALS,
    declared_stateful_canonicals,
    execution_contract,
)

load_all()
apply_stateful_contract_migration()


def test_every_r20_derivative_is_recursive_full_history():
    for name in _R20_DERIVATIVE_RECURSIVE:
        contract = execution_contract(name)
        assert contract.state_model == "recursive", name
        assert contract.chunking == "required_full_history", name
        assert contract.legacy_seed_fallback is False, (
            f"{name}: declared contract must resolve without the legacy seed"
        )


def test_r20_derivatives_were_not_seed_members():
    """The whole point of the slice: before the declaration these names
    resolved stateless because they were NEVER in the legacy seed.  If one
    ever gets added to the seed, the declaration here would become redundant
    (and the seed is a deprecated migration marker, not an authority)."""
    leaked = [n for n in _R20_DERIVATIVE_RECURSIVE if n in _STATEFUL_CANONICALS]
    assert leaked == []


def test_stateless_operators_stay_stateless():
    for name in ("ts_mean", "cs_zscore", "zscore"):
        contract = execution_contract(name)
        assert contract.state_model == "stateless", name
        assert contract.legacy_seed_fallback is False, name


def test_preexisting_declared_contracts_unaffected():
    """ts_ema (checkpoint recursive) and KAMA (full-history recursive) keep
    their existing declared contracts after the R20 extension."""
    ec_ema = execution_contract("ts_ema")
    assert ec_ema.state_model == "recursive"
    assert ec_ema.chunking == "checkpoint"
    ec_kama = execution_contract("KAMA")
    assert ec_kama.state_model == "recursive"
    assert ec_kama.chunking == "required_full_history"


def test_declared_set_contains_union_of_migration_tuples():
    declared = declared_stateful_canonicals()
    union = (
        set(_CHECKPOINT_RECURSIVE) | set(_FULL_HISTORY_RECURSIVE)
        | set(_R20_DERIVATIVE_RECURSIVE)
    )
    # subset, not equality: other modules also declare their own contracts.
    # The assert catches a tuple member silently failing to declare.
    assert union <= declared


def test_no_name_duplicated_across_tuples():
    all_names = (
        list(_CHECKPOINT_RECURSIVE) + list(_FULL_HISTORY_RECURSIVE)
        + list(_R20_DERIVATIVE_RECURSIVE)
    )
    assert len(all_names) == len(set(all_names)), "duplicate declare would raise"


def test_migration_is_idempotent():
    # NOTE: read ``_APPLIED`` as a live MODULE attribute — a from-import at
    # collection time snapshots the pre-fixture ``False`` (the session
    # conftest's ``load_all()`` runs AFTER test-module import).  This assert
    # alone is weak (the module-level apply above already set it); the REAL
    # idempotence pin is that the conftest load_all() apply + the line-51
    # apply both ran without a duplicate-declare raise, plus the direct
    # re-declare below raising against the API.
    assert _scm._APPLIED is True
    # a second apply is a no-op (the _APPLIED guard); a genuine re-declare of
    # any single name must raise — pinned directly against the API.
    from runtime.execution_contract import declare_stateful

    with pytest.raises(ExecutionContractResolutionError):
        declare_stateful(
            "bvc_imbalance_ma",
            state_model="recursive",
            chunking="required_full_history",
        )
