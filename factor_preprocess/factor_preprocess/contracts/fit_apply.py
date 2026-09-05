"""
Fit-apply governance contract for treatments (R61-FI-029, plan §29).

Every treatment step declares, as first-class frozen metadata:

- ``causality_class`` — how the step consumes information (e.g.
  ``cross_sectional_causal`` / ``one_sided_causal`` / ``offline_only`` /
  ``train_fitted``). ``OFFLINE_ONLY`` steps may never run in a live/production
  apply path.
- ``fit_scope`` — which data scope the step may fit on (``train_only`` /
  ``expanding_causal`` / ``cross_sectional_date`` / ``full_sample_research``).
  A ``full_sample_research`` fit scope is NEVER evaluation-valid.
- ``fit_split`` — the split the step's fitted state was derived from
  (train / validation / test / full_sample_research).
- ``state_identity`` — the content identity of the fitted state (binds the
  state to the snapshot/universe/split/feature contract it was fitted on).
- ``application_scope`` — which splits the step may be APPLIED to
  (train / validation / test / production / full_sample_research).

Fail-closed guards (future-leakage safety: 宁严勿松):

- **future append** — a causal step's output must be prefix-invariant: the
  value at time ``t`` must not change when later observations are appended.
  :func:`assert_prefix_invariant` recomputes on a truncated prefix and fails
  when the historical window changes.
- **split leakage** — a step whose fit scope / application scope spans a
  future or test split is rejected. ``assert_no_split_leakage`` fails when the
  application split requires information from a later split.
- **fit-state reuse across wrong snapshot/universe** — fitted state carries
  ``state_identity``; applying it against a data snapshot / universe /
  feature order that does not match the fit context fails closed
  (:func:`assert_fit_state_context_match`).
- **wrong label access** — a step that requires labels may only access labels
  whose knowledge time is <= the current as-of time
  (:func:`assert_label_access_legal`).
- **wrong as-of date** — an apply at an as-of date outside the declared
  ``application_scope`` window / before the fit window completes fails closed
  (:func:`assert_asof_legal`).

R61-FI-044 interplay: a *model-mandated* normalization/neutralization is
admitted only under the representation policy's non-inferiority guard
(:mod:`factor_preprocess.representation.policy`), and the resulting
representation is recorded as a model-specific artifact — never written over
the canonical factor asset. The fit-apply rules in this module are the
step-level execution counterpart of that governance.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from factor_preprocess.errors import (
    ContractError,
    InvalidContractError,
    TimingContractError,
    StaleFittedStateError,
    SnapshotMismatchError,
)


class CausalityClass(str, Enum):
    """Information-consumption class of a treatment step (plan §29)."""

    CROSS_SECTIONAL_CAUSAL = "cross_sectional_causal"
    ONE_SIDED_CAUSAL = "one_sided_causal"
    TRAIN_FITTED = "train_fitted"
    OFFLINE_ONLY = "offline_only"
    NO_OP = "no_op"


class FitScope(str, Enum):
    """The data scope a step is allowed to fit on."""

    TRAIN_ONLY = "train_only"
    EXPANDING_CAUSAL = "expanding_causal"
    CROSS_SECTIONAL_DATE = "cross_sectional_date"
    FULL_SAMPLE_RESEARCH = "full_sample_research"

    @property
    def is_research_only(self) -> bool:
        return self is FitScope.FULL_SAMPLE_RESEARCH


class ApplicationSplit(str, Enum):
    """The split a step may be applied to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"
    FULL_SAMPLE_RESEARCH = "full_sample_research"

    @property
    def is_evaluation_valid(self) -> bool:
        """Splits that feed evaluation/production decisions."""
        return self in (
            ApplicationSplit.VALIDATION,
            ApplicationSplit.TEST,
            ApplicationSplit.PRODUCTION,
        )


@dataclass(frozen=True)
class TreatmentFitApplyDeclaration:
    """Declared fit-apply contract of one treatment step (plan §29).

    Every treatment MUST declare all five fields — there is no default that
    silently means "safe". A declaration is deeply immutable and hash-safe.
    """

    treatment_id: str
    causality_class: CausalityClass
    fit_scope: FitScope
    fit_split: ApplicationSplit
    state_identity: str
    application_scope: ApplicationSplit

    def __post_init__(self):
        for name in ("treatment_id", "state_identity"):
            value = getattr(self, name)
            if not value:
                raise InvalidContractError(
                    f"TreatmentFitApplyDeclaration.{name} cannot be empty"
                )
        for name in ("causality_class", "fit_scope", "fit_split", "application_scope"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, self._coerce(name, value))

    @staticmethod
    def _coerce(name, value):
        mapping = {
            "causality_class": CausalityClass,
            "fit_scope": FitScope,
            "fit_split": ApplicationSplit,
            "application_scope": ApplicationSplit,
        }
        try:
            return mapping[name](value)
        except ValueError:
            raise InvalidContractError(
                f"invalid {name} value {value!r} for treatment fit-apply "
                "declaration"
            )

    # -- hard semantic validations -----------------------------------------

    def validate_self_consistent(self) -> None:
        """Fail closed on internally contradictory declarations."""
        if (
            self.causality_class is CausalityClass.OFFLINE_ONLY
            and self.application_scope.is_evaluation_valid
        ):
            raise InvalidContractError(
                f"treatment {self.treatment_id!r}: OFFLINE_ONLY causality "
                "cannot be applied to an evaluation-valid split "
                f"({self.application_scope.value!r})"
            )
        if (
            self.fit_scope.is_research_only
            and self.application_scope.is_evaluation_valid
        ):
            raise InvalidContractError(
                f"treatment {self.treatment_id!r}: full-sample-research fit "
                "scope cannot be applied to an evaluation-valid split"
            )
        if self.fit_split.is_evaluation_valid and self.fit_scope is FitScope.TRAIN_ONLY:
            raise InvalidContractError(
                f"treatment {self.treatment_id!r}: TRAIN_ONLY fit scope cannot "
                f"declare an evaluation-valid fit_split ({self.fit_split.value!r})"
            )

    def assert_no_split_leakage(
        self, *, application_split=None
    ) -> None:
        """Reject ANY split leakage: only TRAIN-fitted steps are evaluable.

        Standard ML-governance rule (宁严勿松): a fitted treatment step may be
        applied to an evaluation-valid split (validation / test / production)
        ONLY when its fitted state was derived from the TRAIN split. Fitting on
        validation/test/production rows contaminates the fit and is rejected —
        including a "validation-fit applied to validation" in-sample reuse.

        A full-sample-research fit consumes future rows relative to every real
        split and can therefore never be applied to TRAIN / VALIDATION / TEST /
        PRODUCTION (only to research-only full-sample exploration).
        """
        if application_split is None:
            application_split = self.application_scope
        if isinstance(application_split, str):
            application_split = ApplicationSplit(application_split)

        if application_split.is_evaluation_valid:
            if self.fit_split is not ApplicationSplit.TRAIN:
                raise TimingContractError(
                    f"split leakage: treatment {self.treatment_id!r} was fitted "
                    f"on {self.fit_split.value!r} but is being applied to "
                    f"evaluation-valid split {application_split.value!r}; only "
                    "TRAIN-fitted states may feed evaluation/production"
                )
            return

        if application_split is ApplicationSplit.TRAIN:
            if self.fit_split is not ApplicationSplit.TRAIN:
                raise TimingContractError(
                    f"split leakage: treatment {self.treatment_id!r} was fitted "
                    f"on {self.fit_split.value!r} but is being applied to "
                    f"{application_split.value!r} — a later-split fit can never "
                    "be applied to an earlier split"
                )
            return

        # FULL_SAMPLE_RESEARCH application: a full-sample research fit is
        # allowed (research-only exploration); a TRAIN fit is also allowed for
        # research convenience. Anything else is contradictory.
        if self.fit_split not in (
            ApplicationSplit.TRAIN,
            ApplicationSplit.FULL_SAMPLE_RESEARCH,
        ):
            raise TimingContractError(
                f"split leakage: treatment {self.treatment_id!r} fitted on "
                f"{self.fit_split.value!r} cannot be applied to "
                f"{application_split.value!r}"
            )

    def assert_asof_legal(self, asof_timestamp, fit_end_timestamp) -> None:
        """Reject an apply whose as-of date precedes the fit window end."""
        if fit_end_timestamp is not None and asof_timestamp < fit_end_timestamp:
            raise TimingContractError(
                f"wrong as-of date: applying treatment {self.treatment_id!r} at "
                f"{asof_timestamp} before its fit window completed at "
                f"{fit_end_timestamp} would use a partially-fitted state"
            )


# ---------------------------------------------------------------------------
# Reusable assertion helpers (the five adversarial guards).
# ---------------------------------------------------------------------------


def assert_prefix_invariant(
    compute_fn,
    values,
    *,
    split_index,
    atol: float = 1e-12,
) -> None:
    """Reject FUTURE APPEND leakage: recompute and compare on a prefix.

    ``compute_fn`` must be deterministic on its full input (stateless or
    frozen fitted state). The guard computes ``compute_fn(full_values)``,
    truncates at ``split_index``, recomputes ``compute_fn(prefix)``, and fails
    when the two disagree on the shared historical window. A step whose value
    at time ``t`` changes when later rows are appended is NOT causal.
    """
    import numpy as np

    full = compute_fn(values)
    prefix_values = values[:split_index]
    prefix = compute_fn(prefix_values)

    full_hist = np.asarray(full[:split_index])
    prefix_arr = np.asarray(prefix)

    if full_hist.shape != prefix_arr.shape:
        raise TimingContractError(
            "future append leakage: full-window computation produces shape "
            f"{full_hist.shape} but prefix-only computation produces "
            f"{prefix_arr.shape}"
        )
    if full_hist.dtype.kind in ("f", "c"):
        mismatch = ~np.isclose(full_hist, prefix_arr, atol=atol, equal_nan=True)
    else:
        mismatch = full_hist != prefix_arr
    if np.any(mismatch):
        raise TimingContractError(
            "future append leakage: the value at time t changed when later "
            "observations were appended (prefix-invariance violated)"
        )


def assert_fit_state_context_match(
    *,
    state_identity: str,
    state_data_snapshot_ref: str,
    apply_data_snapshot_ref: str,
    state_universe_ref: str,
    apply_universe_ref: str,
    state_feature_order,
    apply_feature_order,
) -> None:
    """Reject FIT-STATE REUSE across a wrong snapshot or universe.

    A fitted state is bound to the data snapshot / universe / feature order it
    was fitted on. Reusing it against a different snapshot or universe fails
    closed (:class:`SnapshotMismatchError` / :class:`StaleFittedStateError`).
    """
    if state_data_snapshot_ref != apply_data_snapshot_ref:
        raise SnapshotMismatchError(
            "fit-state reuse across wrong data snapshot: fitted state "
            f"{state_identity!r} was fitted on snapshot "
            f"{state_data_snapshot_ref!r} but is being applied to "
            f"{apply_data_snapshot_ref!r}"
        )
    if state_universe_ref != apply_universe_ref:
        raise StaleFittedStateError(
            "fit-state reuse across wrong universe: fitted state "
            f"{state_identity!r} was fitted on universe {state_universe_ref!r} "
            f"but is being applied to {apply_universe_ref!r}"
        )
    if tuple(state_feature_order) != tuple(apply_feature_order):
        raise StaleFittedStateError(
            "fit-state reuse across wrong feature order: fitted state "
            f"{state_identity!r} expects feature order {tuple(state_feature_order)} "
            f"but the apply carries {tuple(apply_feature_order)}"
        )


def assert_label_access_legal(
    *,
    label_knowledge_time,
    asof_timestamp,
    label_name: str = "label",
) -> None:
    """Reject WRONG LABEL ACCESS: a label must be known at the as-of time.

    A label whose knowledge time is AFTER the as-of timestamp must never be
    accessible from the treatment step — reading it would be future leakage.
    """
    if label_knowledge_time > asof_timestamp:
        raise TimingContractError(
            f"wrong label access: {label_name} knowledge time "
            f"{label_knowledge_time} is after the apply as-of "
            f"{asof_timestamp}; the label is not PIT-accessible and must not "
            "be read"
        )


__all__ = [
    "CausalityClass",
    "FitScope",
    "ApplicationSplit",
    "TreatmentFitApplyDeclaration",
    "assert_prefix_invariant",
    "assert_fit_state_context_match",
    "assert_label_access_legal",
]
