# -*- coding: utf-8 -*-
"""Model-layer contracts (Model Layer Major Redesign taskbook §1/§5/§7/§11/
§14/§16/§33/§34/§35).

This module is the single home for the NEW model governance types that the
operator registry historically did not model:

* :class:`ModelExecutionClass`      — the five model classes (§1.1).
* :class:`TimingKind`               — the seven timing kinds (§35), superset of
  ``cleaned_operators.model_timing.TimingKind`` (adds
  ``FORWARD_LABEL_SUPERVISED``).
* :class:`RichModelTiming`          — rich per-model timing (§34).
* :class:`SampleAdequacyContract`   — replaces scattered ``if n < 8/10/25``
  guards (§5).
* :class:`LabelContract`            — label origin / maturity / basis (§7).
* :class:`DecisionClock`            — hidden-future-function defense (§11).
* :class:`ParamRole`                — model parameter role taxonomy (§14).
* :class:`ParameterSearchPolicy`    — per-parameter search governance (§16).
* :class:`ModelOperatorSpec`        — the §33 refactored operator contract
  (additive superset of ``cleaned_operators.model_contract.ModelOperatorContract``;
  does not edit the concurrent-shared module).
* :class:`LegacyLocalPredictive`    — the legacy short-window local variant
  marker (§2.1/§67).

The A-share daily clock scenarios (§11.1) are exported as module constants.
"""
from __future__ import annotations

import enum
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ModelExecutionClass",
    "TimingKind",
    "RichModelTiming",
    "SampleAdequacyContract",
    "LabelContract",
    "DecisionClock",
    "TradingTimestamp",
    "SessionPhase",
    "PredictionOutputContract",
    "PredictionBatch",
    "RegimeMetadata",
    "FitFingerprint",
    "ParamRole",
    "ParameterSearchPolicy",
    "ModelOperatorSpec",
    "LegacyLocalPredictive",
    "AFTER_CLOSE_TO_NEXT_VWAP",
    "BEFORE_SAME_DAY_VWAP",
    "ashare_decision_clock",
    "sample_adequacy_met",
    "validate_model_operator_spec",
    "derive_overlap",
    "validate_overlap",
]


class SessionPhase(str, enum.Enum):
    PRE_OPEN = "pre_open"
    OPEN = "open"
    MID = "mid"
    CLOSE = "close"
    POST_CLOSE = "post_close"


@dataclass(frozen=True, order=True)
class TradingTimestamp:
    """Timezone-aware trading instant with explicit market/session phase."""

    value: datetime
    market: str = "CN"
    session: str = "regular"
    phase: SessionPhase = SessionPhase.CLOSE

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime) or self.value.tzinfo is None:
            raise ValueError("TradingTimestamp.value must be timezone-aware datetime")
        if not self.market or not self.session:
            raise ValueError("market and session are required")

    @classmethod
    def parse(cls, value: str | datetime, **kwargs: Any) -> "TradingTimestamp":
        return cls(datetime.fromisoformat(value) if isinstance(value, str) else value, **kwargs)

    def to_dict(self) -> dict[str, str]:
        return {"value": self.value.isoformat(), "market": self.market,
                "session": self.session, "phase": self.phase.value}


def _canonical(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, TradingTimestamp):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _semantic_hash(payload: Any) -> str:
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Model execution class (§1.1)
# --------------------------------------------------------------------------- #
class ModelExecutionClass(str, enum.Enum):
    """The five model-layer classes.  ``D`` alone owns the artifact lifecycle."""

    LOCAL_ROLLING_ESTIMATOR = "local_rolling_estimator"
    RECURSIVE_STATE_ESTIMATOR = "recursive_state_estimator"
    SAME_TIME_CROSS_SECTIONAL = "same_time_cross_sectional"
    PREDICTIVE_SUPERVISED = "predictive_supervised"
    RESEARCH_STRUCTURAL = "research_structural"


# --------------------------------------------------------------------------- #
# 2. Timing kinds (§35) — seven classes.
# --------------------------------------------------------------------------- #
class TimingKind(str, enum.Enum):
    """The seven model timing kinds (§35)."""

    SELF_FIT_DESCRIPTIVE = "self_fit_descriptive"
    PRIOR_FIT_PREDICTIVE = "prior_fit_predictive"
    PRIOR_REFERENCE_CURRENT_QUERY = "prior_reference_current_query"
    SAME_TIME_CROSS_SECTIONAL = "same_time_cross_sectional"
    RECURSIVE_CAUSAL_FILTER = "recursive_causal_filter"
    FORWARD_LABEL_SUPERVISED = "forward_label_supervised"
    MATURED_HISTORICAL_OUTCOME = "matured_historical_outcome"


# --------------------------------------------------------------------------- #
# 3. Rich model timing (§34).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RichModelTiming:
    """Rich per-model timing (§34).  Backward-compatible superset of the R35
    ``cleaned_operators.model_timing.ModelTimingContract`` semantics.

    ``timing_kind`` uses :class:`TimingKind`.  Rule strings are declarative and
    referenced (not re-implemented) by the walk-forward / clock validators.
    """

    timing_kind: str
    decision_time: str = "t"                      # e.g. "t", "t_close", "t_mid"
    feature_cutoff_rule: str = "features through decision_time"
    train_feature_cutoff_rule: str | None = None
    train_label_maturity_rule: str | None = None
    reference_cutoff_rule: str | None = None
    query_time_rule: str | None = None
    cross_section_time_rule: str | None = None
    self_exclusion: bool = False
    state_update_rule: str | None = None
    score_time_rule: str = "decision_time + 1"    # when the score is written
    execution_time_rule: str = "next session VWAP"
    label_contract_id: str | None = None

    def __post_init__(self) -> None:
        if self.timing_kind not in {k.value for k in TimingKind}:
            raise ValueError(f"unknown timing_kind {self.timing_kind!r}")
        if self.timing_kind in {
            TimingKind.PRIOR_FIT_PREDICTIVE.value,
            TimingKind.FORWARD_LABEL_SUPERVISED.value,
        } and self.train_feature_cutoff_rule is None:
            object.__setattr__(self, "train_feature_cutoff_rule", self.feature_cutoff_rule)


# --------------------------------------------------------------------------- #
# 4. Sample adequacy contract (§5) — replaces scattered ``if n < 8/10/25``.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleAdequacyContract:
    """Unified sample-adequacy gate (§5).

    A predictive learner must satisfy these BEFORE training; a local rolling
    estimator uses a window-based contract instead (§5.2).  ``None`` fields are
    not enforced.
    """

    min_raw_obs: int
    min_effective_obs: int
    min_unique_dates: int
    min_unique_stocks: int
    min_obs_per_parameter: float

    # predictive-learner / regime / MoE specific floors
    min_regime_obs: int | None = None
    min_expert_obs: int | None = None
    min_state_transitions: int | None = None
    min_cross_section_peers: int | None = None

    max_missing_fraction: float | None = None
    min_date_coverage: float | None = None

    def requirement_summary(self) -> str:
        parts = [
            f"raw>={self.min_raw_obs}",
            f"effective>={self.min_effective_obs}",
            f"dates>={self.min_unique_dates}",
            f"stocks>={self.min_unique_stocks}",
            f"obs/param>={self.min_obs_per_parameter:.1f}",
        ]
        if self.min_regime_obs is not None:
            parts.append(f"regime_obs>={self.min_regime_obs}")
        if self.min_expert_obs is not None:
            parts.append(f"expert_obs>={self.min_expert_obs}")
        return "; ".join(parts)


def sample_adequacy_met(
    *,
    contract: SampleAdequacyContract,
    raw_obs: int,
    effective_obs: int,
    unique_dates: int,
    unique_stocks: int,
    free_parameter_count: int,
    regime_obs: int | None = None,
    expert_obs: int | None = None,
    state_transitions: int | None = None,
    cross_section_peers: int | None = None,
    missing_fraction: float | None = None,
    date_coverage: float | None = None,
) -> tuple[bool, list[str]]:
    """Check a :class:`SampleAdequacyContract` against measured telemetry.

    Returns ``(met, failures)`` where ``failures`` lists each unmet constraint
    (empty when met).

    ``obs_per_parameter = effective_obs / max(1, free_parameter_count)`` where
    ``free_parameter_count`` is the learner's ``effective_parameter_count``
    (modelling.diagnostics) — the number of free parameters the fit actually
    estimates, never ``len(hyperparams)``.

    Optional contract fields (``min_regime_obs``, ``min_expert_obs``,
    ``min_state_transitions``, ``min_cross_section_peers``,
    ``max_missing_fraction``, ``min_date_coverage``) are three-state:

    * contract field ``None`` (not required)           -> skipped, OK;
    * contract requires + telemetry supplied           -> compared normally;
    * contract requires + telemetry ``None``           -> FAIL with
      ``"<field> required but telemetry not supplied"`` so a required-but-
      unmeasured field can NEVER silently pass the gate.
    """
    failures: list[str] = []
    if raw_obs < contract.min_raw_obs:
        failures.append(f"raw_obs {raw_obs} < {contract.min_raw_obs}")
    if effective_obs < contract.min_effective_obs:
        failures.append(f"effective_obs {effective_obs} < {contract.min_effective_obs}")
    if unique_dates < contract.min_unique_dates:
        failures.append(f"unique_dates {unique_dates} < {contract.min_unique_dates}")
    if unique_stocks < contract.min_unique_stocks:
        failures.append(f"unique_stocks {unique_stocks} < {contract.min_unique_stocks}")
    opp = effective_obs / max(1, free_parameter_count)
    if opp < contract.min_obs_per_parameter:
        failures.append(f"obs/param {opp:.2f} < {contract.min_obs_per_parameter}")
    if contract.min_regime_obs is not None:
        if regime_obs is None:
            failures.append("regime_obs required but telemetry not supplied")
        elif regime_obs < contract.min_regime_obs:
            failures.append(f"regime_obs {regime_obs} < {contract.min_regime_obs}")
    if contract.min_expert_obs is not None:
        if expert_obs is None:
            failures.append("expert_obs required but telemetry not supplied")
        elif expert_obs < contract.min_expert_obs:
            failures.append(f"expert_obs {expert_obs} < {contract.min_expert_obs}")
    if contract.min_state_transitions is not None:
        if state_transitions is None:
            failures.append("state_transitions required but telemetry not supplied")
        elif state_transitions < contract.min_state_transitions:
            failures.append(
                f"state_transitions {state_transitions} < {contract.min_state_transitions}"
            )
    if contract.min_cross_section_peers is not None:
        if cross_section_peers is None:
            failures.append("cross_section_peers required but telemetry not supplied")
        elif cross_section_peers < contract.min_cross_section_peers:
            failures.append(
                f"cross_section_peers {cross_section_peers} < {contract.min_cross_section_peers}"
            )
    if contract.max_missing_fraction is not None:
        if missing_fraction is None:
            failures.append("missing_fraction required but telemetry not supplied")
        elif missing_fraction > contract.max_missing_fraction:
            failures.append(
                f"missing_fraction {missing_fraction:.3f} > {contract.max_missing_fraction}"
            )
    if contract.min_date_coverage is not None:
        if date_coverage is None:
            failures.append("date_coverage required but telemetry not supplied")
        elif date_coverage < contract.min_date_coverage:
            failures.append(f"date_coverage {date_coverage:.3f} < {contract.min_date_coverage}")
    return (not failures), failures


# --------------------------------------------------------------------------- #
# 5. Label contract (§7) — VWAP-to-VWAP is the A-share label authority (§8).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelContract:
    """Label definition (§7).  For A-share, ``return_basis`` defaults to
    ``"vwap_to_vwap"`` (§8)::

        label_t(H) = VWAP_{t+H} / VWAP_t - 1

    ``origin_time`` names the label anchor; ``availability_time_rule`` states
    when the label is truly mature and usable in training.
    """

    label_name: str
    origin_time: str = "t"
    start_time_rule: str = "entry VWAP completes at t close"
    end_time_rule: str = "exit VWAP completes at t+H close"
    availability_time_rule: str = "label matured at t+H close"
    horizon_bars: int = 1
    overlapping: bool = False
    return_basis: str = "vwap_to_vwap"
    entry_price_basis: str = "VWAP_t"
    exit_price_basis: str = "VWAP_{t+H}"
    purge_by_interval: bool = True
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        if self.horizon_bars < 1 or self.embargo_bars < 0:
            raise ValueError("horizon_bars must be >= 1 and embargo_bars >= 0")
        if not self.label_name or not self.return_basis:
            raise ValueError("label_name and return_basis are required")

    @property
    def semantic_hash(self) -> str:
        return _semantic_hash({
            "label_name": self.label_name,
            "horizon_bars": self.horizon_bars,
            "return_basis": self.return_basis,
            "availability_time_rule": self.availability_time_rule,
            "overlapping": self.overlapping,
            "purge_by_interval": self.purge_by_interval,
            "embargo_bars": self.embargo_bars,
        })

    def label_interval(self, anchor_index: int) -> tuple[int, int]:
        """The ``[entry_time, exit_time]`` bar interval of a training sample
        anchored at ``anchor_index`` (§9).  Purge uses interval overlap."""
        return (anchor_index, anchor_index + self.horizon_bars)


# --------------------------------------------------------------------------- #
# 6. Decision clock (§11) — the hidden-future-function defense line.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionClock:
    """Declares when each piece of information is available (§11)."""

    decision_at: str = "t_close"
    feature_available_at: dict[str, str] = field(default_factory=dict)
    label_available_at: str = "t+H close"
    score_written_at: str = "t+1 open"
    execution_at: str = "t+1 VWAP"

    @property
    def semantic_hash(self) -> str:
        return _semantic_hash({
            "decision_timestamp": self.decision_at,
            "feature_timestamps": self.feature_available_at,
            "label_timestamp": self.label_available_at,
            "score_timestamp": self.score_written_at,
            "execution_timestamp": self.execution_at,
        })

    def feature_available(self, feature: str) -> str:
        return self.feature_available_at.get(feature, self.decision_at)


@dataclass(frozen=True)
class PredictionOutputContract:
    """Shape, alignment, finiteness, and status gate for model scores."""

    status: str = "ok"
    allow_nan: bool = False
    ndim: int = 1
    dtype_kinds: str = "fiu"

    def validate(self, values: Any, *, expected_rows: int | None = None) -> np.ndarray:
        out = np.asarray(values)
        if out.ndim != self.ndim:
            raise ValueError(f"prediction ndim {out.ndim} != {self.ndim}")
        if expected_rows is not None and len(out) != expected_rows:
            raise ValueError("prediction rows are not aligned with input rows")
        if out.dtype.kind not in self.dtype_kinds:
            raise ValueError(f"prediction dtype kind {out.dtype.kind!r} is not numeric")
        if not self.allow_nan and not np.isfinite(out).all():
            raise ValueError("prediction output must be finite")
        if self.status != "ok":
            raise ValueError(f"prediction status gate is {self.status!r}")
        return out


@dataclass(frozen=True)
class PredictionBatch:
    values: Any
    row_ids: tuple[Any, ...]
    timestamps: tuple[Any, ...] = ()
    status: str = "ok"

    def validate(self, contract: PredictionOutputContract | None = None) -> np.ndarray:
        out = (contract or PredictionOutputContract()).validate(
            self.values, expected_rows=len(self.row_ids)
        )
        if self.timestamps and len(self.timestamps) != len(self.row_ids):
            raise ValueError("prediction timestamps are not row-aligned")
        if tuple(sorted(self.row_ids, key=str)) != self.row_ids:
            raise ValueError("prediction rows must be deterministic and sorted")
        if self.status != "ok":
            raise ValueError(f"prediction batch status is {self.status!r}")
        return out


@dataclass(frozen=True)
class RegimeMetadata:
    mode: str = "hard"
    gating_transform: str = "identity"
    temperature: float = 1.0
    tie_policy: str = "lower"
    n_regimes: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"hard", "soft"}:
            raise ValueError("unsupported regime mode")
        if self.gating_transform not in {"identity", "zscore", "rank"}:
            raise ValueError("unsupported gating transform")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if self.tie_policy not in {"lower", "upper", "stable"} or self.n_regimes < 2:
            raise ValueError("invalid regime tie policy or regime count")

    @property
    def semantic_hash(self) -> str:
        return _semantic_hash(self.__dict__)


@dataclass(frozen=True)
class FitFingerprint:
    row_order_hash: str
    runtime_environment: Mapping[str, str]
    model_config_hash: str
    code_hash: str

    @classmethod
    def from_rows(
        cls, row_ids: Sequence[Any], model_config: Mapping[str, Any], *, code_hash: str = ""
    ) -> "FitFingerprint":
        environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
        return cls(
            row_order_hash=_semantic_hash([str(x) for x in row_ids]),
            runtime_environment=environment,
            model_config_hash=_semantic_hash(model_config),
            code_hash=code_hash,
        )

    @property
    def semantic_hash(self) -> str:
        return _semantic_hash(self.__dict__)


def derive_overlap(horizon_bars: int, *, stride_bars: int = 1) -> bool:
    if horizon_bars < 1 or stride_bars < 1:
        raise ValueError("horizon_bars and stride_bars must be >= 1")
    return stride_bars < horizon_bars


def validate_overlap(contract: LabelContract, *, stride_bars: int) -> None:
    derived = derive_overlap(contract.horizon_bars, stride_bars=stride_bars)
    if contract.overlapping != derived:
        raise ValueError("label overlap flag does not match horizon/stride derivation")


# A-share daily execution scenarios (§11.1).
AFTER_CLOSE_TO_NEXT_VWAP = "AFTER_CLOSE_TO_NEXT_VWAP"
BEFORE_SAME_DAY_VWAP = "BEFORE_SAME_DAY_VWAP"


def ashare_decision_clock(scenario: str = AFTER_CLOSE_TO_NEXT_VWAP) -> DecisionClock:
    """A-share daily decision clocks (§11.1).

    AFTER_CLOSE_TO_NEXT_VWAP — all of day-t close data is known, factor_t is
    produced after close, execution at t+1 VWAP.

    BEFORE_SAME_DAY_VWAP     — execution at day-t VWAP; every feature must be
    available strictly before that execution (close / full-day VWAP / full-day
    volume are forbidden feature inputs).
    """
    if scenario == AFTER_CLOSE_TO_NEXT_VWAP:
        return DecisionClock(
            decision_at="t_close",
            feature_available_at={
                "open": "t_open",
                "vwap": "t_close",
                "close": "t_close",
                "volume": "t_close",
                "pre_close": "t_prev_close",
            },
            label_available_at="t+H close",
            score_written_at="t+1 open",
            execution_at="t+1 VWAP",
        )
    if scenario == BEFORE_SAME_DAY_VWAP:
        return DecisionClock(
            decision_at="t_prev_close",
            feature_available_at={
                "open": "t_open",          # open of day t precedes VWAP_t execution
                "vwap": "t_close",         # VWAP_t completes only at t close
                "close": "t_close",        # close_t completes after VWAP_t
                "volume": "t_close",       # full-day volume_t completes after VWAP_t
                "pre_close": "t_prev_close",
            },
            label_available_at="t+H close",
            score_written_at="t open",
            execution_at="t VWAP",
        )
    raise ValueError(f"unknown decision-clock scenario {scenario!r}")


# --------------------------------------------------------------------------- #
# 7. Model parameter role taxonomy (§14) — model-level, distinct from the
#    operator alpha-search ParamRole in cleaned_operators.base.
# --------------------------------------------------------------------------- #
class ParamRole(str, enum.Enum):
    """Model-parameter role (§14).  Decides whether a parameter may be searched
    by FactorMiner and at what resolution.

    This is the model-training taxonomy; the operator-level alpha-search roles
    live in ``cleaned_operators.base.ParamRole``.  :func:`operator_role_map`
    bridges the two for reporting.
    """

    ECONOMIC_HORIZON = "economic_horizon"
    MODEL_COMPLEXITY = "model_complexity"
    REGULARIZATION = "regularization"
    ESTIMATOR_RESOLUTION = "estimator_resolution"
    NUMERICAL_POLICY = "numerical_policy"
    DATA_POLICY = "data_policy"
    TIMING_POLICY = "timing_policy"


#: model ParamRole -> operator alpha-search ParamRole (for reporting only).
OPERATOR_ROLE_MAP: dict[ParamRole, str] = {
    ParamRole.ECONOMIC_HORIZON: "horizon",
    ParamRole.MODEL_COMPLEXITY: "model_order",
    ParamRole.REGULARIZATION: "policy",
    ParamRole.ESTIMATOR_RESOLUTION: "estimator_resolution",
    ParamRole.NUMERICAL_POLICY: "numerical",
    ParamRole.DATA_POLICY: "source_policy",
    ParamRole.TIMING_POLICY: "policy",
}


# --------------------------------------------------------------------------- #
# 8. Parameter search policy (§16).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParameterSearchPolicy:
    """Per-parameter search governance (§16)."""

    role: ParamRole
    searchable: bool
    certified_values: tuple | None = None
    continuous_bounds: tuple | None = None
    tune_inside_validation: bool = False
    contributes_to_factor_identity: bool = True
    robustness_only: bool = False

    def __post_init__(self) -> None:
        if self.role == ParamRole.NUMERICAL_POLICY and self.searchable:
            raise ValueError("NUMERICAL_POLICY parameters are never searchable (§15)")
        if self.role == ParamRole.DATA_POLICY and self.searchable:
            raise ValueError("DATA_POLICY parameters are never searchable by FactorMiner (§15)")


# --------------------------------------------------------------------------- #
# 9. Refactored model-operator contract (§33) — additive superset.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelOperatorSpec:
    """The §33 refactored operator-level model contract.

    Additive: it does not modify ``cleaned_operators.model_contract.ModelOperatorContract``
    (concurrent-shared), but ``from_r35_contract()`` builds it from the existing
    R35 contract so a canonical can carry both views.
    """

    canonical: str
    execution_class: ModelExecutionClass
    semantic_role: str                       # model_feature / model_score / diagnostic
    timing_contract_id: str = "auto"
    sample_contract_id: str = "auto"
    parameter_policy_id: str = "auto"
    input_contract_id: str = "auto"
    training_lifecycle: str = "none"         # none / rolling_local / artifact_walk_forward
    artifact_required: bool = False
    stateful: bool = False
    checkpoint_supported: bool = False
    cost_class: str = "medium"               # low / medium / high / very_high
    default_searchable: bool = True

    @property
    def is_predictive_supervised(self) -> bool:
        return self.execution_class == ModelExecutionClass.PREDICTIVE_SUPERVISED


def validate_model_operator_spec(spec: ModelOperatorSpec) -> list[str]:
    """Validate a :class:`ModelOperatorSpec` (used by the registry gate)."""
    problems: list[str] = []
    if spec.execution_class == ModelExecutionClass.PREDICTIVE_SUPERVISED:
        if not spec.artifact_required:
            problems.append(
                f"{spec.canonical}: PREDICTIVE_SUPERVISED must be artifact_required"
            )
        if spec.training_lifecycle != "artifact_walk_forward":
            problems.append(
                f"{spec.canonical}: PREDICTIVE_SUPERVISED must use "
                "training_lifecycle='artifact_walk_forward'"
            )
        if spec.default_searchable:
            problems.append(
                f"{spec.canonical}: PREDICTIVE_SUPERVISED must be default_searchable=False "
                "(search is via validated SearchPolicy only)"
            )
    return problems


# --------------------------------------------------------------------------- #
# 10. Legacy local predictive variant (§2.1 / §67).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegacyLocalPredictive:
    """Marker for the retained short-window local rolling variants (§2.1).

    The legacy operators are NOT deleted (DSL / historical factor identity is
    preserved); they are classified as ``LOCAL_ROLLING_ESTIMATOR``,
    ``default_searchable=False`` and ``research_only=True``.  ``ts_local_*``
    canonicals are the suggested future rename targets; old names stay as
    compat aliases.  §67 raises their local sample floors.
    """

    canonical: str
    legacy_label: str = "LEGACY_LOCAL_ROLLING"
    execution_class: ModelExecutionClass = ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR
    default_searchable: bool = False
    research_only: bool = True
    # §67 research-safety floors (min effective obs per free parameter).
    min_effective_obs: int | None = None
    min_obs_per_parameter: float = 10.0
    min_regime_obs: int | None = None
    min_expert_obs: int | None = None

    def floor_for(self, free_parameters: int) -> int | None:
        if self.min_effective_obs is not None:
            return self.min_effective_obs
        if free_parameters > 0:
            return max(60, int(self.min_obs_per_parameter * free_parameters))
        return None
