"""
Named metric preset bundles + versioned EvidenceProfile registry.

Part 1 (historical, unchanged): ``MetricPreset`` — the legacy named id-list
bundles (``factor_core`` / ``factor_extended`` / ``production_daily``).  Kept
byte-compatible: same class, same functions, same three registrations.

Part 2 (R61-FI-020 / plan section 12 workstream D): ``EvidenceProfile`` — the
QE-side named metric-binding authority.  A profile is a FROZEN dataclass that
binds a set of existing ``MetricRegistry`` metric ids (never invented ids),
declares the required inputs those metrics need, a cost class, an explicit
``target`` market/frequency annotation (``CN`` / ``1d``), and a GPU
preference (vocabulary: ``auto`` / ``cpu_reference`` / ``cuda``).  Profiles
are versioned via ``policy_id`` + ``policy_version`` (``SemVer``-formatted
string) and resolved through a fail-closed registry: unknown profile ids and
unknown metric ids raise typed errors — a profile may never reference a
metric that does not exist in the single ``MetricRegistry`` authority
(``quant_evaluator.registry.metrics``), which is exactly what guarantees
profiles cannot fabricate metrics.

R61-FI-020 scope note: this module adds bindings and naming ONLY.  It does
not add, remove, or change a single metric computation or kernel.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Set, Tuple

from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.registry.metrics import MetricSpec, get_metric


class MetricPreset:
    """
    Named collection of metrics for a specific evaluation workflow.

    Attributes:
        name: Preset identifier
        display_name: Human-readable name
        description: Workflow description
        metric_names: List of metric names in this preset
    """
    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        metric_names: List[str],
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.metric_names = metric_names

    def get_metrics(self) -> List[MetricSpec]:
        """
        Retrieve all metric specifications in this preset.

        Returns:
            List of MetricSpec objects

        Raises:
            KeyError: If any metric name is not registered
        """
        return [get_metric(name) for name in self.metric_names]

    def __repr__(self) -> str:
        return (
            f"MetricPreset(name={self.name!r}, "
            f"display_name={self.display_name!r}, "
            f"metrics={len(self.metric_names)})"
        )


# Preset catalog
_PRESET_CATALOG: Dict[str, MetricPreset] = {}


def register_preset(preset: MetricPreset) -> None:
    """
    Register a metric preset.

    Args:
        preset: MetricPreset to register

    Raises:
        ValueError: If preset name already registered
    """
    if preset.name in _PRESET_CATALOG:
        raise ValueError(f"Preset '{preset.name}' already registered")
    _PRESET_CATALOG[preset.name] = preset


def get_preset(name: str) -> MetricPreset:
    """
    Retrieve metric preset by name.

    Args:
        name: Preset identifier

    Returns:
        MetricPreset for the requested preset

    Raises:
        KeyError: If preset not found
    """
    if name not in _PRESET_CATALOG:
        raise KeyError(f"Preset '{name}' not found in registry")
    return _PRESET_CATALOG[name]


def list_presets() -> List[str]:
    """
    List all registered preset names.

    Returns:
        Sorted list of preset names
    """
    return sorted(_PRESET_CATALOG.keys())


# Define standard presets
FACTOR_CORE = MetricPreset(
    name="factor_core",
    display_name="Factor Core Metrics",
    description="Essential metrics for daily factor evaluation",
    metric_names=[
        "mean_ic",
        "ic_std",
        "ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
    ],
)

FACTOR_EXTENDED = MetricPreset(
    name="factor_extended",
    display_name="Factor Extended Metrics",
    description="Comprehensive factor evaluation including robustness and temporal properties",
    metric_names=[
        # Core metrics
        "mean_ic",
        "ic_std",
        "ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
        # Extended metrics
        "hac_tstat",
        "subsample_stability",
        "ic_autocorr_lag1",
        "rank_stability",
        "half_life",
    ],
)

PRODUCTION_DAILY = MetricPreset(
    name="production_daily",
    display_name="Production Daily Report",
    description="Fast metrics for daily production monitoring",
    metric_names=[
        "mean_ic",
        "ic_ir",
        "coverage",
        "turnover",
    ],
)

# Register all presets
register_preset(FACTOR_CORE)
register_preset(FACTOR_EXTENDED)
register_preset(PRODUCTION_DAILY)


# ===========================================================================
# Part 2 (R61-FI-020): versioned EvidenceProfile registry.
# ===========================================================================


class CostClass(Enum):
    """Computational cost tier of an EvidenceProfile.

    Ordering is meaningful (``CHEAP < MODERATE < EXPENSIVE``) so profiles
    can be compared / tiered.  Declared here because no cost-class
    vocabulary existed in the package before R61-FI-020.
    """

    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"

    def __lt__(self, other: "CostClass") -> bool:
        """Compare cost tiers by the declared member order."""
        if not isinstance(other, CostClass):
            return NotImplemented
        order = tuple(member for member in CostClass)
        return order.index(self) < order.index(other)

    @classmethod
    def from_value(cls, value: object) -> "CostClass":
        """Resolve a str/enum value to a CostClass (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown CostClass value {value!r}; expected one of "
                    f"{[member.value for member in cls]}"
                )
        raise TypeError(
            f"CostClass.from_value expects str or CostClass, got "
            f"{type(value).__name__}"
        )


class GPUPreference(Enum):
    """GPU execution preference for a profile's metric set.

    Vocabulary reuses the existing ``BackendPolicy`` semantics
    (``quant_evaluator.contracts.backend_policy``): ``auto`` (runtime
    decides), ``cpu_reference`` (reference kernels), ``cuda`` (GPU when
    available, never silently wrong on CPU).
    """

    AUTO = "auto"
    CPU_REFERENCE = "cpu_reference"
    CUDA = "cuda"

    @classmethod
    def from_value(cls, value: object) -> "GPUPreference":
        """Resolve a str/enum value to a GPUPreference (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown GPUPreference value {value!r}; expected one of "
                    f"{[member.value for member in cls]}"
                )
        raise TypeError(
            f"GPUPreference.from_value expects str or GPUPreference, got "
            f"{type(value).__name__}"
        )


class TargetFrequency(str, Enum):
    """Market/frequency annotation vocabulary (plan section 12)."""

    CN_1D = "cn_1d"  # China A-share daily cross-section

    @classmethod
    def from_value(cls, value: object) -> "TargetFrequency":
        """Resolve a str/enum value to a TargetFrequency (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown TargetFrequency value {value!r}; expected one of "
                    f"{[member.value for member in cls]}"
                )
        raise TypeError(
            f"TargetFrequency.from_value expects str or TargetFrequency, got "
            f"{type(value).__name__}"
        )


@dataclass(frozen=True)
class EvidenceProfile:
    """Versioned named binding of MetricRegistry metric ids (R61-FI-020).

    Attributes:
        profile_id: Unique profile identifier (e.g. ``CHEAP_SCREEN_CN_1D``).
        policy_id: Stable policy family identifier the version belongs to
            (e.g. ``QE_EVIDENCE_PROFILE``).  Distinct from ``profile_id`` so
            that a versioned policy family is queryable independently.
        policy_version: Version of this profile binding (SemVer-formatted
            string, e.g. ``"1.0.0"``).  Required non-empty.
        display_name: Human-readable name.
        description: Workflow description.
        metric_ids: Tuple of registered MetricRegistry metric ids bound to
            this profile.  Frozen and validated in ``__post_init__``: every
            id MUST resolve in the single MetricRegistry authority or the
            profile fails closed with ``UnsupportedMetricError`` (this is the
            anti-fabrication guarantee).
        required_inputs: Union of the artifact inputs the bound metrics need
            (``"factor_batch"`` / ``"label_bundle"`` / ``"ICSeriesArtifact"``
            / ``"QuantileReturnArtifact"`` / ``"p_values"``).  Frozen set.
        cost_class: :class:`CostClass` tier (CHEAP/MODERATE/EXPENSIVE).
        target: :class:`TargetFrequency` market/frequency annotation.
        gpu_preference: :class:`GPUPreference` for the metric set.
        parallelizable: Whether the bound metrics may be computed
            concurrently (True for all current profiles; each metric is an
            independent kernel).
    """

    profile_id: str
    metric_ids: Tuple[str, ...]
    required_inputs: FrozenSet[str]
    cost_class: CostClass
    target: TargetFrequency
    display_name: str = ""
    description: str = ""
    policy_id: str = "QE_EVIDENCE_PROFILE"
    policy_version: str = "1.0.0"
    gpu_preference: GPUPreference = GPUPreference.AUTO
    parallelizable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("EvidenceProfile.profile_id must be a non-empty string")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("EvidenceProfile.policy_id must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError(
                "EvidenceProfile.policy_version must be a non-empty string"
            )
        # Canonical target values are lower-case tokens; profile ids are
        # upper-case tokens by convention.
        if not self.display_name:
            object.__setattr__(self, "display_name", self.profile_id)
        # Normalize enums (accept plain strings too) -- fail closed.
        object.__setattr__(self, "cost_class", CostClass.from_value(self.cost_class))
        object.__setattr__(self, "target", TargetFrequency.from_value(self.target))
        object.__setattr__(
            self, "gpu_preference", GPUPreference.from_value(self.gpu_preference)
        )
        metric_tuple = tuple(self.metric_ids)
        object.__setattr__(self, "metric_ids", metric_tuple)
        if not metric_tuple:
            raise ValueError(
                f"EvidenceProfile {self.profile_id!r} binds no metrics; a profile "
                "with zero metrics is a capability lie"
            )
        # De-duplicate without changing order (a profile binds a SET of
        # metrics; duplicates are a definition bug).
        seen: List[str] = []
        for metric_id in metric_tuple:
            if not isinstance(metric_id, str) or not metric_id.strip():
                raise ValueError(
                    f"EvidenceProfile {self.profile_id!r} has a non-string "
                    "metric id in its binding"
                )
            if metric_id in seen:
                raise ValueError(
                    f"EvidenceProfile {self.profile_id!r} binds metric "
                    f"{metric_id!r} more than once"
                )
            seen.append(metric_id)
        object.__setattr__(self, "metric_ids", tuple(seen))
        object.__setattr__(self, "required_inputs", frozenset(self.required_inputs))
        # Anti-fabrication: every bound metric id must EXIST in the single
        # MetricRegistry authority (quant_evaluator.registry.metrics).  An
        # unknown id fails closed with the typed UnsupportedMetricError.
        for metric_id in self.metric_ids:
            if metric_id not in _metric_registry_ids():
                raise UnsupportedMetricError(
                    f"EvidenceProfile {self.profile_id!r} binds metric "
                    f"{metric_id!r} which is NOT registered in the "
                    "MetricRegistry authority (quant_evaluator.registry."
                    "metrics); profiles may only bind existing metrics"
                )

    def metric_specs(self) -> List[MetricSpec]:
        """Resolve the bound metric ids to their MetricSpec objects."""
        return [get_metric(metric_id) for metric_id in self.metric_ids]

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "profile_id": self.profile_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "display_name": self.display_name,
            "description": self.description,
            "metric_ids": list(self.metric_ids),
            "required_inputs": sorted(self.required_inputs),
            "cost_class": self.cost_class.value,
            "target": self.target.value,
            "gpu_preference": self.gpu_preference.value,
            "parallelizable": self.parallelizable,
        }


def _metric_registry_ids() -> Set[str]:
    """Read-only snapshot of the single MetricRegistry authority's ids."""
    from quant_evaluator.registry.metrics import list_metrics

    return set(list_metrics())


# Profile registry (versioned by policy_id + policy_version at the module
# level; the profile registry itself is a plain mapping, the versioning
# contract lives on each frozen EvidenceProfile).
_PROFILE_CATALOG: Dict[str, EvidenceProfile] = {}


class UnknownEvidenceProfileError(KeyError):
    """Raised when an EvidenceProfile id is not registered.

    Inherits from ``KeyError`` (the historical ``get_preset`` contract) and
    is the typed error callers can pattern-match on.
    """

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(
            f"Unknown EvidenceProfile {profile_id!r}; registered profiles: "
            f"{sorted(_PROFILE_CATALOG)}"
        )


def register_profile(profile: EvidenceProfile) -> None:
    """Register an EvidenceProfile (duplicate profile_id fails closed)."""
    if not isinstance(profile, EvidenceProfile):
        raise TypeError(
            f"register_profile expects EvidenceProfile, got {type(profile).__name__}"
        )
    if profile.profile_id in _PROFILE_CATALOG:
        raise ValueError(
            f"EvidenceProfile {profile.profile_id!r} already registered; "
            "no silent overwrite"
        )
    _PROFILE_CATALOG[profile.profile_id] = profile


def get_profile(profile_id: str) -> EvidenceProfile:
    """Resolve a profile id to its frozen EvidenceProfile (typed error)."""
    try:
        return _PROFILE_CATALOG[profile_id]
    except KeyError:
        raise UnknownEvidenceProfileError(profile_id) from None


def get_profile_or_none(profile_id: str) -> EvidenceProfile | None:
    """Resolve a profile id, returning None when it is not registered."""
    return _PROFILE_CATALOG.get(profile_id)


def list_profiles() -> List[str]:
    """Return a sorted list of all registered EvidenceProfile ids."""
    return sorted(_PROFILE_CATALOG.keys())


def list_profile_versions() -> Dict[str, str]:
    """Return profile_id -> policy_version for every registered profile."""
    return {
        profile_id: profile.policy_version
        for profile_id, profile in sorted(_PROFILE_CATALOG.items())
    }


# ---------------------------------------------------------------------------
# The five named R61 EvidenceProfiles (CN A-share daily cross-section).
#
# Binding rules (enforced by EvidenceProfile.__post_init__):
#   * every metric id below resolves in the 105-id MetricRegistry;
#   * every metric id has a bound compute_fn (verified 2026-09-04: no
#     bound id has compute_fn=None);
#   * no profile invents a metric id, and no metric computation is added or
#     changed by this task (bindings/naming only).
# ---------------------------------------------------------------------------

# Cost class CHEAP — per-factor screening over an (T, N, F) daily panel:
# IC family (rank_ic / pearson_ic / ic_ir / ic_median / ic_std /
# rank_ic_series / pearson_ic_series / pearson_ic_std / pearson_ic_ir),
# coverage, turnover, quantile spread, plus DATA_QUALITY cheap scans
# (missing_ratio / staleness / outlier_ratio / effective_n /
# distinct_level_ratio / tie_ratio / label_maturity) and the
# RESEARCH_INTEGRITY multiple-testing corrections (cheap elementwise p-value
# transforms).  This is the daily-screening default: O(#metrics) independent
# kernels, all CUDA-parity families in R60.
CHEAP_SCREEN_CN_1D = EvidenceProfile(
    profile_id="CHEAP_SCREEN_CN_1D",
    display_name="Cheap Screen (CN Daily)",
    description=(
        "Per-factor daily screening bundle: core IC family (rank/pearson mean, "
        "ir, median, std, daily series), coverage, turnover, quantile spread, "
        "data-quality cheap scans and multiple-testing corrections. Intended "
        "for first-pass factor screening over large factor batches; all bound "
        "metrics are cheap elementwise/rank kernels."
    ),
    metric_ids=(
        "rank_ic",
        "pearson_ic",
        "ic_ir",
        "ic_median",
        "ic_std",
        "rank_ic_series",
        "pearson_ic_series",
        "pearson_ic_std",
        "pearson_ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
        "missing_ratio",
        "staleness",
        "outlier_ratio",
        "effective_n",
        "distinct_level_ratio",
        "tie_ratio",
        "label_maturity",
        "bonferroni_correction",
        "benjamini_hochberg_correction",
        "holm_bonferroni_correction",
        "sidak_correction",
    ),
    required_inputs=frozenset(
        {"factor_batch", "label_bundle", "ICSeriesArtifact", "p_values"}
    ),
    cost_class=CostClass.CHEAP,
    target=TargetFrequency.CN_1D,
    policy_version="1.0.0",
    gpu_preference=GPUPreference.AUTO,
    parallelizable=True,
)

# Cost class CHEAP — quantile-shape diagnostics.  The quantile-shape family
# consumes a QuantileReturnArtifact (computed once from quantile_returns_full)
# and adds the shape-family kernels (monotonicity / curvature / tail
# asymmetry / adjacent spread / extreme cliff / top+bottom cliff).  All bound
# ids exist and are CPU + GPU parity kernels (R60).
SHAPE_DIAGNOSTIC_CN_1D = EvidenceProfile(
    profile_id="SHAPE_DIAGNOSTIC_CN_1D",
    display_name="Shape Diagnostic (CN Daily)",
    description=(
        "Quantile-profile shape diagnostics: full quantile returns plus the "
        "quantile-shape family (monotonicity, curvature, tail asymmetry, "
        "adjacent spread, extreme cliff, top/bottom cliff). For U-shape / "
        "inverted-U / cliff / monotonicity checks on the CN daily quantile "
        "profile."
    ),
    metric_ids=(
        "quantile_returns_full",
        "quantile_monotonicity",
        "quantile_curvature",
        "quantile_tail_asymmetry",
        "quantile_adjacent_spread",
        "quantile_extreme_cliff",
        "top_quantile_cliff",
        "bottom_quantile_cliff",
    ),
    required_inputs=frozenset({"factor_batch", "label_bundle", "QuantileReturnArtifact"}),
    cost_class=CostClass.CHEAP,
    target=TargetFrequency.CN_1D,
    policy_version="1.0.0",
    gpu_preference=GPUPreference.AUTO,
    parallelizable=True,
)

# Cost class MODERATE — the full validation spectrum for a factor that has
# passed screening.  Superset of CHEAP_SCREEN + SHAPE_DIAGNOSTIC, plus the
# robustness/stability/temporal extras that are still moderate-cost:
# long_short_returns + sharpe/sortino/win_rate (return-based portfolio
# stats), rank_stability, factor_turnover_rate, missing_timeline,
# universe_churn.  The containment CHEAP_SCREEN ids SUBSET FULL_VALIDATION
# ids is asserted in tests/test_evidence_profiles.py (monotonicity of the
# evaluation ladder).
FULL_VALIDATION_CN_1D = EvidenceProfile(
    profile_id="FULL_VALIDATION_CN_1D",
    display_name="Full Validation (CN Daily)",
    description=(
        "Full-validation spectrum for a factor that passed screening: every "
        "CHEAP_SCREEN and SHAPE_DIAGNOSTIC metric plus portfolio-stat "
        "long/short (returns, sharpe, sortino, win rate), rank stability, "
        "factor turnover rate, missing timeline and universe churn. "
        "Superset of CHEAP_SCREEN_CN_1D and SHAPE_DIAGNOSTIC_CN_1D."
    ),
    metric_ids=(
        # CHEAP_SCREEN block (kept in the same order for auditability).
        "rank_ic",
        "pearson_ic",
        "ic_ir",
        "ic_median",
        "ic_std",
        "rank_ic_series",
        "pearson_ic_series",
        "pearson_ic_std",
        "pearson_ic_ir",
        "coverage",
        "turnover",
        "quantile_spread",
        "missing_ratio",
        "staleness",
        "outlier_ratio",
        "effective_n",
        "distinct_level_ratio",
        "tie_ratio",
        "label_maturity",
        "bonferroni_correction",
        "benjamini_hochberg_correction",
        "holm_bonferroni_correction",
        "sidak_correction",
        # SHAPE_DIAGNOSTIC block.
        "quantile_returns_full",
        "quantile_monotonicity",
        "quantile_curvature",
        "quantile_tail_asymmetry",
        "quantile_adjacent_spread",
        "quantile_extreme_cliff",
        "top_quantile_cliff",
        "bottom_quantile_cliff",
        # MODERATE extras (portfolio stats + stability/temporal).
        "long_short_returns",
        "sharpe_ratio",
        "sortino_ratio",
        "win_rate",
        "rank_stability",
        "factor_turnover_rate",
        "missing_timeline",
        "universe_churn",
    ),
    required_inputs=frozenset(
        {
            "factor_batch",
            "label_bundle",
            "ICSeriesArtifact",
            "QuantileReturnArtifact",
            "p_values",
        }
    ),
    cost_class=CostClass.MODERATE,
    target=TargetFrequency.CN_1D,
    policy_version="1.0.0",
    gpu_preference=GPUPreference.AUTO,
    parallelizable=True,
)

# Cost class EXPENSIVE — statistical-confidence family.  HAC t-stat/p-value
# (serial autocorrelation-aware), block-bootstrap CI, subsample stability and
# IC autocorrelation lag-1 all need the full IC history and are the
# expensive end of per-factor statistical inference in the current 105-id
# registry.  (DSR/PBO/SPA/White-reality-check do NOT exist as registered
# metrics — they are declared NOT_IMPLEMENTED in the capability matrix, and
# this profile refuses to bind ids that do not exist.)
EXPENSIVE_STATISTICAL_CN_1D = EvidenceProfile(
    profile_id="EXPENSIVE_STATISTICAL_CN_1D",
    display_name="Expensive Statistical (CN Daily)",
    description=(
        "Statistical-confidence bundle for final factor validation: HAC "
        "t-stat and p-value, block-bootstrap 95% CI, subsample stability and "
        "IC lag-1 autocorrelation. Consumes the full IC series history. "
        "DSR/PBO/SPA/White are NOT registered metrics in this release and are "
        "honestly absent (no fake stubs)."
    ),
    metric_ids=(
        "hac_tstat",
        "hac_pvalue",
        "block_bootstrap_ci",
        "subsample_stability",
        "ic_autocorr_lag1",
    ),
    required_inputs=frozenset({"ICSeriesArtifact"}),
    cost_class=CostClass.EXPENSIVE,
    target=TargetFrequency.CN_1D,
    policy_version="1.0.0",
    gpu_preference=GPUPreference.CUDA,
    parallelizable=True,
)

# Cost class MODERATE — model-feature diagnostics.  A model consumes a
# factor as a feature and needs predictive-consistency + regime evidence:
# the PREDICTIVE family (positive ratios, yearly/monthly/quarterly mean rank
# IC, rolling mean/IR, recent 3/6/12m, worst year/quarter, decay, sign
# consistency, recent-vs-history delta) and the REGIME family (year/quarter/
# month consistency, rolling volatility/drawdown, sign flip rate,
# change-point/cusum, recent degradation, regime conditional/worst/dispersion/
# sign consistency).  All bound ids exist (EXPERIMENTAL status, compute_fn
# bound); model training itself is out of QE scope.
MODEL_FEATURE_DIAGNOSTIC_CN_1D = EvidenceProfile(
    profile_id="MODEL_FEATURE_DIAGNOSTIC_CN_1D",
    display_name="Model Feature Diagnostic (CN Daily)",
    description=(
        "Predictive-consistency and regime diagnostics for factors consumed "
        "as model features: IC positive ratios, multi-window mean rank IC "
        "(yearly/monthly/quarterly/rolling/recent), worst-period rank IC, "
        "decay, sign consistency, recent-vs-history delta, plus the regime "
        "family (consistencies, rolling vol/drawdown, sign-flip rate, "
        "change-point/cusum, degradation, regime conditional/worst/"
        "dispersion/sign)."
    ),
    metric_ids=(
        "ic_positive_ratio",
        "rank_ic_positive_ratio",
        "yearly_rank_ic",
        "monthly_rank_ic",
        "quarterly_rank_ic",
        "rolling_rank_ic_mean",
        "rolling_rank_ic_ir",
        "recent_3m_rank_ic",
        "recent_6m_rank_ic",
        "recent_12m_rank_ic",
        "worst_year_rank_ic",
        "worst_quarter_rank_ic",
        "rank_ic_decay_h01_h05_h10_h20",
        "ic_sign_consistency",
        "ic_recent_vs_history_delta",
        "year_consistency",
        "quarter_consistency",
        "month_consistency",
        "rolling_ic_volatility",
        "rolling_ic_drawdown",
        "ic_sign_flip_rate",
        "change_point_score",
        "cusum_break_score",
        "recent_degradation_score",
        "regime_conditional_ic",
        "regime_worst_ic",
        "regime_dispersion",
        "regime_sign_consistency",
    ),
    required_inputs=frozenset({"ICSeriesArtifact"}),
    cost_class=CostClass.MODERATE,
    target=TargetFrequency.CN_1D,
    policy_version="1.0.0",
    gpu_preference=GPUPreference.CUDA,
    parallelizable=True,
)


# Register all five EvidenceProfiles.
register_profile(CHEAP_SCREEN_CN_1D)
register_profile(SHAPE_DIAGNOSTIC_CN_1D)
register_profile(FULL_VALIDATION_CN_1D)
register_profile(EXPENSIVE_STATISTICAL_CN_1D)
register_profile(MODEL_FEATURE_DIAGNOSTIC_CN_1D)
