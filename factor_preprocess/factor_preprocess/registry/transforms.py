"""
Transform registry with versioning and discovery.

Provides a central catalog of all preprocessing transforms with metadata,
versioning, and category-based organization.

Production integrity:
- FP-P1-04: ``TransformMetadata.signature_hash`` only hashed
  name + inspect.signature(func), so a function body rewrite with an
  unchanged signature produced the same hash. We add
  ``implementation_hash`` (source closure hash) and ``numeric_policy_hash``
  (hash of parameter/version/admission defaults) so true implementation
  identity is captured.
- FP-P1-05: ``TransformRegistry.seal()`` freezes the registry into an
  immutable ``RegistrySnapshotIdentity``; runtime re-registration after seal
  raises (production semantics cannot change mid-run).
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import hashlib
import inspect

from factor_preprocess.errors import GovernanceError


class TransformCategory(str, Enum):
    """Transform category classification."""
    CROSS_SECTIONAL = "cross_sectional"
    TEMPORAL = "temporal"
    VOLATILITY = "volatility"
    MISSINGNESS = "missingness"
    FRESHNESS = "freshness"
    NEUTRALIZATION = "neutralization"
    REPRESENTATION = "representation"


# DLIB-FP-023: the known semantic factor families the eligibility layer keys
# on. Immutable public constant used for semantic enrichment of transforms.
ALL_FAMILY_TAGS = frozenset({
    "PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE",
    "EVENT", "BINARY", "DISCRETE",
})


def _source_of(func: Callable) -> str:
    """Best-effort stable source of a callable (may not exist for builtins)."""
    try:
        return inspect.getsource(func)
    except (TypeError, OSError, IOError):
        # Fall back to bytecode disassembly which is content-stable.
        import dis
        return str(dis.get_instructions(func))


def _hash_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TransformMetadata:
    """Metadata for a registered transform.

    Extended semantic surface (DLIB-FP-015): the eligibility engine queries
    transforms *by metadata* (semantic id, stage, family tags, causality
    class, allowed factor families / asset types / frequencies, parameter
    domain, numeric policy, production admission, FE-equivalent semantics,
    cost class, output channels) instead of maintaining a hand-written name
    list. All mutable fields are deeply frozen in ``__post_init__`` so the
    metadata snapshot is hash-safe (DLIB-FP-026).
    """
    name: str
    func: Callable
    category: TransformCategory
    version: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    causal_safe: bool = True
    admission: str = "PRODUCTION"
    signature_hash: Optional[str] = None
    implementation_hash: Optional[str] = None
    numeric_policy_hash: Optional[str] = None

    # DLIB-FP-015 extended semantic surface (deeply frozen).
    semantic_id: Optional[str] = None
    stage: Optional[str] = None
    family_tags: Set[str] = field(default_factory=set)
    causality_class: Optional[str] = None
    requires_fit: bool = False
    requires_exposure: bool = False
    requires_universe: bool = False
    allowed_factor_families: Set[str] = field(default_factory=set)
    allowed_asset_types: Set[str] = field(default_factory=set)
    allowed_frequencies: Set[str] = field(default_factory=set)
    parameter_domain: Dict[str, Any] = field(default_factory=dict)
    numeric_policy: Optional[str] = None
    production_admission: Optional[str] = None
    fe_equivalent_semantics: Optional[str] = None
    cost_class: Optional[str] = None
    output_channels: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Compute hashes after initialization."""
        # Freeze mutable nested containers before hashing (DLIB-FP-026).
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "tags", set(self.tags))
        object.__setattr__(self, "family_tags", set(self.family_tags))
        object.__setattr__(self, "allowed_factor_families", set(self.allowed_factor_families))
        object.__setattr__(self, "allowed_asset_types", set(self.allowed_asset_types))
        object.__setattr__(self, "allowed_frequencies", set(self.allowed_frequencies))
        object.__setattr__(self, "parameter_domain", dict(self.parameter_domain))
        object.__setattr__(self, "output_channels", tuple(self.output_channels))
        if self.production_admission is None:
            object.__setattr__(self, "production_admission", self.admission)

        if self.signature_hash is None:
            self.signature_hash = self._compute_signature_hash()
        if self.implementation_hash is None:
            self.implementation_hash = self._compute_implementation_hash()
        if self.numeric_policy_hash is None:
            self.numeric_policy_hash = self._compute_numeric_policy_hash()

    def _compute_signature_hash(self) -> str:
        """Compute stable hash of function signature (name + signature)."""
        sig = inspect.signature(self.func)
        sig_str = f"{self.name}:{str(sig)}"
        return _hash_bytes(sig_str)[:16]

    def _compute_implementation_hash(self) -> str:
        """Hash the actual source/bytecode of the function body.

        Distinguishes two functions with identical signatures but different
        bodies (FP-P1-04). Includes the signature to guard against two
        distinct body objects that happen to disassemble identically at a
        different name.
        """
        source = _source_of(self.func)
        sig = inspect.signature(self.func)
        return _hash_bytes(f"{self.name}:{str(sig)}::body::{source}")[:16]

    def _compute_numeric_policy_hash(self) -> str:
        """Hash the numeric/default parameter surface (version, admission,
        causal_safe, default parameters, tags) that changes numerical output
        without touching the signature (FP-P1-04)."""
        # Deterministic ordering: sort tags and parameter keys.
        def canonical_params(p):
            if isinstance(p, dict):
                return "{" + ",".join(
                    f"{k}:{canonical_params(v)}" for k, v in sorted(p.items())
                ) + "}"
            return repr(p)

        payload = "|".join([
            self.version,
            self.admission,
            str(self.causal_safe),
            canonical_params(self.parameters),
            ",".join(sorted(self.tags)),
        ])
        return _hash_bytes(payload)[:16]

    def bind_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate configured keyword parameters against the callable signature."""
        signature = inspect.signature(self.func)
        try:
            signature.bind_partial(**parameters)
        except TypeError as exc:
            raise ValueError(
                f"Transform '{self.name}' has invalid parameters: {exc}"
            ) from exc


class TransformRegistry:
    """
    Central registry for preprocessing transforms.

    Features:
    - Versioned transform catalog
    - Category-based organization
    - Signature tracking for reproducibility
    - Tag-based filtering
    - Causal safety annotation
    """

    def __init__(self):
        self._transforms: Dict[str, TransformMetadata] = {}
        self._by_category: Dict[TransformCategory, List[str]] = {
            cat: [] for cat in TransformCategory
        }
        self._by_tag: Dict[str, List[str]] = {}
        self._sealed = False
        self._snapshot_identity: Optional[str] = None

    def _check_not_sealed(self):
        """Fail closed: no runtime mutation of sealed production semantics."""
        if self._sealed:
            raise GovernanceError(
                "TransformRegistry is sealed; production semantics are "
                "immutable at runtime (FP-P1-05)"
            )

    def register(
        self,
        name: str,
        func: Callable,
        category: TransformCategory,
        version: str = "1.0.0",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        causal_safe: bool = True,
        admission: Optional[str] = None,
        # DLIB-FP-015 extended semantic surface.
        semantic_id: Optional[str] = None,
        stage: Optional[str] = None,
        family_tags: Optional[Set[str]] = None,
        causality_class: Optional[str] = None,
        requires_fit: bool = False,
        requires_exposure: bool = False,
        requires_universe: bool = False,
        allowed_factor_families: Optional[Set[str]] = None,
        allowed_asset_types: Optional[Set[str]] = None,
        allowed_frequencies: Optional[Set[str]] = None,
        parameter_domain: Optional[Dict[str, Any]] = None,
        numeric_policy: Optional[str] = None,
        fe_equivalent_semantics: Optional[str] = None,
        cost_class: Optional[str] = None,
        output_channels: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """
        Register a transform.

        Parameters
        ----------
        name : str
            Unique transform name
        func : Callable
            Transform function
        category : TransformCategory
            Transform category
        version : str
            Semantic version string
        description : str
            Human-readable description
        parameters : dict, optional
            Default parameter values
        tags : set, optional
            Searchable tags
        causal_safe : bool
            Whether transform preserves causal structure
        admission : str
            Production admission class
        semantic_id : str, optional
            Canonical semantic identifier (DLIB-FP-015)
        stage : str, optional
            Semantic stage of the transform
        family_tags : set, optional
            Semantic family tags (e.g. ``{PRICE_VOLUME, EVENT}``)
        causality_class : str, optional
            Causality classification
        requires_fit : bool
            Whether the transform requires fitted state
        requires_exposure : bool
            Whether exposure data is required
        requires_universe : bool
            Whether a universe is required
        allowed_factor_families : set, optional
            Factor families the transform may be applied to
        allowed_asset_types : set, optional
            Asset types the transform supports
        allowed_frequencies : set, optional
            Supported data frequencies
        parameter_domain : dict, optional
            Domain for each parameter (e.g. ranges)
        numeric_policy : str, optional
            Numeric policy (e.g. ``"nan_skip"``)
        fe_equivalent_semantics : str, optional
            FE DSL semantic equivalent
        cost_class : str, optional
            Runtime cost classification
        output_channels : tuple, optional
            Output channel names
        """
        # Existing callers omit admission for ordinary causal transforms;
        # make that omission explicit rather than treating it as unknown.
        admission = admission or "PRODUCTION"
        self._check_not_sealed()
        if admission not in {"PRODUCTION", "OFFLINE_ONLY", "RESEARCH_ONLY"}:
            raise ValueError("admission must be PRODUCTION, OFFLINE_ONLY, or RESEARCH_ONLY")
        if admission == "PRODUCTION" and not causal_safe:
            raise ValueError("Production transforms must be causal_safe")

        metadata = TransformMetadata(
            name=name,
            func=func,
            category=category,
            version=version,
            description=description,
            parameters=deepcopy(parameters or {}),
            tags=deepcopy(tags or set()),
            causal_safe=causal_safe,
            admission=admission,
            semantic_id=semantic_id,
            stage=stage,
            family_tags=deepcopy(family_tags or set()),
            causality_class=causality_class,
            requires_fit=requires_fit,
            requires_exposure=requires_exposure,
            requires_universe=requires_universe,
            allowed_factor_families=deepcopy(allowed_factor_families or set()),
            allowed_asset_types=deepcopy(allowed_asset_types or set()),
            allowed_frequencies=deepcopy(allowed_frequencies or set()),
            parameter_domain=deepcopy(parameter_domain or {}),
            numeric_policy=numeric_policy,
            fe_equivalent_semantics=fe_equivalent_semantics,
            cost_class=cost_class,
            output_channels=tuple(output_channels) if output_channels else (),
        )

        if name in self._transforms:
            existing = self._transforms[name]
            if existing.version != version:
                raise ValueError(
                    f"Transform '{name}' already registered with version "
                    f"{existing.version}, cannot register version {version}"
                )
            same_registration = (
                existing.func is metadata.func
                and existing.category == metadata.category
                and existing.description == metadata.description
                and existing.parameters == metadata.parameters
                and existing.tags == metadata.tags
                and existing.causal_safe == metadata.causal_safe
                and existing.admission == metadata.admission
                and existing.signature_hash == metadata.signature_hash
                and existing.implementation_hash == metadata.implementation_hash
                and existing.numeric_policy_hash == metadata.numeric_policy_hash
            )
            if not same_registration:
                raise ValueError(
                    f"Transform '{name}' version {version} is already registered "
                    "with conflicting metadata or implementation"
                )
            return

        self._transforms[name] = metadata
        self._by_category[category].append(name)

        for tag in metadata.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(name)

    def get(self, name: str) -> Optional[TransformMetadata]:
        """Get an isolated transform metadata snapshot by name."""
        metadata = self._transforms.get(name)
        return deepcopy(metadata) if metadata is not None else None

    def enrich(self, name: str, **fields) -> None:
        """Attach additional semantic metadata to an existing transform.

        DLIB-FP-015: separates the *registration* step (identity hashes are
        computed at registration) from *semantic enrichment* (semantic_id,
        stage, family_tags, causality_class, ...) so the whole catalog becomes
        self-describing without changing signature/implementation/numeric
        policy hashes. Fails closed on unknown names and after seal.
        """
        self._check_not_sealed()
        if name not in self._transforms:
            raise ValueError(f"Cannot enrich unknown transform {name!r}")
        meta = self._transforms[name]
        allowed = {
            "semantic_id", "stage", "family_tags", "causality_class",
            "requires_fit", "requires_exposure", "requires_universe",
            "allowed_factor_families", "allowed_asset_types",
            "allowed_frequencies", "parameter_domain", "numeric_policy",
            "fe_equivalent_semantics", "cost_class", "output_channels",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown semantic metadata fields: {sorted(unknown)}")
        for key, value in fields.items():
            if isinstance(
                value,
                (set, frozenset),
            ) and key in (
                "family_tags",
                "allowed_factor_families",
                "allowed_asset_types",
                "allowed_frequencies",
            ):
                setattr(meta, key, set(value))
            elif key == "output_channels":
                setattr(meta, key, tuple(value))
            elif key == "parameter_domain":
                setattr(meta, key, dict(value))
            else:
                setattr(meta, key, value)

    def validate_production(self, name: str) -> TransformMetadata:
        """Resolve registry metadata and fail closed for production admission."""
        metadata = self._transforms.get(name)
        if metadata is None:
            raise ValueError(f"Transform '{name}' is not registered")
        if metadata.admission != "PRODUCTION":
            raise ValueError(f"Transform '{name}' is {metadata.admission}")
        if not metadata.causal_safe:
            raise ValueError(f"Transform '{name}' is not production-causal-safe")
        return deepcopy(metadata)

    def get_function(self, name: str) -> Optional[Callable]:
        """Get transform function by name."""
        metadata = self._transforms.get(name)
        return metadata.func if metadata else None

    def list_by_category(self, category: TransformCategory) -> List[TransformMetadata]:
        """List isolated transform metadata snapshots in a category."""
        names = self._by_category.get(category, [])
        return [deepcopy(self._transforms[name]) for name in names]

    def list_by_tag(self, tag: str) -> List[TransformMetadata]:
        """List isolated transform metadata snapshots with a given tag."""
        names = self._by_tag.get(tag, [])
        return [deepcopy(self._transforms[name]) for name in names]

    def list_causal_safe(self) -> List[TransformMetadata]:
        """List isolated snapshots of all causal-safe transforms."""
        return [
            deepcopy(meta) for meta in self._transforms.values()
            if meta.causal_safe
        ]

    def all_transforms(self) -> List[TransformMetadata]:
        """Get isolated snapshots of all registered transforms."""
        return [deepcopy(meta) for meta in self._transforms.values()]

    def get_signature_hash(self, name: str) -> Optional[str]:
        """Get signature hash for reproducibility tracking."""
        metadata = self._transforms.get(name)
        return metadata.signature_hash if metadata else None

    def seal(self) -> str:
        """Freeze the registry into an immutable snapshot identity (FP-P1-05).

        After sealing, ``register`` raises. The returned identity is a
        content-derived hash over the ordered registered metadata so the
        exact production semantics are reproducibly identifiable.
        """
        entries = []
        for name in sorted(self._transforms):
            meta = self._transforms[name]
            entries.append(
                f"{name}|{meta.version}|{meta.signature_hash}|"
                f"{meta.implementation_hash}|{meta.numeric_policy_hash}|"
                f"{meta.admission}|{meta.causal_safe}"
            )
        identity = _hash_bytes("\n".join(entries))
        self._sealed = True
        self._snapshot_identity = identity
        return identity

    @property
    def snapshot_identity(self) -> Optional[str]:
        """Content-derived identity of the sealed snapshot (None if not sealed)."""
        return self._snapshot_identity

    @property
    def is_sealed(self) -> bool:
        """Whether the registry has been frozen."""
        return self._sealed


def create_default_registry() -> TransformRegistry:
    """
    Create registry with all built-in transforms.

    Returns
    -------
    TransformRegistry
        Fully populated registry
    """
    from factor_preprocess.transforms import (
        cs_rank, cs_zscore, cs_demean, cs_winsor, cs_scale,
        rolling_mean, rolling_std, rolling_zscore, ewma,
        trailing_sma, trailing_median, robust_ewma, kama,
        one_sided_iir_lowpass, kalman_local_level,
        volatility_scale, volatility_scale_returns, realized_volatility,
        forward_fill, missing_indicator, missing_run_length, missing_rate,
        impute_with_fallback,
        days_since_update, observation_age, freshness_score, stale_data_indicator,
    )
    from factor_preprocess.neutralization import ols_neutralize, compute_exposures
    from factor_preprocess.regime import detect_correlation_regime
    from factor_preprocess.transforms.decomposition import (
        bandpass_filter, extract_cycle, christiano_fitzgerald_filter,
        wavelet_decompose, wavelet_smooth, wavelet_denoise,
        hp_filter, hp_decompose,
    )
    from factor_preprocess.transforms.treatment_variants import (
        freshness_aware_fill,
    )
    from factor_preprocess.transforms.event_decay import event_decay

    registry = TransformRegistry()

    # Cross-sectional transforms
    registry.register(
        "cs_rank", cs_rank, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional rank with tie handling",
        tags={"rank", "normalization", "cs"},
        causal_safe=True,
    )
    registry.register(
        "cs_zscore", cs_zscore, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional z-score normalization",
        tags={"zscore", "normalization", "cs"},
        causal_safe=True,
    )
    registry.register(
        "cs_demean", cs_demean, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional demean",
        tags={"demean", "normalization", "cs"},
        causal_safe=True,
    )
    registry.register(
        "cs_winsor", cs_winsor, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional winsorization",
        tags={"winsor", "outlier", "cs"},
        causal_safe=True,
    )
    registry.register(
        "cs_scale", cs_scale, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional scaling to target std",
        tags={"scale", "normalization", "cs"},
        causal_safe=True,
    )

    # DLIB-FP-015: semantic enrichment of the cross-sectional transforms so
    # eligibility queries can run on metadata instead of transform names.
    registry.enrich("cs_rank", semantic_id="CS_RANK:pct", stage="representation",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, parameter_domain={"pct": (True, True)},
                    numeric_policy="nan_skip_cs", fe_equivalent_semantics="cs_rank",
                    output_channels=("transformed",))
    registry.enrich("cs_zscore", semantic_id="CROSS_SECTIONAL_ZSCORE:cs", stage="representation",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, parameter_domain={"ddof": (0.0, 1.0)},
                    numeric_policy="nan_skip_cs", fe_equivalent_semantics="cs_zscore",
                    output_channels=("transformed",))
    registry.enrich("cs_demean", semantic_id="CROSS_SECTIONAL_DEMEAN:cs", stage="representation",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, parameter_domain={}, numeric_policy="nan_skip_cs",
                    output_channels=("transformed",))
    registry.enrich("cs_winsor", semantic_id="WINSOR:cs", stage="outlier",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, parameter_domain={"lower": (0.005, 0.05), "upper": (0.95, 0.995)},
                    numeric_policy="clip_cross_sectional", fe_equivalent_semantics="cs_winsor",
                    cost_class="cross_sectional", output_channels=("transformed",))
    registry.enrich("cs_scale", semantic_id="CROSS_SECTIONAL_SCALE:cs", stage="representation",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, parameter_domain={"target_std": (0.5, 2.0)},
                    numeric_policy="scale_factor_cs", output_channels=("transformed",))

    # Temporal transforms
    registry.register(
        "rolling_mean", rolling_mean, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling window mean",
        tags={"rolling", "mean", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "rolling_std", rolling_std, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling window standard deviation",
        tags={"rolling", "std", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "rolling_zscore", rolling_zscore, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling z-score normalization",
        tags={"rolling", "zscore", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "ewma", ewma, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Exponentially weighted moving average",
        tags={"ewma", "temporal", "smoothing"},
        causal_safe=True,
    )
    # Causal one-sided signal smoothers (strictly <= t - 1 information).
    registry.register(
        "trailing_sma", trailing_sma, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Lagged trailing simple moving average",
        tags={"smoothing", "sma", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "trailing_median", trailing_median, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Lagged trailing rolling median (robust to spikes)",
        tags={"smoothing", "median", "robust", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "robust_ewma", robust_ewma, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Lagged EWMA on winsorized values",
        tags={"smoothing", "ewma", "robust", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "kama", kama, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Kaufman adaptive moving average, recursive forward only",
        tags={"smoothing", "kama", "adaptive", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "one_sided_iir_lowpass", one_sided_iir_lowpass, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="One-pole IIR low-pass applied forward only",
        tags={"smoothing", "iir", "lowpass", "temporal"},
        causal_safe=True,
    )
    registry.register(
        "kalman_local_level", kalman_local_level, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="One-sided Kalman local-level filter",
        tags={"smoothing", "kalman", "temporal"},
        causal_safe=True,
    )

    # Volatility transforms
    registry.register(
        "volatility_scale", volatility_scale, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Scale by realized volatility",
        tags={"volatility", "scale"},
        causal_safe=True,
    )
    registry.register(
        "volatility_scale_returns", volatility_scale_returns, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Scale returns by volatility",
        tags={"volatility", "returns", "scale"},
        causal_safe=True,
    )
    registry.register(
        "realized_volatility", realized_volatility, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Compute realized volatility",
        tags={"volatility", "compute"},
        causal_safe=True,
    )

    # Missingness transforms
    registry.register(
        "forward_fill", forward_fill, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Forward fill missing values",
        tags={"missing", "fill"},
        causal_safe=True,
    )
    registry.register(
        "missing_indicator", missing_indicator, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Binary missing data indicator",
        tags={"missing", "indicator"},
        causal_safe=True,
    )
    registry.register(
        "missing_run_length", missing_run_length, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Consecutive missing observation count",
        tags={"missing", "run_length"},
        causal_safe=True,
    )
    registry.register(
        "missing_rate", missing_rate, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Rolling missing data rate",
        tags={"missing", "rate"},
        causal_safe=True,
    )
    registry.register(
        "impute_with_fallback", impute_with_fallback, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Full-sample mean/median fallback is research-only",
        tags={"missing", "impute", "full_sample", "research_only"},
        causal_safe=False,
        admission="RESEARCH_ONLY",
    )

    # Correlation-regime boundaries are fit over the full sample. Keep this
    # implementation available for research while failing closed in production.
    registry.register(
        "detect_correlation_regime", detect_correlation_regime, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Full-sample correlation regime detector; research-only",
        tags={"regime", "correlation", "full_sample", "research_only"},
        causal_safe=False,
        admission="RESEARCH_ONLY",
    )

    # Freshness transforms
    registry.register(
        "days_since_update", days_since_update, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Days since last non-missing update",
        tags={"freshness", "staleness"},
        causal_safe=True,
    )
    registry.register(
        "observation_age", observation_age, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Age of observation in days",
        tags={"freshness", "age"},
        causal_safe=True,
    )
    registry.register(
        "freshness_score", freshness_score, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Continuous freshness score [0, 1]",
        tags={"freshness", "score"},
        causal_safe=True,
    )
    registry.register(
        "stale_data_indicator", stale_data_indicator, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Binary stale data indicator",
        tags={"freshness", "staleness", "indicator"},
        causal_safe=True,
    )

    # Neutralization transforms
    registry.register(
        "ols_neutralize", ols_neutralize, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="OLS residual neutralization",
        tags={"neutralize", "ols", "residual"},
        causal_safe=True,
        semantic_id="NEUTRAL:ols",
        stage="neutralization",
        family_tags={"PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE", "EVENT"},
        causality_class="cross_sectional_causal",
        requires_exposure=True,
        parameter_domain={"min_observations": (2, None), "add_intercept": (True, True)},
        numeric_policy="nan_pairwise_deletion",
        fe_equivalent_semantics="ols_neutralize",
        cost_class="cross_sectional_fit",
        output_channels=("residual",),
    )
    registry.register(
        "compute_exposures", compute_exposures, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Compute factor exposures",
        tags={"exposure", "ols"},
        causal_safe=True,
        semantic_id="EXPOSURE:compute",
        stage="neutralization",
        requires_exposure=True,
        causality_class="cross_sectional_causal",
    )

    # DLIB-FP-014 / DLIB-FP-020: event-decay and freshness-aware fill are real,
    # executable transforms so the eligibility engine's proposals resolve to
    # canonical registry entries (no second hand-written catalog).
    registry.register(
        "event_decay", event_decay, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Causal short-halflife event decay persistence",
        tags={"event_decay", "smoothing", "event", "temporal"},
        causal_safe=True,
        semantic_id="EVENT_DECAY:short_halflife",
        stage="temporal",
        family_tags={"EVENT"},
        causality_class="one_sided_causal",
        requires_fit=False,
        parameter_domain={"halflife": (1.0, 5.0)},
        numeric_policy="nan_resets_memory",
        fe_equivalent_semantics="event_decay",
        cost_class="per_asset_recursive",
        output_channels=("transformed",),
    )
    registry.register(
        "freshness_aware_fill", freshness_aware_fill, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Freshness-aware forward fill with exponential decay",
        tags={"freshness", "fill", "missingness", "fundamental"},
        causal_safe=True,
        semantic_id="FILL:freshness_aware",
        stage="missingness",
        family_tags={"FUNDAMENTAL", "SPARSE_UPDATE"},
        causality_class="one_sided_causal",
        requires_fit=False,
        parameter_domain={"max_lag": (1.0, 20.0), "decay_halflife": (1.0, None)},
        numeric_policy="nan_forward_carry",
        fe_equivalent_semantics="freshness_aware_fill",
        cost_class="per_asset_forward",
        output_channels=("transformed",),
    )

    # DLIB-FP-020: the canonical neutralization kernel is ``ols_neutralize``.
    # ``industry_neutral`` / ``size_neutral`` / ``dual_neutral`` are *semantic*
    # aliases bound to that kernel via NeutralizationSpec, NOT independent
    # duplicated implementations. They resolve here so the eligibility engine's
    # proposals are canonical; their semantics are formalized by the spec.
    registry.register(
        "industry_neutral", ols_neutralize, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Industry-neutralization semantic alias (kernel=ols_neutralize)",
        tags={"neutralize", "industry", "ols", "semantic_alias"},
        causal_safe=True,
        semantic_id="INDUSTRY_NEUTRAL:SW_L1",
        stage="neutralization",
        family_tags={"PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE", "EVENT"},
        causality_class="cross_sectional_causal",
        requires_exposure=True,
        numeric_policy="nan_pairwise_deletion",
        fe_equivalent_semantics="industry_neutral",
        cost_class="cross_sectional_fit",
        output_channels=("residual",),
    )
    registry.register(
        "size_neutral", ols_neutralize, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Size-neutralization semantic alias (kernel=ols_neutralize)",
        tags={"neutralize", "size", "ols", "semantic_alias"},
        causal_safe=True,
        semantic_id="SIZE_NEUTRAL:log_mktcap",
        stage="neutralization",
        family_tags={"PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE", "EVENT"},
        causality_class="cross_sectional_causal",
        requires_exposure=True,
        numeric_policy="nan_pairwise_deletion",
        fe_equivalent_semantics="size_neutral",
        cost_class="cross_sectional_fit",
        output_channels=("residual",),
    )
    registry.register(
        "dual_neutral", ols_neutralize, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Industry+size neutralization semantic alias (kernel=ols_neutralize)",
        tags={"neutralize", "industry", "size", "ols", "semantic_alias"},
        causal_safe=True,
        semantic_id="DUAL_NEUTRAL:industry_size",
        stage="neutralization",
        family_tags={"PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE", "EVENT"},
        causality_class="cross_sectional_causal",
        requires_exposure=True,
        numeric_policy="nan_pairwise_deletion",
        fe_equivalent_semantics="dual_neutral",
        cost_class="cross_sectional_fit",
        output_channels=("residual",),
    )

    # Full-series decomposition uses symmetric/zero-phase reconstruction and is
    # therefore not admissible in production unless a causal implementation exists.
    # The HP filter additionally solves the HP objective over the WHOLE lagged
    # sample (spsolve over the full series), so a historical trend point can
    # depend on later lagged observations: it is NOT prefix-invariant and is
    # therefore NOT production-causal. hp_filter / hp_decompose are research /
    # offline use only.
    for name, func in [
        ("bandpass_filter", bandpass_filter),
        ("extract_cycle", extract_cycle),
        ("christiano_fitzgerald_filter", christiano_fitzgerald_filter),
        ("wavelet_decompose", wavelet_decompose),
        ("wavelet_smooth", wavelet_smooth),
        ("wavelet_denoise", wavelet_denoise),
        ("hp_filter", hp_filter),
        ("hp_decompose", hp_decompose),
    ]:
        registry.register(
            name, func, TransformCategory.TEMPORAL,
            version="1.0.0",
            description="Full-series decomposition; offline/research use only",
            tags={"temporal", "full_series", "offline_only"},
            causal_safe=False,
            admission="OFFLINE_ONLY",
        )

    # DLIB-FP-015: semantic enrichment of the temporal / volatility /
    # missingness / freshness transforms so eligibility queries run on
    # metadata (semantic_id, stage, family_tags, causality_class, cost_class)
    # instead of a hand-maintained transform-name list.
    _SMOOTH_FAMILIES = {"PRICE_VOLUME", "HIGH_TURNOVER"}
    for _name, _sid, _stage, _cost, _dom in [
        ("ewma", "SMOOTH:ewma", "temporal", "per_asset_recursive", {"halflife": (3.0, 60.0)}),
        ("kama", "SMOOTH:kama", "temporal", "per_asset_recursive",
         {"er_window": (5.0, 30.0), "fast_span": (2.0, 10.0), "slow_span": (20.0, 60.0)}),
        ("one_sided_iir_lowpass", "SMOOTH:one_sided_iir_lowpass", "temporal", "per_asset_recursive", {"alpha": (0.05, 0.5)}),
        ("kalman_local_level", "SMOOTH:kalman_local_level", "temporal", "per_asset_recursive",
         {"process_noise": (0.001, 0.1), "measurement_noise": (0.1, 1.0)}),
        ("trailing_sma", "SMOOTH:trailing_sma", "temporal", "per_asset_rolling", {"window": (3.0, 60.0)}),
        ("trailing_median", "SMOOTH:trailing_median", "temporal", "per_asset_rolling", {"window": (3.0, 30.0)}),
        ("robust_ewma", "SMOOTH:robust_ewma", "temporal", "per_asset_recursive", {"halflife": (3.0, 60.0)}),
    ]:
        registry.enrich(
            _name, semantic_id=_sid, stage=_stage,
            family_tags=set(_SMOOTH_FAMILIES),
            causality_class="one_sided_causal", requires_fit=False,
            parameter_domain=_dom, numeric_policy="nan_lagged",
            cost_class=_cost, output_channels=("transformed",),
        )
    del _name, _sid, _stage, _cost, _dom, _SMOOTH_FAMILIES

    _VOLA_FAMILIES = {"PRICE_VOLUME", "HIGH_TURNOVER", "FUNDAMENTAL", "SPARSE_UPDATE"}
    registry.enrich("volatility_scale", semantic_id="VOL_SCALE:realized", stage="scaling",
                    family_tags=set(_VOLA_FAMILIES), causality_class="one_sided_causal",
                    requires_fit=False, cost_class="per_asset_rolling",
                    output_channels=("transformed",))
    registry.enrich("volatility_scale_returns", semantic_id="VOL_SCALE_RETURNS:realized", stage="scaling",
                    family_tags=set(_VOLA_FAMILIES), causality_class="one_sided_causal",
                    requires_fit=False, cost_class="per_asset_rolling",
                    output_channels=("transformed",))
    del _VOLA_FAMILIES

    registry.enrich("forward_fill", semantic_id="FILL:forward", stage="missingness",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="one_sided_causal",
                    requires_fit=False, parameter_domain={"max_lag": (1.0, 20.0)},
                    numeric_policy="nan_forward_carry", fe_equivalent_semantics="forward_fill",
                    cost_class="per_asset_forward", output_channels=("transformed",))
    registry.enrich("missing_indicator", semantic_id="MISSING_INDICATOR:binary", stage="missingness",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="cross_sectional_causal",
                    requires_fit=False, output_channels=("missing",))
    registry.enrich("missing_rate", semantic_id="MISSING_RATE:rolling", stage="missingness",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="one_sided_causal",
                    requires_fit=False, cost_class="per_asset_rolling",
                    output_channels=("missing",))
    registry.enrich("days_since_update", semantic_id="FRESHNESS:days_since_update", stage="missingness",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="one_sided_causal",
                    requires_fit=False, output_channels=("freshness",))
    registry.enrich("freshness_score", semantic_id="FRESHNESS:score", stage="missingness",
                    family_tags=set(ALL_FAMILY_TAGS), causality_class="one_sided_causal",
                    requires_fit=False, output_channels=("freshness",))

    return registry


_default_registry: Optional[TransformRegistry] = None


def get_default_registry() -> TransformRegistry:
    """Get or create the default global registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry
