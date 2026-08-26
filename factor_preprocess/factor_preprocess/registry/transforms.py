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
from typing import Callable, Dict, List, Optional, Set, Any
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
    """Metadata for a registered transform."""
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

    def __post_init__(self):
        """Compute hashes after initialization."""
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
    )
    registry.register(
        "compute_exposures", compute_exposures, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Compute factor exposures",
        tags={"exposure", "ols"},
        causal_safe=True,
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

    return registry


_default_registry: Optional[TransformRegistry] = None


def get_default_registry() -> TransformRegistry:
    """Get or create the default global registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry
