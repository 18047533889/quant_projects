"""
Transform registry with versioning and discovery.

Provides a central catalog of all preprocessing transforms with metadata,
versioning, and category-based organization.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any
from enum import Enum
import hashlib
import inspect


class TransformCategory(str, Enum):
    """Transform category classification."""
    CROSS_SECTIONAL = "cross_sectional"
    TEMPORAL = "temporal"
    VOLATILITY = "volatility"
    MISSINGNESS = "missingness"
    FRESHNESS = "freshness"
    NEUTRALIZATION = "neutralization"
    REPRESENTATION = "representation"


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
    # Fail-closed defaults: causal safety and production admission are never
    # implied by omission.  A registration without an explicit causal claim
    # stays UNVERIFIED until a certification pipeline upgrades it.
    causal_safe: bool = False
    causal_verified: bool = False
    admission: str = "UNVERIFIED"
    signature_hash: Optional[str] = None
    implementation_hash: Optional[str] = None

    def __post_init__(self):
        """Compute signature and implementation hashes after initialization."""
        if self.signature_hash is None:
            self.signature_hash = self._compute_signature_hash()
        if self.implementation_hash is None:
            self.implementation_hash = self._compute_implementation_hash()

    def _compute_signature_hash(self) -> str:
        """Compute stable hash of function signature."""
        sig = inspect.signature(self.func)
        sig_str = f"{self.name}:{str(sig)}"
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def _compute_implementation_hash(self) -> str:
        """Hash source code + semantic version, not just the signature.

        ``signature_hash`` cannot detect an algorithm rewrite that keeps the
        same parameter list; ``implementation_hash`` binds the transform body
        so reproducibility identities change when the code changes.
        """
        try:
            source = inspect.getsource(self.func)
        except (OSError, TypeError):
            source = repr(getattr(self.func, "__code__", self.func))
        impl_str = f"{self.name}:{self.version}:{source}"
        return hashlib.sha256(impl_str.encode()).hexdigest()[:16]

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

    def register(
        self,
        name: str,
        func: Callable,
        category: TransformCategory,
        version: str = "1.0.0",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        causal_safe: Optional[bool] = None,
        causal_verified: bool = False,
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
        causal_safe : bool, optional
            Whether transform preserves causal structure.  Omission means
            UNVERIFIED (fail-closed), not True.
        causal_verified : bool
            Whether causal safety has been certified by a test pipeline.
        admission : str
            Production admission class.  Omission means UNVERIFIED.
        """
        if causal_safe is None:
            causal_safe = causal_verified
        if admission is None:
            admission = "CAUSAL_CERTIFIED" if causal_verified else "UNVERIFIED"
        elif admission not in {
            "PRODUCTION", "OFFLINE_ONLY", "RESEARCH_ONLY",
            "UNVERIFIED", "CAUSAL_CERTIFIED",
        }:
            raise ValueError(
                "admission must be PRODUCTION, OFFLINE_ONLY, RESEARCH_ONLY, "
                "UNVERIFIED, or CAUSAL_CERTIFIED"
            )
        if admission in {"PRODUCTION", "CAUSAL_CERTIFIED"} and not causal_safe:
            raise ValueError(
                f"admission={admission} requires causal_safe=True; certify the "
                "transform before claiming causal safety"
            )

        metadata = TransformMetadata(
            name=name,
            func=func,
            category=category,
            version=version,
            description=description,
            parameters=deepcopy(parameters or {}),
            tags=deepcopy(tags or set()),
            causal_safe=causal_safe,
            causal_verified=causal_verified,
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
                and existing.causal_verified == metadata.causal_verified
                and existing.admission == metadata.admission
                and existing.signature_hash == metadata.signature_hash
                and existing.implementation_hash == metadata.implementation_hash
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

    def get_implementation_hash(self, name: str) -> Optional[str]:
        """Get source-bound implementation hash for reproducibility tracking."""
        metadata = self._transforms.get(name)
        return metadata.implementation_hash if metadata else None


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
    )

    registry = TransformRegistry()

    # Cross-sectional transforms
    registry.register(
        "cs_rank", cs_rank, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional rank with tie handling",
        tags={"rank", "normalization", "cs"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "cs_zscore", cs_zscore, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional z-score normalization",
        tags={"zscore", "normalization", "cs"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "cs_demean", cs_demean, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional demean",
        tags={"demean", "normalization", "cs"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "cs_winsor", cs_winsor, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional winsorization",
        tags={"winsor", "outlier", "cs"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "cs_scale", cs_scale, TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Cross-sectional scaling to target std",
        tags={"scale", "normalization", "cs"},
        causal_safe=True,
        admission="PRODUCTION",
    )

    # Temporal transforms
    registry.register(
        "rolling_mean", rolling_mean, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling window mean",
        tags={"rolling", "mean", "temporal"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "rolling_std", rolling_std, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling window standard deviation",
        tags={"rolling", "std", "temporal"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "rolling_zscore", rolling_zscore, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Rolling z-score normalization",
        tags={"rolling", "zscore", "temporal"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "ewma", ewma, TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Exponentially weighted moving average",
        tags={"ewma", "temporal", "smoothing"},
        causal_safe=True,
        admission="PRODUCTION",
    )

    # Volatility transforms
    registry.register(
        "volatility_scale", volatility_scale, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Scale by realized volatility",
        tags={"volatility", "scale"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "volatility_scale_returns", volatility_scale_returns, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Scale returns by volatility",
        tags={"volatility", "returns", "scale"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "realized_volatility", realized_volatility, TransformCategory.VOLATILITY,
        version="1.0.0",
        description="Compute realized volatility",
        tags={"volatility", "compute"},
        causal_safe=True,
        admission="PRODUCTION",
    )

    # Missingness transforms
    registry.register(
        "forward_fill", forward_fill, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Forward fill missing values",
        tags={"missing", "fill"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "missing_indicator", missing_indicator, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Binary missing data indicator",
        tags={"missing", "indicator"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "missing_run_length", missing_run_length, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Consecutive missing observation count",
        tags={"missing", "run_length"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "missing_rate", missing_rate, TransformCategory.MISSINGNESS,
        version="1.0.0",
        description="Rolling missing data rate",
        tags={"missing", "rate"},
        causal_safe=True,
        admission="PRODUCTION",
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
        admission="PRODUCTION",
    )
    registry.register(
        "observation_age", observation_age, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Age of observation in days",
        tags={"freshness", "age"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "freshness_score", freshness_score, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Continuous freshness score [0, 1]",
        tags={"freshness", "score"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "stale_data_indicator", stale_data_indicator, TransformCategory.FRESHNESS,
        version="1.0.0",
        description="Binary stale data indicator",
        tags={"freshness", "staleness", "indicator"},
        causal_safe=True,
        admission="PRODUCTION",
    )

    # Neutralization transforms
    registry.register(
        "ols_neutralize", ols_neutralize, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="OLS residual neutralization",
        tags={"neutralize", "ols", "residual"},
        causal_safe=True,
        admission="PRODUCTION",
    )
    registry.register(
        "compute_exposures", compute_exposures, TransformCategory.NEUTRALIZATION,
        version="1.0.0",
        description="Compute factor exposures",
        tags={"exposure", "ols"},
        causal_safe=True,
        admission="PRODUCTION",
    )

    # Full-series decomposition uses symmetric/zero-phase reconstruction and is
    # therefore not admissible in production unless a causal implementation exists.
    for name, func in [
        ("bandpass_filter", bandpass_filter),
        ("extract_cycle", extract_cycle),
        ("christiano_fitzgerald_filter", christiano_fitzgerald_filter),
        ("wavelet_decompose", wavelet_decompose),
        ("wavelet_smooth", wavelet_smooth),
        ("wavelet_denoise", wavelet_denoise),
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
