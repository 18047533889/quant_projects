"""
Policy presets for transform pipelines.

Named policy configurations that encode common preprocessing workflows
with explicit causal safety and production readiness guarantees.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import warnings


class PolicyLevel(str, Enum):
    """Policy strictness level."""
    RESEARCH = "research"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class TransformStep:
    """Single transform step in a pipeline.

    ``step_id`` is the unique key inside a pipeline.  Duplicate ``name``
    values are allowed as long as each step carries a distinct ``step_id``
    (e.g. ``rank_raw`` followed by ``rank_residual``).
    """
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    step_id: Optional[str] = None
    skip_if_missing: bool = False

    def __post_init__(self):
        # Backwards compat: callers that relied solely on ``name`` as the
        # uniqueness key should still work; we only enforce uniqueness on
        # ``step_id`` when the pipeline is validated.
        if self.step_id is None:
            object.__setattr__(self, 'step_id', self.name)


@dataclass
class PolicyPreset:
    """
    Named policy configuration for preprocessing pipelines.

    A policy defines:
    - Ordered sequence of transforms
    - Parameter overrides
    - Causal safety guarantees
    - Production readiness level
    """
    name: str
    description: str
    level: PolicyLevel
    steps: List[TransformStep]
    causal_safe: bool = True
    requires_universe: bool = False
    requires_returns: bool = False
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        # No instance-level deprecation warning here; PolicyRegistry.register
        # already emits the canonical authority warning once.
        pass

    def validate(self, transform_registry=None) -> Tuple[bool, List[str]]:
        """
        Validate policy configuration.

        Returns
        -------
        valid : bool
            Whether policy is valid
        errors : list of str
            Validation error messages
        """
        errors = []

        if not self.name:
            errors.append("Policy name cannot be empty")

        if not self.steps:
            errors.append("Policy must contain at least one step")

        if transform_registry is None:
            from factor_preprocess.registry.transforms import get_default_registry
            transform_registry = get_default_registry()

        for step in self.steps:
            try:
                if self.level == PolicyLevel.PRODUCTION:
                    metadata = transform_registry.validate_production(step.name)
                else:
                    metadata = transform_registry.get(step.name)
                    if metadata is None:
                        raise ValueError(f"Transform '{step.name}' is not registered")
                metadata.bind_parameters(step.parameters)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))

        if self.level == PolicyLevel.PRODUCTION and not self.causal_safe:
            errors.append("Production policies must be causal_safe=True")

        # Duplicate step names are allowed when the caller explicitly assigns
        # different step_ids, so the uniqueness constraint is enforced on
        # step_id instead of name.
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step_id in pipeline")

        return len(errors) == 0, errors


class PolicyRegistry:
    """
    Registry of named policy presets.

    Provides catalog of validated preprocessing workflows for
    different use cases and safety levels.
    """

    def __init__(self):
        self._policies: Dict[str, PolicyPreset] = {}

    def register(self, policy: PolicyPreset) -> None:
        """
        Register a policy preset.

        Parameters
        ----------
        policy : PolicyPreset
            Policy to register

        Raises
        ------
        ValueError
            If policy validation fails
        """
        valid, errors = policy.validate()
        if not valid:
            raise ValueError(f"Invalid policy '{policy.name}': {'; '.join(errors)}")

        if policy.name in self._policies:
            raise ValueError(f"Policy '{policy.name}' already registered")

        self._policies[policy.name] = deepcopy(policy)

    def get(self, name: str) -> Optional[PolicyPreset]:
        """Get an isolated policy snapshot by name."""
        policy = self._policies.get(name)
        return deepcopy(policy) if policy is not None else None

    def list_by_level(self, level: PolicyLevel) -> List[PolicyPreset]:
        """List isolated policy snapshots at a given strictness level."""
        return [
            deepcopy(policy) for policy in self._policies.values()
            if policy.level == level
        ]

    def list_causal_safe(self) -> List[PolicyPreset]:
        """List isolated snapshots of all causal-safe policies."""
        return [
            deepcopy(policy) for policy in self._policies.values()
            if policy.causal_safe
        ]

    def all_policies(self) -> List[PolicyPreset]:
        """Get isolated snapshots of all registered policies."""
        return [deepcopy(policy) for policy in self._policies.values()]


def create_default_policies() -> PolicyRegistry:
    """
    Create registry with standard policy presets.

    Returns
    -------
    PolicyRegistry
        Registry with built-in policies
    """
    registry = PolicyRegistry()

    # Policy 1: cs_only - Pure cross-sectional normalization
    cs_only = PolicyPreset(
        name="cs_only",
        description="Cross-sectional rank normalization only, no temporal smoothing",
        level=PolicyLevel.RESEARCH,
        steps=[
            TransformStep(
                name="cs_winsor",
                parameters={"lower": 0.01, "upper": 0.99},
            ),
            TransformStep(
                name="cs_rank",
                parameters={"pct": True},
            ),
            TransformStep(
                name="cs_zscore",
                parameters={"ddof": 1},
            ),
        ],
        causal_safe=True,
        requires_universe=False,
        requires_returns=False,
        tags=["cross_sectional", "rank", "minimal"],
    )
    registry.register(cs_only)

    # Policy 2: causal_basic - Basic causal-safe pipeline with temporal smoothing
    causal_basic = PolicyPreset(
        name="causal_basic",
        description="Causal-safe preprocessing with outlier removal and temporal smoothing",
        level=PolicyLevel.STAGING,
        steps=[
            TransformStep(
                name="forward_fill",
                parameters={"max_lag": 5},
            ),
            TransformStep(
                name="cs_winsor",
                parameters={"lower": 0.025, "upper": 0.975},
            ),
            TransformStep(
                name="ewma",
                parameters={"halflife": 20},
            ),
            TransformStep(
                name="cs_rank",
                parameters={"pct": True},
            ),
            TransformStep(
                name="cs_zscore",
                parameters={"ddof": 1},
            ),
        ],
        causal_safe=True,
        requires_universe=False,
        requires_returns=False,
        tags=["causal", "temporal", "smoothing"],
    )
    registry.register(causal_basic)

    # Policy 3: production_full - Full production pipeline with all safety checks
    production_full = PolicyPreset(
        name="production_full",
        description="Production-grade pipeline with volatility scaling and neutralization",
        level=PolicyLevel.PRODUCTION,
        steps=[
            TransformStep(
                name="forward_fill",
                parameters={"max_lag": 3},
            ),
            TransformStep(
                name="missing_indicator",
                parameters={},
            ),
            TransformStep(
                name="cs_winsor",
                parameters={"lower": 0.01, "upper": 0.99},
            ),
            TransformStep(
                name="ewma",
                parameters={"halflife": 20, "min_periods": 10},
            ),
            TransformStep(
                name="volatility_scale",
                parameters={"window": 60, "min_periods": 20},
            ),
            TransformStep(
                name="cs_rank",
                parameters={"pct": True},
            ),
            TransformStep(
                name="cs_zscore",
                parameters={"ddof": 1},
            ),
            TransformStep(
                name="ols_neutralize",
                parameters={},
                skip_if_missing=False,
            ),
        ],
        causal_safe=True,
        requires_universe=True,
        requires_returns=False,
        tags=["production", "full", "volatility", "neutralization"],
    )
    registry.register(production_full)

    # Policy 4: returns_preprocessing - Specialized for return series
    returns_preprocessing = PolicyPreset(
        name="returns_preprocessing",
        description="Preprocessing for factor returns (not levels)",
        level=PolicyLevel.STAGING,
        steps=[
            TransformStep(
                name="cs_winsor",
                parameters={"lower": 0.01, "upper": 0.99},
            ),
            TransformStep(
                name="volatility_scale_returns",
                parameters={"window": 60},
            ),
            TransformStep(
                name="cs_zscore",
                parameters={"ddof": 1},
            ),
        ],
        causal_safe=True,
        requires_universe=False,
        requires_returns=True,
        tags=["returns", "volatility"],
    )
    registry.register(returns_preprocessing)

    # Policy 5: minimal - Absolute minimal preprocessing
    minimal = PolicyPreset(
        name="minimal",
        description="Minimal preprocessing for already-clean factors",
        level=PolicyLevel.RESEARCH,
        steps=[
            TransformStep(
                name="cs_rank",
                parameters={"pct": True},
            ),
        ],
        causal_safe=True,
        requires_universe=False,
        requires_returns=False,
        tags=["minimal", "rank_only"],
    )
    registry.register(minimal)

    # Policy 6: research_full - Full research pipeline without production constraints
    research_full = PolicyPreset(
        name="research_full",
        description="Full-featured research pipeline with generous fill limits",
        level=PolicyLevel.RESEARCH,
        steps=[
            TransformStep(
                name="forward_fill",
                parameters={"max_lag": 10},
            ),
            TransformStep(
                name="missing_rate",
                parameters={"window": 60},
            ),
            TransformStep(
                name="cs_winsor",
                parameters={"lower": 0.025, "upper": 0.975},
            ),
            TransformStep(
                name="rolling_zscore",
                parameters={"window": 120, "min_periods": 60},
            ),
            TransformStep(
                name="cs_rank",
                parameters={"pct": True},
            ),
        ],
        causal_safe=True,
        requires_universe=False,
        requires_returns=False,
        tags=["research", "full", "diagnostic"],
    )
    registry.register(research_full)

    return registry


# Global default policy registry
_default_policy_registry: Optional[PolicyRegistry] = None


def get_default_policy_registry() -> PolicyRegistry:
    """Get or create the default global policy registry."""
    global _default_policy_registry
    if _default_policy_registry is None:
        _default_policy_registry = create_default_policies()
    return _default_policy_registry
