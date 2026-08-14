"""Tests for policy registry."""
import pytest
from factor_preprocess.registry import (
    PolicyRegistry,
    PolicyPreset,
    PolicyLevel,
    TransformRegistry,
    TransformCategory,
    TransformStep,
    get_default_policy_registry,
)


class TestTransformStep:
    """Test TransformStep dataclass."""

    def test_transform_step_creation(self):
        """Test creating a transform step."""
        step = TransformStep(
            name="cs_rank",
            parameters={"pct": True},
        )

        assert step.name == "cs_rank"
        assert step.parameters == {"pct": True}
        assert step.skip_if_missing is False

    def test_transform_step_skip_flag(self):
        """Test skip_if_missing flag."""
        step = TransformStep(
            name="ols_neutralize",
            parameters={},
            skip_if_missing=True,
        )

        assert step.skip_if_missing is True


class TestPolicyPreset:
    """Test PolicyPreset dataclass and validation."""

    def test_policy_creation(self):
        """Test creating a policy preset."""
        policy = PolicyPreset(
            name="test_policy",
            description="Test policy",
            level=PolicyLevel.RESEARCH,
            steps=[
                TransformStep("cs_rank", {"pct": True}),
                TransformStep("cs_zscore", {}),
            ],
        )

        assert policy.name == "test_policy"
        assert policy.level == PolicyLevel.RESEARCH
        assert len(policy.steps) == 2
        assert policy.causal_safe is True

    def test_policy_validation_success(self):
        """Test validation of valid policy."""
        policy = PolicyPreset(
            name="valid",
            description="Valid policy",
            level=PolicyLevel.STAGING,
            steps=[TransformStep("cs_rank", {})],
        )

        valid, errors = policy.validate()
        assert valid is True
        assert len(errors) == 0

    def test_policy_validation_empty_name(self):
        """Test validation fails on empty name."""
        policy = PolicyPreset(
            name="",
            description="No name",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
        )

        valid, errors = policy.validate()
        assert valid is False
        assert any("name cannot be empty" in err for err in errors)

    def test_policy_validation_no_steps(self):
        """Test validation fails on empty steps."""
        policy = PolicyPreset(
            name="empty",
            description="No steps",
            level=PolicyLevel.RESEARCH,
            steps=[],
        )

        valid, errors = policy.validate()
        assert valid is False
        assert any("at least one step" in err for err in errors)

    def test_policy_validation_production_must_be_causal_safe(self):
        """Test production policies must be causal_safe."""
        policy = PolicyPreset(
            name="bad_prod",
            description="Production but not causal safe",
            level=PolicyLevel.PRODUCTION,
            steps=[TransformStep("cs_rank", {})],
            causal_safe=False,
        )

        valid, errors = policy.validate()
        assert valid is False
        assert any("must be causal_safe=True" in err for err in errors)

    def test_production_cannot_trust_user_causal_safe_flag(self):
        from factor_preprocess.registry import TransformCategory

        registry = TransformRegistry()
        registry.register(
            "unsafe_full_series", lambda values: values,
            TransformCategory.TEMPORAL, causal_safe=False,
            admission="OFFLINE_ONLY",
        )
        policy = PolicyPreset(
            name="claimed_safe", description="unsafe", level=PolicyLevel.PRODUCTION,
            steps=[TransformStep("unsafe_full_series", {})], causal_safe=True,
        )
        valid, errors = policy.validate(transform_registry=registry)
        assert valid is False
        assert any("OFFLINE_ONLY" in error for error in errors)

    def test_production_rejects_unregistered_transform(self):
        policy = PolicyPreset(
            name="unknown_step", description="unknown", level=PolicyLevel.PRODUCTION,
            steps=[TransformStep("unregistered_transform", {})], causal_safe=True,
        )
        valid, errors = policy.validate(transform_registry=TransformRegistry())
        assert valid is False
        assert any("not registered" in error for error in errors)

    def test_production_resolves_every_step_against_registry(self):
        registry = TransformRegistry()
        registry.register(
            "safe_step", lambda values: values,
            TransformCategory.TEMPORAL,
        )
        policy = PolicyPreset(
            name="mixed_steps",
            description="one registered and one unknown step",
            level=PolicyLevel.PRODUCTION,
            steps=[
                TransformStep("safe_step", {}),
                TransformStep("missing_step", {}),
            ],
            causal_safe=True,
        )

        valid, errors = policy.validate(transform_registry=registry)
        assert valid is False
        assert any("missing_step" in error and "not registered" in error for error in errors)

    def test_policy_validation_duplicate_steps(self):
        """Test validation fails on duplicate step names."""
        policy = PolicyPreset(
            name="duplicate",
            description="Duplicate steps",
            level=PolicyLevel.RESEARCH,
            steps=[
                TransformStep("cs_rank", {}),
                TransformStep("cs_rank", {"pct": True}),
            ],
        )

        valid, errors = policy.validate()
        assert valid is False
        assert any("Duplicate transform names" in err for err in errors)


class TestPolicyRegistry:
    """Test PolicyRegistry functionality."""

    def test_register_and_get(self):
        """Test registering and retrieving policies."""
        registry = PolicyRegistry()

        policy = PolicyPreset(
            name="test",
            description="Test",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
        )

        registry.register(policy)

        retrieved = registry.get("test")
        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved is policy

    def test_register_invalid_policy_raises(self):
        """Test registering invalid policy raises."""
        registry = PolicyRegistry()

        policy = PolicyPreset(
            name="invalid",
            description="Invalid",
            level=PolicyLevel.RESEARCH,
            steps=[],  # Empty steps - invalid
        )

        with pytest.raises(ValueError, match="Invalid policy"):
            registry.register(policy)

    def test_register_duplicate_raises(self):
        """Test registering duplicate policy name raises."""
        registry = PolicyRegistry()

        policy1 = PolicyPreset(
            name="test",
            description="First",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
        )
        policy2 = PolicyPreset(
            name="test",
            description="Second",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_zscore", {})],
        )

        registry.register(policy1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(policy2)

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent policy returns None."""
        registry = PolicyRegistry()
        assert registry.get("nonexistent") is None

    def test_list_by_level(self):
        """Test listing policies by level."""
        registry = PolicyRegistry()

        research = PolicyPreset(
            name="research",
            description="Research",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
        )
        staging = PolicyPreset(
            name="staging",
            description="Staging",
            level=PolicyLevel.STAGING,
            steps=[TransformStep("cs_rank", {})],
        )
        production = PolicyPreset(
            name="production",
            description="Production",
            level=PolicyLevel.PRODUCTION,
            steps=[TransformStep("cs_rank", {})],
        )

        registry.register(research)
        registry.register(staging)
        registry.register(production)

        research_policies = registry.list_by_level(PolicyLevel.RESEARCH)
        assert len(research_policies) == 1
        assert research_policies[0].name == "research"

        staging_policies = registry.list_by_level(PolicyLevel.STAGING)
        assert len(staging_policies) == 1
        assert staging_policies[0].name == "staging"

        production_policies = registry.list_by_level(PolicyLevel.PRODUCTION)
        assert len(production_policies) == 1
        assert production_policies[0].name == "production"

    def test_list_causal_safe(self):
        """Test listing causal-safe policies."""
        registry = PolicyRegistry()

        safe1 = PolicyPreset(
            name="safe1",
            description="Safe",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
            causal_safe=True,
        )
        safe2 = PolicyPreset(
            name="safe2",
            description="Safe",
            level=PolicyLevel.STAGING,
            steps=[TransformStep("cs_rank", )],
            causal_safe=True,
        )
        unsafe = PolicyPreset(
            name="unsafe",
            description="Unsafe",
            level=PolicyLevel.RESEARCH,
            steps=[TransformStep("cs_rank", {})],
            causal_safe=False,
        )

        registry.register(safe1)
        registry.register(safe2)
        registry.register(unsafe)

        safe_policies = registry.list_causal_safe()
        assert len(safe_policies) == 2
        assert set(p.name for p in safe_policies) == {"safe1", "safe2"}

    def test_all_policies(self):
        """Test getting all policies."""
        registry = PolicyRegistry()

        p1 = PolicyPreset("p1", "First", PolicyLevel.RESEARCH, [TransformStep("cs_rank", {})])
        p2 = PolicyPreset("p2", "Second", PolicyLevel.STAGING, [TransformStep("cs_zscore", {})])

        registry.register(p1)
        registry.register(p2)

        all_policies = registry.all_policies()
        assert len(all_policies) == 2
        assert set(p.name for p in all_policies) == {"p1", "p2"}


class TestDefaultPolicies:
    """Test default policy registry."""

    def test_default_policy_registry_exists(self):
        """Test default policy registry can be retrieved."""
        registry = get_default_policy_registry()
        assert registry is not None
        assert isinstance(registry, PolicyRegistry)

    def test_default_policy_registry_singleton(self):
        """Test default policy registry is singleton."""
        registry1 = get_default_policy_registry()
        registry2 = get_default_policy_registry()
        assert registry1 is registry2

    def test_cs_only_policy_exists(self):
        """Test cs_only policy exists and is valid."""
        registry = get_default_policy_registry()

        policy = registry.get("cs_only")
        assert policy is not None
        assert policy.name == "cs_only"
        assert policy.level == PolicyLevel.RESEARCH
        assert policy.causal_safe is True
        assert len(policy.steps) > 0

        # Check specific steps
        step_names = [step.name for step in policy.steps]
        assert "cs_rank" in step_names
        assert "cs_zscore" in step_names

    def test_causal_basic_policy_exists(self):
        """Test causal_basic policy exists and is valid."""
        registry = get_default_policy_registry()

        policy = registry.get("causal_basic")
        assert policy is not None
        assert policy.level == PolicyLevel.STAGING
        assert policy.causal_safe is True

        step_names = [step.name for step in policy.steps]
        assert "forward_fill" in step_names
        assert "ewma" in step_names
        assert "cs_winsor" in step_names

    def test_production_full_policy_exists(self):
        """Test production_full policy exists and is valid."""
        registry = get_default_policy_registry()

        policy = registry.get("production_full")
        assert policy is not None
        assert policy.level == PolicyLevel.PRODUCTION
        assert policy.causal_safe is True
        assert policy.requires_universe is True

        step_names = [step.name for step in policy.steps]
        assert "volatility_scale" in step_names
        assert "ols_neutralize" in step_names
        assert "missing_indicator" in step_names

    def test_returns_preprocessing_policy_exists(self):
        """Test returns_preprocessing policy exists."""
        registry = get_default_policy_registry()

        policy = registry.get("returns_preprocessing")
        assert policy is not None
        assert policy.requires_returns is True

        step_names = [step.name for step in policy.steps]
        assert "volatility_scale_returns" in step_names

    def test_minimal_policy_exists(self):
        """Test minimal policy exists."""
        registry = get_default_policy_registry()

        policy = registry.get("minimal")
        assert policy is not None
        assert policy.level == PolicyLevel.RESEARCH
        assert len(policy.steps) == 1
        assert policy.steps[0].name == "cs_rank"

    def test_research_full_policy_exists(self):
        """Test research_full policy exists."""
        registry = get_default_policy_registry()

        policy = registry.get("research_full")
        assert policy is not None
        assert policy.level == PolicyLevel.RESEARCH

        step_names = [step.name for step in policy.steps]
        assert "missing_rate" in step_names
        assert "rolling_zscore" in step_names

    def test_all_default_policies_valid(self):
        """Test all default policies pass validation."""
        registry = get_default_policy_registry()

        all_policies = registry.all_policies()
        assert len(all_policies) >= 6  # At least 6 default policies

        for policy in all_policies:
            valid, errors = policy.validate()
            assert valid is True, f"Policy {policy.name} failed validation: {errors}"

    def test_all_default_policies_causal_safe(self):
        """Test all default policies are causal safe."""
        registry = get_default_policy_registry()

        all_policies = registry.all_policies()
        for policy in all_policies:
            assert policy.causal_safe is True, f"Policy {policy.name} not causal_safe"

    def test_production_level_policies_meet_requirements(self):
        """Test production-level policies meet strictness requirements."""
        registry = get_default_policy_registry()

        prod_policies = registry.list_by_level(PolicyLevel.PRODUCTION)
        assert len(prod_policies) > 0

        for policy in prod_policies:
            assert policy.causal_safe is True
            valid, _ = policy.validate()
            assert valid is True

    def test_policy_step_parameters(self):
        """Test that policy steps have reasonable parameters."""
        registry = get_default_policy_registry()

        policy = registry.get("production_full")
        assert policy is not None

        # Find forward_fill step and check limit parameter
        ff_step = next((s for s in policy.steps if s.name == "forward_fill"), None)
        assert ff_step is not None
        assert "limit" in ff_step.parameters
        assert ff_step.parameters["limit"] <= 5  # Production should be conservative
