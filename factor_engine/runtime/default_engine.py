"""Versioned durable facade over FactorEngine's existing admitted execution.

An administrator selects a deployment file via FACTOR_ENGINE_V2_PROFILE. The
file binds business scope, not per-request performance tuning or publication.
No default dataset, universe, date range or output path is guessed here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path

from .default_execution_policy import DefaultExecutionPolicy, resolve_default_policy


class DeploymentConfigurationError(ValueError):
    reason_code = "DEPLOYMENT_CONFIGURATION_REQUIRED"

    def __init__(self, missing_fields=(), *, detail=""):
        self.missing_fields = tuple(sorted(missing_fields))
        super().__init__("v2 deployment configuration: " +
                         (", ".join(self.missing_fields) or detail))


@dataclass(frozen=True)
class ApprovedDeploymentProfile:
    """Immutable configuration identity; never a substitute for DA authorization."""
    canonical_json: str

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise DeploymentConfigurationError(detail="profile must be a mapping")
        required = {"profile_id", "approval_id", "data_source", "artifact_root", "market",
                    "calendar_id", "timezone", "frequency", "universe_id", "adjustment"}
        missing = [k for k in required if value.get(k) is None or value.get(k) == ""]
        source = value.get("data_source")
        if not isinstance(source, Mapping):
            missing.append("data_source.type")
        else:
            missing.extend("data_source." + k for k in
                           ("dataset", "start_date", "end_date", "instrument_filter")
                           if not source.get(k))
        if missing:
            raise DeploymentConfigurationError(missing)
        unknown = set(value) - required - {"execution", "expected_snapshot_token"}
        if unknown:
            raise DeploymentConfigurationError(detail="unknown fields: " + ", ".join(sorted(unknown)))
        if any(not isinstance(value[k], str) or not value[k].strip() for k in required - {"data_source"}):
            raise DeploymentConfigurationError(detail="business identity fields must be nonempty strings")
        if value["market"] != "ashare" or value["adjustment"] != "hfq":
            raise DeploymentConfigurationError(detail="v2 default requires approved ashare/hfq semantics")
        if source.get("type") != "data_access":
            raise DeploymentConfigurationError(detail="v2 source must use the DataAccess gateway")
        # A profile label is not evidence that the selected data are adjusted.
        # This default entry currently implements the canonical daily HFQ
        # anchor only. Other sources need an explicit semantic adapter first.
        if source.get("dataset") != "ashare_stock_daily_adj":
            raise DeploymentConfigurationError(
                detail="v2 hfq source requires canonical ashare_stock_daily_adj dataset")
        if source.get("read_mode", "panel") != "panel" or value["frequency"] != "1d":
            raise DeploymentConfigurationError(detail="canonical hfq anchor requires daily 1d panel scope")
        if source.get("fields"):
            raise DeploymentConfigurationError(
                detail="v2 hfq field aliases must come from the semantic catalog, not profile overrides")
        for key in ("run_mode", "production", "strict_unknown_fields", "pit_enforce"):
            if key in source and (source[key] != "production" if key == "run_mode" else source[key] is not True):
                raise DeploymentConfigurationError(detail=f"data_source.{key} cannot weaken production gates")
        instruments = source["instrument_filter"]
        if (not isinstance(instruments, list) or not instruments or
                any(not isinstance(x, str) or not x for x in instruments) or
                len(instruments) != len(set(instruments))):
            raise DeploymentConfigurationError(detail="instrument_filter must be an explicit unique list")
        try:
            start = date.fromisoformat(source["start_date"])
            end = date.fromisoformat(source["end_date"])
        except (ValueError, TypeError):
            raise DeploymentConfigurationError(detail="source dates must be ISO dates") from None
        if start > end:
            raise DeploymentConfigurationError(detail="source start_date exceeds end_date")
        if not Path(value["artifact_root"]).is_absolute():
            raise DeploymentConfigurationError(detail="artifact_root must be an approved absolute directory")
        snapshot = value.get("expected_snapshot_token")
        if snapshot is not None and (not isinstance(snapshot, str) or not snapshot.strip()):
            raise DeploymentConfigurationError(detail="expected_snapshot_token must be a nonempty string")
        resolve_default_policy(profile=value.get("execution"))
        try:
            payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            raise DeploymentConfigurationError(detail="profile must contain finite JSON-compatible values") from None
        return cls(payload)

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def digest(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


def load_deployment_profile(path=None):
    import yaml

    selected = path if path is not None else os.environ.get("FACTOR_ENGINE_V2_PROFILE")
    if not selected:
        raise DeploymentConfigurationError(("FACTOR_ENGINE_V2_PROFILE",))
    profile_path = Path(selected)
    if not profile_path.is_absolute():
        raise DeploymentConfigurationError(detail="profile path must be absolute")
    try:
        with profile_path.open("rb") as handle:
            raw = handle.read(1048577)
        if len(raw) > 1048576:
            raise DeploymentConfigurationError(detail="profile exceeds 1 MiB")
        class UniqueLoader(yaml.SafeLoader):
            pass

        def unique_mapping(loader, node):
            output = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if not isinstance(key, str) or key in output:
                    raise DeploymentConfigurationError(detail="profile has duplicate or non-string keys")
                output[key] = loader.construct_object(value_node)
            return output

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
        # Do not implicitly turn a YAML ISO date into a non-JSON date object.
        UniqueLoader.add_constructor("tag:yaml.org,2002:timestamp", lambda loader, node: loader.construct_scalar(node))
        value = yaml.load(raw, Loader=UniqueLoader)
    except (OSError, UnicodeError) as exc:
        raise DeploymentConfigurationError(detail=f"profile cannot be read ({type(exc).__name__})") from None
    except yaml.YAMLError:
        # Parser diagnostics may contain profile contents or credentials.
        raise DeploymentConfigurationError(detail="profile contains invalid YAML/JSON") from None
    return ApprovedDeploymentProfile.from_mapping(value)


class DurableFactorEngine:
    """The v2 result is a receipt; the original FactorEngine dict API is unchanged."""

    def __init__(self, engine, deployment: ApprovedDeploymentProfile, policy: DefaultExecutionPolicy):
        self._engine = engine
        self.deployment = deployment
        self.policy = policy
        if engine.run_mode != "production":
            raise DeploymentConfigurationError(detail="v2 cannot run through research mode")
        engine.default_execution_policy = policy

    def run_many(self, factors):
        from .bounded_pipeline import execute_run_many_durable
        from .resource_broker import heavy_run_guard
        business = self.deployment.to_dict()
        root = Path(business["artifact_root"])
        if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
            raise DeploymentConfigurationError(detail="approved artifact_root must exist and be writable")
        expected = business.get("expected_snapshot_token")
        if expected is not None:
            self._engine.data_source.refresh_snapshot(force=True)
            observed = self._engine.data_source.snapshot_token
            if observed != expected:
                raise DeploymentConfigurationError(detail="approved source snapshot token does not match")
        with heavy_run_guard(timeout_seconds=self.policy.resource_wait_seconds):
            receipt = execute_run_many_durable(
                self._engine, factors, policy=self.policy, artifact_root=root,
                engine_factory=build_execution_core_from_worker_config,
                engine_factory_config=ExecutionCoreWorkerConfig(self.deployment, self.policy),
                execution_scope={k: business[k] for k in
                                 ("market", "calendar_id", "frequency", "universe_id")},
                run_kwargs={"auto_warmup": True, "trim_warmup": True,
                            "market": business["market"], "input_dq_check": True,
                            "input_dq_strict": True, "pit_enforce": True},
            )
        receipt["deployment_digest"] = self.deployment.digest
        receipt["policy_digest"] = self.policy.digest
        return receipt

    def close(self):
        close = getattr(self._engine.data_source, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_engine(*, profile_path=None, execution=None):
    """Load approved deployment once; callers need no performance parameters."""
    deployment = load_deployment_profile(profile_path)
    business = deployment.to_dict()
    policy = resolve_default_policy(execution, business.get("execution"))
    core = build_execution_core_from_worker_config(ExecutionCoreWorkerConfig(deployment, policy))
    return DurableFactorEngine(core, deployment, policy)


@dataclass(frozen=True)
class ExecutionCoreWorkerConfig:
    deployment: ApprovedDeploymentProfile
    policy: DefaultExecutionPolicy


def build_execution_core_from_worker_config(config):
    """Construct inside a spawned worker after its parent broker proxy is installed.

    No parent engine, data-source lock, connection or broker is serialized.
    The same constructor is also used by the parent-facing facade.
    """
    if not isinstance(config, ExecutionCoreWorkerConfig):
        raise TypeError("validated ExecutionCoreWorkerConfig is required")
    if not isinstance(config.deployment, ApprovedDeploymentProfile) or not isinstance(config.policy, DefaultExecutionPolicy):
        raise TypeError("worker deployment and policy must be validated")
    business = config.deployment.to_dict()
    policy = config.policy
    from factor_engine.backend.factory import build_backend
    from factor_engine.storage.factory import build_data_source, DataSourceBuildContext
    from .engine import FactorEngine
    from .resource_broker import get_v2_resource_broker
    context = DataSourceBuildContext(run_mode="production", market=business["market"],
                                     calendar_id=business["calendar_id"], timezone=business["timezone"],
                                     pit_enforce=True)
    source = build_data_source(business["data_source"], build_context=context)
    try:
        _validate_hfq_source_contract(source)
        backend = build_backend("pandas" if policy.backend == "pandas_numpy" else policy.backend)
        core = FactorEngine(backend, source, run_mode="production")
        # A failed source/backend construction must not claim the process-wide
        # policy singleton and block a later, valid deployment profile.
        broker = get_v2_resource_broker(policy)
        source.bind_resource_broker(broker)
        core.resource_broker = broker
        core.default_execution_policy = policy
        return core
    except BaseException:
        source.close()
        raise


def _validate_hfq_source_contract(source):
    """Check deployed DA contract and registry mapping, without reading prices.

    This proves routing semantics, not data correctness or permission to publish.
    Query-time catalog, snapshot, PIT and production gates still apply.
    """
    from data_access.cos_contract import get_cos_contract
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
    dataset = getattr(source, "dataset", None)
    contract = get_cos_contract(dataset)
    if (dataset != "ashare_stock_daily_adj" or contract is None or
            contract.market != "ashare" or contract.temporal_model != "D1" or
            contract.adjustment_convention != "backward_vendor_factor" or
            contract.adjustment_column != "Factor"):
        raise DeploymentConfigurationError(detail="DataAccess HFQ contract is missing or incompatible")
    for logical, physical in {"open": "AdjOpen", "high": "AdjHigh",
                              "low": "AdjLow", "close": "AdjClose"}.items():
        spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(
            "ashare", logical, table=dataset, strict=True)
        if (spec is None or spec.dataset != dataset or spec.source_name != physical or
                spec.price_basis != "ADJUSTED"):
            raise DeploymentConfigurationError(detail=f"HFQ field contract mismatch: {logical}")
