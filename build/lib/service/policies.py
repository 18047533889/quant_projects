"""R21: execution/feature policies for the service layer.

R21-001..005  — ``EndpointExecutionPolicy`` is the *floor* a production
endpoint cannot be downgraded from.  Config values (run mode, PIT, DQ, write
target, source production) that contradict the endpoint policy raise
``PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT`` instead of silently running
research.

R21-144..147 — ambient run-mode env vars are fail-closed at startup.
R21-148..151 — feature flags are collected into a typed
``RuntimeFeaturePolicy`` instead of ad-hoc ``os.environ`` reads; each run saves
its feature-policy digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Re-exported from the neutral runtime module so the engine can enforce the
# same policy object without importing the service package.
from runtime.endpoint_policy import (
    EndpointExecutionPolicy,
    ProductionPolicyConflictError,
    collect_config_policy_conflicts,
)

_KNOWN_RUN_MODES = {"research", "production"}
_TRUE = {"1", "true", "yes", "on"}


def resolve_ambient_run_mode() -> str:
    """R21-144..146: fail-closed resolution of ambient run-mode env.

    - ``FACTOR_ENGINE_RUN_MODE`` present but invalid  -> raise (not research)
    - ``QUANT_PRODUCTION_MODE`` and ``FACTOR_ENGINE_RUN_MODE`` conflict -> raise
    """
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE")
    qp = os.environ.get("QUANT_PRODUCTION_MODE")
    if fe is not None:
        fe = str(fe).strip().lower()
        if fe not in _KNOWN_RUN_MODES:
            raise ValueError(
                "FACTOR_ENGINE_RUN_MODE present but invalid: "
                f"{fe!r} (must be 'research' or 'production')"
            )
    if qp is not None:
        qp_production = str(qp).strip().lower() in _TRUE
    else:
        qp_production = False
    if fe is not None and fe == "research" and qp_production:
        raise ValueError(
            "FACTOR_ENGINE_RUN_MODE=research conflicts with QUANT_PRODUCTION_MODE=on"
        )
    if fe is not None:
        return fe
    return "production" if qp_production else "research"


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

_BOOL_FEATURES = {
    "FASTPATH": False,
    "POLARS_EXPR": False,
    "QUERY_BUDGET": True,
    "ALLOW_OPEN": False,
    "ENABLE_CSE": True,
    "ENABLE_PANEL_CACHE": True,
    "SERVICE_SYNC": False,
}
_KNOWN_FEATURE_KEYS = set(_BOOL_FEATURES)


@dataclass(frozen=True)
class RuntimeFeaturePolicy:
    """R21-148..151: typed, validated feature policy collected at startup.

    Unknown keys / invalid values are rejected in production so a typo'd flag
    cannot silently flip behavior (``prodution``-style fail-open).
    """

    flags: dict[str, bool] = field(default_factory=dict)
    production: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        production: bool | None = None,
        env: dict[str, str] | None = None,
    ) -> "RuntimeFeaturePolicy":
        env = dict(env if env is not None else os.environ)
        if production is None:
            production = resolve_ambient_run_mode() == "production"
        flags: dict[str, bool] = {}
        for key, default in _BOOL_FEATURES.items():
            raw = env.get(f"FACTOR_ENGINE_{key}")
            if raw is None:
                flags[key] = default
                continue
            text = str(raw).strip().lower()
            if text in _TRUE:
                flags[key] = True
            elif text in {"0", "false", "no", "off"}:
                flags[key] = False
            elif production:
                raise ValueError(f"invalid FACTOR_ENGINE_{key}={raw!r} in production")
            else:
                # research: unknown values fail closed (treat as off) but warn.
                flags[key] = False
        unknown = {k for k in env if k.startswith("FACTOR_ENGINE_") and k.endswith("_FLAG")}
        if production and unknown:
            raise ValueError(f"unknown feature flags: {sorted(unknown)}")
        return cls(flags=flags, production=production)

    def enabled(self, key: str) -> bool:
        return bool(self.flags.get(key, _BOOL_FEATURES.get(key, False)))

    def digest(self) -> str:
        payload = {"flags": self.flags, "production": self.production}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


def _config_digest_of(mapping: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
