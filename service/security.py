"""R21-011..018 (auth/authz) + R21-019..031 (data-source/config path hardening).

- ``_require_service_api_key`` is replaced by ``resolve_principal``: production
  compute/materialize routes are unconditionally authenticated, regardless of
  ``QUANT_PRODUCTION_MODE``.  ``X-Request-Identity`` is *never* a trusted
  principal — identity must come from an API-key→principal mapping, JWT,
  mTLS, or a reverse-proxy auth header.
- ``ApprovedSourcePolicy`` restricts remote/local data-source access to
  allowlisted profiles / hosts / roots for production (R21-020..027).
- ``authorized_config_path`` hardens symlink / TOCTOU handling (R21-028..031).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from service.errors import ServiceError

_TRUSTED_PROXY_HEADER = "X-Remote-User"


class PrincipalSource:
    API_KEY_MAPPING = "api_key_mapping"
    JWT = "jwt"
    MUTUAL_TLS = "mtls"
    REVERSE_PROXY = "reverse_proxy"
    ANONYMOUS = "anonymous"


# R21-228: RBAC privilege tiers.
ROLE_TIERS = {"READ": 1, "COMPUTE": 2, "MATERIALIZE": 3, "PUBLISH": 4, "ADMIN": 5}


@dataclass(frozen=True)
class Principal:
    identity: str
    roles: tuple[str, ...] = ("READ",)
    source: str = PrincipalSource.ANONYMOUS
    tenant: str | None = None
    project: str | None = None

    @property
    def max_role_tier(self) -> int:
        return max((ROLE_TIERS.get(r, 0) for r in self.roles), default=0)

    def has_role(self, role: str) -> bool:
        return self.max_role_tier >= ROLE_TIERS.get(role, 1)

    def to_public(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "roles": list(self.roles),
            "source": self.source,
            "tenant": self.tenant,
            "project": self.project,
        }

    def scope_key(self) -> str:
        return f"{self.identity}@{self.tenant or '-'}"

    def authenticated(self) -> bool:
        return self.source != PrincipalSource.ANONYMOUS


ANONYMOUS_PRINCIPAL = Principal(identity="anonymous", source=PrincipalSource.ANONYMOUS)


class _PrincipalRegistry:
    """Loads API-key→principal mapping once from env JSON + optional file."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, *, single_principal: bool = False):
        self._single = single_principal
        self._single_identity = os.environ.get("FACTOR_ENGINE_SERVICE_PRINCIPAL", "service")
        self._keys: dict[str, Principal] = {}
        raw = mapping if mapping is not None else self._load_env_mapping()
        self._load(raw)

    def _load_env_mapping(self) -> dict[str, Any]:
        text = os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY_MAPPING", "").strip()
        if text:
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _load(self, raw: Mapping[str, Any]) -> None:
        for key, spec in (raw or {}).items():
            if not isinstance(spec, dict):
                spec = {"identity": str(spec)}
            identity = str(spec.get("identity") or key)
            roles = tuple(str(r) for r in (spec.get("roles") or ("COMPUTE",)))
            self._keys[str(key)] = Principal(
                identity=identity,
                roles=roles,
                source=PrincipalSource.API_KEY_MAPPING,
                tenant=spec.get("tenant"),
                project=spec.get("project"),
            )

    def resolve(self, api_key: str | None, service_key: str | None) -> Principal:
        if api_key and api_key in self._keys:
            return self._keys[api_key]
        if self._single and service_key and api_key == service_key:
            return Principal(
                identity=self._single_identity,
                roles=("ADMIN",),
                source=PrincipalSource.API_KEY_MAPPING,
            )
        # A known single shared key with no explicit mapping → single-principal
        # contract (R21-018): never let the caller impersonate via header.
        if service_key and api_key == service_key:
            return Principal(
                identity=self._single_identity,
                roles=("COMPUTE", "MATERIALIZE"),
                source=PrincipalSource.API_KEY_MAPPING,
            )
        raise ServiceError("AUTH_INVALID_KEY", "invalid API key", status=401)

    def has_keys(self) -> bool:
        return bool(self._keys) or bool(os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY", "").strip())


_REGISTRY_LOCK = threading.Lock()
_registry: Optional[_PrincipalRegistry] = None


def get_principal_registry() -> _PrincipalRegistry:
    global _registry
    with _REGISTRY_LOCK:
        if _registry is None:
            _registry = _PrincipalRegistry()
        return _registry


def reset_principal_registry() -> None:
    global _registry
    with _REGISTRY_LOCK:
        _registry = None


def resolve_principal(
    request: Any,
    *,
    service_key: str | None = None,
    required: bool = False,
) -> Principal:
    """Resolve a trusted principal for a request.

    Trust sources, in order: API-key mapping > single shared key > JWT
    (``Authorization: Bearer`` verified against a configured shared secret) >
    reverse-proxy ``X-Remote-User`` (only when explicitly enabled) > anonymous.
    ``X-Request-Identity`` is intentionally ignored as a trust source.
    """
    if service_key is None:
        service_key = os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY", "").strip() or None
    headers = dict(getattr(request, "headers", {}) or {})
    supplied = str(headers.get("x-api-key") or headers.get("X-API-Key") or "")

    reg = get_principal_registry()
    if service_key or reg.has_keys():
        try:
            principal = reg.resolve(supplied, service_key)
            return principal
        except ServiceError:
            if required:
                raise
    else:
        # No key configured: production routes are always authenticated, so the
        # caller must have passed a key (handled by caller via `required`).
        if required:
            raise ServiceError("AUTH_REQUIRED", "authentication required", status=401)

    # JWT bearer (shared-secret HMAC, R21-015).
    auth_header = str(headers.get("authorization") or headers.get("Authorization") or "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        principal = _resolve_jwt(token)
        if principal is not None:
            return principal

    # Reverse-proxy authenticated identity (only when explicitly trusted).
    proxy_trusted = os.environ.get("FACTOR_ENGINE_SERVICE_TRUST_PROXY_AUTH", "").lower() in {
        "1", "true", "yes",
    }
    remote_user = headers.get(_TRUSTED_PROXY_HEADER.lower()) or headers.get(_TRUSTED_PROXY_HEADER)
    if proxy_trusted and remote_user:
        return Principal(
            identity=str(remote_user),
            roles=("COMPUTE",),
            source=PrincipalSource.REVERSE_PROXY,
        )
    if required:
        raise ServiceError("AUTH_REQUIRED", "authentication required", status=401)
    return ANONYMOUS_PRINCIPAL


def _resolve_jwt(token: str) -> Principal | None:
    secret = os.environ.get("FACTOR_ENGINE_SERVICE_JWT_SECRET", "").strip()
    if not secret:
        return None
    import base64
    import hmac

    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        header, payload, sig = parts
        expected = base64.urlsafe_b64encode(
            hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(sig, expected):
            return None
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return None
    if not isinstance(claims, dict) or not claims.get("sub"):
        return None
    return Principal(
        identity=str(claims["sub"]),
        roles=tuple(claims.get("roles") or ("COMPUTE",)),
        source=PrincipalSource.JWT,
        tenant=claims.get("tenant"),
        project=claims.get("project"),
    )


def require_privilege(principal: Principal, role: str) -> None:
    if not principal.has_role(role):
        raise ServiceError(
            "AUTHORIZATION_DENIED",
            f"principal {principal.identity!r} lacks required role {role}",
            status=403,
        )


# ---------------------------------------------------------------------------
# Approved source policy (R21-020..027)
# ---------------------------------------------------------------------------

ALLOWED_SOURCE_TYPES_PRODUCTION = {"data_access", "composite", "long_table"}


@dataclass(frozen=True)
class ApprovedSourcePolicy:
    """Policy object enforcing production source allowlisting.

    Attributes:
        approved_profiles: mapping ``profile_id -> {"dataset": str}`` for
            DataAccess-backed sources (R21-020/026).
        approved_datasets: set of DataAccess dataset ids allowed in production.
        allowed_clickhouse_hosts: host allowlist (R21-021).
        allowed_clickhouse_tables: ``host:database:table`` allowlist.
        approved_roots: filesystem roots allowed for parquet sources (R21-023/024).
    """

    approved_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    approved_datasets: frozenset[str] = frozenset()
    allowed_clickhouse_hosts: frozenset[str] = frozenset()
    allowed_clickhouse_tables: frozenset[str] = frozenset()
    approved_roots: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ApprovedSourcePolicy":
        env = dict(env if env is not None else os.environ)
        profiles: dict[str, dict[str, Any]] = {}
        text = env.get("FACTOR_ENGINE_SERVICE_SOURCE_PROFILES", "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    profiles = parsed
            except json.JSONDecodeError:
                profiles = {}

        def _split(key: str) -> frozenset[str]:
            raw = env.get(key, "").strip()
            return frozenset(x.strip() for x in raw.split(",") if x.strip())

        datasets = _split("FACTOR_ENGINE_APPROVED_DATASETS")
        hosts = _split("FACTOR_ENGINE_APPROVED_CLICKHOUSE_HOSTS")
        tables = _split("FACTOR_ENGINE_APPROVED_CLICKHOUSE_TABLES")
        roots = frozenset(
            os.path.expanduser(x.strip()) for x in env.get("FACTOR_ENGINE_APPROVED_ROOTS", "").split(",") if x.strip()
        )
        return cls(
            approved_profiles=profiles,
            approved_datasets=datasets,
            allowed_clickhouse_hosts=hosts,
            allowed_clickhouse_tables=tables,
            approved_roots=roots,
        )

    def validate_source(self, source_config: dict[str, Any], *, production: bool) -> None:
        """Validate a data-source config against this policy (R21-027)."""
        if not production:
            return
        source_type = str(source_config.get("type") or "")
        if source_type == "composite":
            subs = source_config.get("sources") or {}
            if isinstance(subs, dict):
                for sub in subs.values():
                    if isinstance(sub, dict):
                        self.validate_source(sub, production=True)
            return
        if source_type == "data_access":
            dataset = str(source_config.get("dataset") or "")
            if dataset not in self.approved_datasets and not any(
                p.get("dataset") == dataset for p in self.approved_profiles.values()
            ):
                raise ServiceError(
                    "SOURCE_EMPTY",
                    f"production data_access dataset {dataset!r} is not in approved datasets",
                    status=422,
                )
            return
        if source_type == "clickhouse":
            host = str(source_config.get("host") or "")
            table = str(source_config.get("table") or "")
            database = str(source_config.get("database") or "")
            if self.allowed_clickhouse_hosts and host not in self.allowed_clickhouse_hosts:
                raise ServiceError(
                    "UNAPPROVED_REMOTE_SOURCE",
                    f"ClickHouse host {host!r} not allowed in production",
                    status=422,
                )
            if self.allowed_clickhouse_tables and f"{host}:{database}:{table}" not in self.allowed_clickhouse_tables:
                raise ServiceError(
                    "UNAPPROVED_REMOTE_SOURCE",
                    f"ClickHouse table {host}:{database}:{table} not allowed in production",
                    status=422,
                )
            return
        if source_type in {"parquet", "multi_parquet", "parquet_kline", "cleaned_parquet"}:
            root = str(source_config.get("root") or "")
            if self.approved_roots and not self._within_any_root(root):
                raise ServiceError(
                    "UNAPPROVED_LOCAL_PATH",
                    f"parquet root {root!r} is not within an approved data root",
                    status=422,
                )
            return
        if source_type in {"intraday_daily", "long_table"}:
            inner = source_config.get("source") or source_config.get("inner")
            if isinstance(inner, dict):
                self.validate_source(inner, production=True)
            return
        raise ServiceError(
            "UNAPPROVED_REMOTE_SOURCE",
            f"source type {source_type!r} not allowed for production",
            status=422,
        )

    def _within_any_root(self, path: str) -> bool:
        from pathlib import PurePosixPath

        try:
            candidate = Path(os.path.expandvars(path)).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        for root in self.approved_roots:
            try:
                r = Path(root).resolve()
            except (OSError, RuntimeError):
                continue
            if candidate == r or r in candidate.parents:
                return True
        return False

    def digest(self) -> str:
        payload = {
            "profiles": self.approved_profiles,
            "datasets": sorted(self.approved_datasets),
            "hosts": sorted(self.allowed_clickhouse_hosts),
            "tables": sorted(self.allowed_clickhouse_tables),
            "roots": sorted(self.approved_roots),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


def resolve_source_profile(payload: dict[str, Any], *, policy: ApprovedSourcePolicy) -> dict[str, Any]:
    """R21-020: production HTTP may only reference ``approved_source_profile_id``.

    Returns the DataAccess source config for the profile.  Raw host/credentials
    from the request are never accepted for production.
    """
    profile_id = str(payload.get("approved_source_profile_id") or "").strip()
    if not profile_id:
        raise ServiceError(
            "UNAPPROVED_REMOTE_SOURCE",
            "production compute requires approved_source_profile_id",
            status=422,
        )
    profile = policy.approved_profiles.get(profile_id)
    if profile is None:
        raise ServiceError(
            "UNAPPROVED_REMOTE_SOURCE",
            f"unknown approved source profile: {profile_id!r}",
            status=422,
        )
    dataset = profile.get("dataset")
    if not dataset:
        raise ServiceError(
            "UNAPPROVED_REMOTE_SOURCE",
            f"approved source profile {profile_id!r} has no dataset",
            status=422,
        )
    return {"type": "data_access", "dataset": str(dataset)}


# ---------------------------------------------------------------------------
# Config path hardening (R21-028..031)
# ---------------------------------------------------------------------------


def _has_symlink_component(path: Path) -> bool:
    """lstat() every component of the *original* path (not the resolved one)."""
    try:
        remaining = Path(path)
    except (OSError, RuntimeError):
        return True
    parts: list[Path] = []
    while True:
        parts.append(remaining)
        parent = remaining.parent
        if parent == remaining:
            break
        remaining = parent
    for part in reversed(parts):
        try:
            st = part.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if stat.S_ISLNK(st.st_mode):
            return True
    return False


def authorized_config_path(raw: Any, *, root: Path | None = None) -> Path:
    """Return a config path inside ``root`` with no symlink components.

    R21-028..030: symlink check runs on the original path's components via
    lstat(); ``resolve()`` alone would follow symlinks and hide their identity.
    """
    base = (root or Path(os.environ.get("FACTOR_ENGINE_CONFIG_ROOT") or "configs")).resolve()
    path = Path(str(raw))
    if not path.is_absolute():
        candidate = (base / path)
    else:
        candidate = path
    if _has_symlink_component(candidate):
        raise ValueError("config_path symlinks are forbidden")
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("config_path is outside FACTOR_ENGINE_CONFIG_ROOT")
    return resolved
