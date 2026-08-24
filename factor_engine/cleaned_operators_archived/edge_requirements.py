# -*- coding: utf-8 -*-
"""Operator-specific IEEE edge evidence requirements.

This module does not manufacture certification. It reports which production
operators still lack required NaN/Inf/zero/domain edge evidence so routing can
remain fail-closed.

review #5 R5-15: the previous gate was a *vacuous pass* — an operator with no
declared required edge dimensions returned ``production_edge_evidence_complete
== True`` ("nothing required, therefore passed") without anyone deciding it is
edge-insensitive.  The gate now exposes an explicit tri-state
(:func:`edge_evidence_status`) — ``complete`` / ``incomplete`` / ``undeclared``.

review #250: production has NO vacuous pass for undeclared operators.  The
runtime ``edge_gate_strict`` flag is gone; in its place ``EdgeGateMode``
distinguishes *production* (undeclared == FAIL) from *research* (undeclared ==
warning only).  ``production_edge_evidence_complete`` defaults to production
mode so every production certification path fails closed until an operator is
declared (legacy required set or EDGE_IMMUNE) AND its edge evidence is complete.

review #251: :class:`EdgeContract` is the declarative edge contract.  The
legacy ``NAN_REQUIRED`` / ``INF_REQUIRED`` membership is treated as the declared
behavior (``nan="propagate"`` / ``pos_inf=neg_inf="propagate"``) and
``EDGE_IMMUNE`` as ``ignore`` for every dimension.  Certification reads the
contract instead of a growing manual list.

review #252: edge evidence is backend-specific.  Evidence is keyed
``{"edge_evidence": {"pandas_numpy": {...}, "polars": {...}, "duckdb": {...}}}``
and every gate accepts a ``backend`` parameter.  R34 P0-035: the DEFAULT is
``None`` — the canonical semantic-layer edge gate unions over the operator's
applicable backends and is NOT bound to duckdb.  Explicit ``backend="duckdb"``
performs per-backend parity checks (legacy flat ``duckdb_nan_edge_verified`` /
``duckdb_inf_edge_verified`` lists still route through it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

# Operators whose output must be validated against NaN inputs (every numeric
# reducer / windowed statistic / conditional that mixes NaN and finite values).
NAN_REQUIRED = frozenset({
    "cs_mean", "cs_std", "cs_sum", "normalize", "zscore", "scale", "winsorize",
    "group_mean", "group_std", "group_zscore", "group_normalize", "group_rank",
    "ts_mean", "ts_std", "ts_var", "ts_corr", "ts_cov", "ts_beta", "ts_zscore",
    "ts_sharpe", "volatility", "maximum", "minimum", "where", "coalesce",
    # R5-15: arithmetic / ratio / log ops are NaN-sensitive too.
    "add", "subtract", "multiply", "divide", "protected_div", "safe_div",
    "log", "log_returns", "ts_pct", "returns", "ratio",
})

# Operators whose output can be corrupted by Inf (Inf propagates into a finite
#-looking average / slope / ratio and must be fail-closed, not silently mixed).
INF_REQUIRED = frozenset({
    "divide", "protected_div", "safe_div", "ratio", "ts_pct", "returns",
    "log", "log_returns", "ts_mean", "ts_std", "ts_beta", "ts_corr",
    "volatility", "zscore", "normalize", "winsorize",
    # cs_mean/c_mean: an Inf input must not silently produce a finite-looking
    # cross-sectional average — fail-closed requires inf-edge evidence too.
    "cs_mean",
})

# Operators explicitly declared edge-insensitive: they only move/reshape values
# or compare booleans and cannot turn NaN/Inf into a wrong finite output, so no
# edge evidence is required.  Anything NOT here (and not in the required sets)
# is *undeclared* — a fail-closed state under production mode, never a pass.
EDGE_IMMUNE = frozenset({
    "reindex", "shift_forward", "alias", "identity", "first_not_null",
    "last_not_null", "cum_sum", "cum_prod", "cum_max", "cum_min",
})

# The edge dimensions a contract can declare.  Ordering is the canonical order.
EDGE_DIMENSIONS = ("nan", "pos_inf", "neg_inf", "zero", "domain_invalid")

# Valid declared behaviors per dimension.
EDGE_BEHAVIORS = ("propagate", "ignore", "break", "invalid")


class EdgeGateMode(Enum):
    """Distinguish production from research edge gating.

    * ``PRODUCTION`` — an undeclared operator FAILS the edge gate (no vacuous
      pass).  This is the default for :func:`production_edge_evidence_complete`.
    * ``RESEARCH`` — an undeclared operator is a warning only (still
      ``undeclared`` in :func:`edge_evidence_status`, but the gate does not
      fail on it).
    """

    PRODUCTION = "production"
    RESEARCH = "research"


@dataclass(frozen=True)
class EdgeContract:
    """Declared IEEE edge behavior for one operator.

    Each dimension is one of ``EDGE_BEHAVIORS``:

    * ``propagate`` — NaN/Inf/zero/domain-invalid input must propagate to the
      output; evidence must verify the propagated value is produced.
    * ``ignore`` — the operator is edge-insensitive for this dimension
      (EDGE_IMMUNE); no evidence required.
    * ``break`` — the operator raises / returns invalid on this input; evidence
      must verify the fail behavior.
    * ``invalid`` — the dimension is not part of the operator's domain
      (undeclared for that dimension).

    A contract with every dimension ``invalid`` is *undeclared*
    (:attr:`declared` is False) and fails closed under
    :data:`EdgeGateMode.PRODUCTION`.

    R40 #255：新增 signed-zero / subnormal / overflow 三档数值边沿策略，进入
    numeric identity —— NaN 的 sign-bit、subnormal 归一、overflow 处理任一不同
    都会改变 factor 数值身份。
    """

    nan: str = "invalid"
    pos_inf: str = "invalid"
    neg_inf: str = "invalid"
    zero: str = "invalid"
    domain_invalid: str = "invalid"
    # R40 #255: +0.0 vs -0.0 —— "preserve" 保留符号位 / "normalize" 归一到 +0。
    signed_zero_policy: str = "preserve"
    # R40 #255: subnormal —— "preserve" 保留 / "flush_to_zero" 归零。
    subnormal_policy: str = "preserve"
    # R40 #255: overflow —— "preserve" 保留 Inf / "default" 回填默认 / "null"。
    overflow_policy: str = "preserve"

    def __post_init__(self) -> None:
        for dim in EDGE_DIMENSIONS:
            value = getattr(self, dim)
            if value not in EDGE_BEHAVIORS:
                raise ValueError(
                    f"EdgeContract.{dim}={value!r} not in {EDGE_BEHAVIORS}"
                )
        if self.signed_zero_policy not in {"preserve", "normalize"}:
            raise ValueError(f"signed_zero_policy must be preserve|normalize, got {self.signed_zero_policy!r}")
        if self.subnormal_policy not in {"preserve", "flush_to_zero"}:
            raise ValueError(f"subnormal_policy must be preserve|flush_to_zero, got {self.subnormal_policy!r}")
        if self.overflow_policy not in {"preserve", "default", "null"}:
            raise ValueError(f"overflow_policy must be preserve|default|null, got {self.overflow_policy!r}")

    @property
    def declared(self) -> bool:
        return any(getattr(self, dim) != "invalid" for dim in EDGE_DIMENSIONS)

    @property
    def required_dimensions(self) -> frozenset[str]:
        """Dimensions whose behavior must be verified (propagate/break)."""
        return frozenset(
            dim
            for dim in EDGE_DIMENSIONS
            if getattr(self, dim) in ("propagate", "break")
        )


def edge_contract_numeric_identity(canonical: str) -> dict[str, str]:
    """R40 #255：EdgeContract 的数值边沿策略进入 factor numeric identity。

    返回 ``{"signed_zero_policy": ..., "subnormal_policy": ...,
    "overflow_policy": ...}``（默认 preserve/preserve/preserve）。
    """
    contract = edge_contract(canonical)
    if contract is None:
        return {
            "signed_zero_policy": "preserve",
            "subnormal_policy": "preserve",
            "overflow_policy": "preserve",
        }
    return {
        "signed_zero_policy": contract.signed_zero_policy,
        "subnormal_policy": contract.subnormal_policy,
        "overflow_policy": contract.overflow_policy,
    }


def _factor_engine_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(canonical: str) -> str:
    try:
        from cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(canonical)
    except Exception:
        return canonical


@lru_cache(maxsize=1)
def load_primitive_evidence() -> dict:
    paths = sorted(_factor_engine_root().rglob("primitive_verified.json"))
    if not paths:
        return {}
    return json.loads(paths[0].read_text(encoding="utf-8"))


def edge_requirements_declared(canonical: str) -> bool:
    return edge_contract(canonical) is not None


def required_edge_dimensions(canonical: str) -> frozenset[str]:
    """Backward-compatible legacy dimension set.

    Returns ``{"nan"}`` / ``{"inf"}`` per membership in the legacy required
    sets.  New callers should read :func:`edge_contract` instead.
    """
    name = _resolve(canonical)
    required: set[str] = set()
    if name in NAN_REQUIRED:
        required.add("nan")
    if name in INF_REQUIRED:
        required.add("inf")
    return frozenset(required)


def edge_contract(canonical: str) -> EdgeContract | None:
    """Return the declared :class:`EdgeContract` for an operator.

    Layers on top of the legacy lists without deleting them:

    * ``NAN_REQUIRED`` membership → ``nan="propagate"``;
    * ``INF_REQUIRED`` membership → ``pos_inf="propagate"``, ``neg_inf="propagate"``;
    * ``EDGE_IMMUNE`` membership → ``"ignore"`` for every dimension;
    * otherwise ``None`` (undeclared).

    Returns ``None`` for an undeclared operator so callers fail closed instead
    of inventing a contract.
    """
    name = _resolve(canonical)
    if name in EDGE_IMMUNE:
        return EdgeContract(
            nan="ignore",
            pos_inf="ignore",
            neg_inf="ignore",
            zero="ignore",
            domain_invalid="ignore",
        )
    nan = "propagate" if name in NAN_REQUIRED else "invalid"
    inf = "propagate" if name in INF_REQUIRED else "invalid"
    if nan == "invalid" and inf == "invalid":
        return None
    return EdgeContract(nan=nan, pos_inf=inf, neg_inf=inf)


def _backend_edge_verified(payload: dict, backend: str) -> dict[str, frozenset[str]]:
    """Extract per-backend verified edge sets from an evidence payload.

    Prefers the review #252 nested schema::

        {"edge_evidence": {
            "pandas_numpy": {"nan_verified": [...], "inf_verified": [...], ...},
            "polars": {...}, "duckdb": {...}}}

    Falls back to the legacy flat keys (``duckdb_nan_edge_verified`` /
    ``duckdb_inf_edge_verified`` and the collective ``polars_edge_verified``)
    so pre-regeneration evidence still routes.
    """
    edge_evidence = payload.get("edge_evidence")
    if isinstance(edge_evidence, dict) and isinstance(edge_evidence.get(backend), dict):
        be = edge_evidence[backend]
        out: dict[str, frozenset[str]] = {}
        for dim, key in _EDGE_EVIDENCE_KEYS.items():
            out[dim] = frozenset(str(x) for x in (be.get(key) or []))
        return out
    # Legacy fallback.
    if backend == "duckdb":
        return {
            "nan": frozenset(str(x) for x in (payload.get("duckdb_nan_edge_verified") or [])),
            "inf": frozenset(str(x) for x in (payload.get("duckdb_inf_edge_verified") or [])),
            "zero": frozenset(),
            "domain_invalid": frozenset(),
        }
    if backend == "polars":
        edge = frozenset(str(x) for x in (payload.get("polars_edge_verified") or []))
        return {
            "nan": edge,
            "inf": edge,
            "zero": frozenset(),
            "domain_invalid": frozenset(),
        }
    return {"nan": frozenset(), "inf": frozenset(), "zero": frozenset(), "domain_invalid": frozenset()}


_EDGE_EVIDENCE_KEYS = {
    "nan": "nan_verified",
    "inf": "inf_verified",
    "zero": "zero_verified",
    "domain_invalid": "domain_invalid_verified",
}


def backend_edge_evidence(
    canonical: str, backend: str = "duckdb", evidence: dict | None = None
) -> dict[str, frozenset[str]]:
    """Return the verified edge-name sets for one backend.

    ``canonical`` is resolved to its final name, but the verified sets are
    whole-registry lists so the name is passed through to match members.
    """
    name = _resolve(canonical)
    payload = evidence if evidence is not None else load_primitive_evidence()
    sets = _backend_edge_verified(payload, backend)
    # Keep dimension names present even when empty, for stable callers.
    return {dim: sets.get(dim, frozenset()) for dim in ("nan", "inf", "zero", "domain_invalid")}


def missing_edge_dimensions(
    canonical: str, evidence: dict | None = None, backend: str = "duckdb"
) -> frozenset[str]:
    """Dimensions that lack verified edge evidence on ``backend``.

    Returns members of ``{"nan", "inf", "zero", "domain_invalid"}``.  An
    undeclared operator returns ``frozenset()`` here — undeclared is a *gating*
    state handled by :func:`edge_evidence_status` /
    :func:`production_edge_evidence_complete`, never a silent pass.
    """
    name = _resolve(canonical)
    contract = edge_contract(name)
    if contract is None:
        return frozenset()
    verified = backend_edge_evidence(name, backend, evidence)
    missing: set[str] = set()
    if contract.nan in ("propagate", "break") and name not in verified["nan"]:
        missing.add("nan")
    if (
        contract.pos_inf in ("propagate", "break")
        or contract.neg_inf in ("propagate", "break")
    ) and name not in verified["inf"]:
        missing.add("inf")
    if contract.zero in ("propagate", "break") and name not in verified["zero"]:
        missing.add("zero")
    if (
        contract.domain_invalid in ("propagate", "break")
        and name not in verified["domain_invalid"]
    ):
        missing.add("domain_invalid")
    return frozenset(missing)


def _all_backend_names() -> tuple[str, ...]:
    """Backends with a verified-edge evidence namespace in primitive evidence."""
    return ("pandas_numpy", "polars", "duckdb")


def _applicable_backends(canonical: str) -> tuple[str, ...]:
    """Backends the canonical actually registers, restricted to evidence namespaces.

    ``backend=None`` 的语义层 union 只遍历算子真实可执行的后端——否则一个只跑
    Pandas 的算子会被 duckdb 的证据误判 complete。
    """
    try:
        from cleaned_operators.registry import OperatorRegistry

        registered = OperatorRegistry.backends_for(canonical)
    except Exception:
        registered = set()
    names = {b for b in _all_backend_names() if b in registered}
    # 至少保留参考后端，避免空集导致恒 incomplete
    if not names:
        names = {"pandas_numpy"}
    return tuple(sorted(names))


def edge_evidence_status(
    canonical: str, evidence: dict | None = None, backend: str | None = None
) -> str:
    """Honest tri-state edge gate.

    R34 P0-035：``backend=None``（默认）表示**跨算子注册的所有后端取并集**——
    canonical 语义层的 edge gate 不绑任何单一 backend（尤其不默认 duckdb）。
    每个 required 维度只要有任一适用 backend 验证过即算语义 complete；逐 backend
    的 parity 由 backend certification 单独检查。

    * ``complete`` — the operator is declared (required set or EDGE_IMMUNE) and
      every required dimension has verified edge evidence on the requested
      backend (or, when ``backend is None``, on at least one applicable backend);
    * ``incomplete`` — declared, but at least one required dimension lacks
      evidence everywhere;
    * ``undeclared`` — the operator has no declared edge contract; under
      production mode this is a fail-closed state, never a pass.
    """
    name = _resolve(canonical)
    if edge_contract(name) is None:
        return "undeclared"
    if backend is None:
        # Semantic layer: a required dimension is covered if ANY applicable
        # backend (the ones the operator actually registers) verifies it.
        # Per-backend compliance is a separate gate.
        for candidate in _applicable_backends(name):
            if not missing_edge_dimensions(name, evidence, candidate):
                return "complete"
        return "incomplete"
    missing = missing_edge_dimensions(name, evidence, backend)
    if missing:
        return "incomplete"
    return "complete"


def production_edge_evidence_complete(
    canonical: str,
    evidence: dict | None = None,
    backend: str | None = None,
    mode: EdgeGateMode = EdgeGateMode.PRODUCTION,
) -> bool:
    """Whether the operator's edge evidence is production-complete.

    R34 P0-035：默认 ``backend=None``（语义层，跨适用后端取并集），不再默认
    绑定 duckdb。显式传 ``backend="duckdb"`` 时才做逐 backend parity 判定。

    Defaults to :data:`EdgeGateMode.PRODUCTION` — an undeclared operator FAILS
    (no vacuous pass).  ``EdgeGateMode.RESEARCH`` demotes undeclared to a
    warning (returns True) for research-only tooling; production certification
    must never pass ``mode=RESEARCH``.
    """
    status = edge_evidence_status(canonical, evidence, backend)
    if status == "complete":
        return True
    if status == "incomplete":
        return False
    # undeclared
    if mode is EdgeGateMode.PRODUCTION:
        return False
    return True
