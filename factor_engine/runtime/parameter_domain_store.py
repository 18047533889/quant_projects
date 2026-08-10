# -*- coding: utf-8 -*-
"""R37-P0-006/007/008/009：ParameterDomainCertificationStore。

R34 参数域认证的缺陷（R37 §4）：只认证 canonical 默认参数、只产离线 JSON/CSV，
production runtime 无法查询「这个具体调用参数点是否被认证」。R37 修正：

1. 认证 key 至少包含 canonical + semantic_version + backend + execution_variant +
   source_context + parameter_point + dtype + grain（P0-006/008）；
2. 区分 ``operator_has_any_certified_region`` 与 ``exact_call_is_certified``
   —— production 准入只看后者（P0-007），绝不因"某测试点至少一个通过"把整个
   operator 标成 certified；
3. production compile/execute 前真正调用 ``assert_parameter_point_certified``：
   production + uncertified point => fail closed；research => allow + telemetry
   （P0-009）。

本模块不重复制造 truth source：认证证据来自独立 oracle（R37-P0-003），由
``scripts/audit_r37_parameter_domains.py`` 写入，本 store 只做持久化 + membership
查询。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging

_log = logging.getLogger(__name__)

FE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CertificationKey:
    """一个参数点认证的唯一身份（P0-006 认证 key 的最小集合）。"""

    canonical: str
    semantic_version: str = ""
    backend: str = "pandas_numpy"
    execution_variant: str = "reference"
    source_context: str = ""
    parameter_point: tuple[tuple[str, Any], ...] = ()  # sorted (name, value)
    dtype: str = "float64"
    grain: str = "daily"

    def to_key(self) -> str:
        params = ",".join(f"{k}={v!r}" for k, v in self.parameter_point)
        return "|".join([
            self.canonical, self.semantic_version, self.backend,
            self.execution_variant, self.source_context, params, self.dtype, self.grain,
        ])

    @classmethod
    def from_kwargs(cls, canonical: str, kwargs: dict[str, Any],
                    **over: Any) -> "CertificationKey":
        return cls(
            canonical=canonical,
            parameter_point=tuple(sorted((k, v) for k, v in kwargs.items() if not k.startswith("_"))),
            **{k: v for k, v in over.items() if v is not None},
        )

    def normalized(self) -> str:
        """参数点正则化 key：同一参数域（canonical+backend+source）下不同
        parameter_point 视为不同认证点；缺省参数仅指已显式传入的。"""
        return self.to_key()


@dataclass
class CertifiedPoint:
    key: CertificationKey
    passed: bool
    evidence_hash: str = ""
    source: str = ""          # 独立 oracle / property invariant / analytic fixture
    details: dict = field(default_factory=dict)


class ParameterDomainCertificationStore:
    """进程内可查询的参数域认证 store + 持久化。

    ``certify_point`` 记录一个精确调用点的认证结果；``exact_call_is_certified``
    做 membership 查询；``operator_has_any_certified_region`` 是弱查询（P0-007
    明确禁止用它在 production 准入）——production 只看 ``exact_call_is_certified``。
    """

    def __init__(self) -> None:
        self._points: dict[str, CertifiedPoint] = {}
        self._lock = threading.RLock()

    # -- 写入 --

    def certify_point(self, key: CertificationKey, passed: bool, *,
                      evidence_hash: str = "", source: str = "",
                      details: dict | None = None) -> None:
        with self._lock:
            self._points[key.to_key()] = CertifiedPoint(
                key=key, passed=passed, evidence_hash=evidence_hash,
                source=source, details=details or {},
            )

    def load_json(self, path: Path) -> int:
        """从 audit 输出的 JSON 加载认证点。返回加载数量。"""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        points = data.get("certified_points") or data.get("points") or []
        n = 0
        for p in points:
            key = CertificationKey(**{k: p[k] for k in
                                      ("canonical", "semantic_version", "backend",
                                       "execution_variant", "source_context", "dtype", "grain")
                                      if k in p})
            if p.get("parameter_point"):
                key = CertificationKey(
                    canonical=key.canonical, semantic_version=key.semantic_version,
                    backend=key.backend, execution_variant=key.execution_variant,
                    source_context=key.source_context,
                    parameter_point=tuple(tuple(x) for x in p["parameter_point"]),
                    dtype=key.dtype, grain=key.grain,
                )
            self.certify_point(key, bool(p.get("passed", True)),
                               evidence_hash=p.get("evidence_hash", ""),
                               source=p.get("source", ""),
                               details=p.get("details"))
            n += 1
        return n

    def save_json(self, path: Path) -> None:
        rows = []
        with self._lock:
            for cp in self._points.values():
                rows.append({
                    "canonical": cp.key.canonical,
                    "semantic_version": cp.key.semantic_version,
                    "backend": cp.key.backend,
                    "execution_variant": cp.key.execution_variant,
                    "source_context": cp.key.source_context,
                    "parameter_point": list(cp.key.parameter_point),
                    "dtype": cp.key.dtype,
                    "grain": cp.key.grain,
                    "passed": cp.passed,
                    "evidence_hash": cp.evidence_hash,
                    "source": cp.source,
                    "details": cp.details,
                })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"certified_points": rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # -- 查询 --

    def exact_call_is_certified(
        self, canonical: str, kwargs: dict[str, Any], *,
        semantic_version: str = "", backend: str = "pandas_numpy",
        execution_variant: str = "reference", source_context: str = "",
        dtype: str = "float64", grain: str = "daily",
    ) -> bool:
        """P0-007：只有精确调用点被认证（且 passed）才算 certified。

        参数缺失（部分 kwargs）不匹配任何认证点 => False（fail closed）。
        这是 production 准入的**唯一**依据。
        """
        key = CertificationKey(
            canonical=canonical, semantic_version=semantic_version, backend=backend,
            execution_variant=execution_variant, source_context=source_context,
            parameter_point=tuple(sorted((k, v) for k, v in kwargs.items())),
            dtype=dtype, grain=grain,
        )
        with self._lock:
            cp = self._points.get(key.to_key())
        return bool(cp and cp.passed)

    def operator_has_any_certified_region(self, canonical: str) -> bool:
        """弱查询：该 operator 至少有一个 passed 认证点。禁止用于 production 准入。"""
        with self._lock:
            return any(cp.passed and cp.key.canonical == canonical
                       for cp in self._points.values())

    def certified_parameter_points(self, canonical: str) -> list[CertifiedPoint]:
        with self._lock:
            return [cp for cp in self._points.values()
                    if cp.key.canonical == canonical and cp.passed]

    def all_passed_certified_points(self) -> list[CertifiedPoint]:
        with self._lock:
            return [cp for cp in self._points.values() if cp.passed]

    def operator_has_any_certified_region_by_backend(
        self, canonical: str, backend: str) -> bool:
        with self._lock:
            return any(cp.passed and cp.key.canonical == canonical
                       and cp.key.backend == backend for cp in self._points.values())


# ---------------------------------------------------------------------------
# production / research 准入
# ---------------------------------------------------------------------------

_GLOBAL_STORE: ParameterDomainCertificationStore | None = None
_STORE_LOCK = threading.Lock()


def get_parameter_domain_store() -> ParameterDomainCertificationStore:
    """进程级 singleton store。"""
    global _GLOBAL_STORE
    with _STORE_LOCK:
        if _GLOBAL_STORE is None:
            _GLOBAL_STORE = ParameterDomainCertificationStore()
        return _GLOBAL_STORE


def reset_parameter_domain_store() -> None:
    global _GLOBAL_STORE
    with _STORE_LOCK:
        _GLOBAL_STORE = None


def _is_production() -> bool:
    try:
        from runtime.production_policy import is_production_mode

        return is_production_mode()
    except Exception:
        return False


def assert_parameter_point_certified(
    canonical: str, kwargs: dict[str, Any], *,
    semantic_version: str = "", backend: str = "pandas_numpy",
    execution_variant: str = "reference", source_context: str = "",
    dtype: str = "float64", grain: str = "daily",
    store: ParameterDomainCertificationStore | None = None,
) -> None:
    """R37-P0-009：production compile/execute 前的参数域 membership 门。

    production + uncertified point => raise ``ParameterDomainError``（fail closed）
    research + uncertified point => allow + warning（telemetry）
    """
    store = store or get_parameter_domain_store()
    certified = store.exact_call_is_certified(
        canonical, kwargs, semantic_version=semantic_version, backend=backend,
        execution_variant=execution_variant, source_context=source_context,
        dtype=dtype, grain=grain,
    )
    if certified:
        return
    from runtime.exceptions import ParameterDomainError

    production = _is_production()
    msg = (
        f"parameter point not certified: canonical={canonical!r} kwargs={kwargs!r} "
        f"backend={backend!r} variant={execution_variant!r} "
        f"source_context={source_context!r} grain={grain!r} "
        f"(production={production})"
    )
    if production:
        raise ParameterDomainError(msg)
    _log.warning("R37-P0-009 research telemetry: %s", msg)
