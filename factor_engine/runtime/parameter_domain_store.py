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

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging

_log = logging.getLogger(__name__)

FE_ROOT = Path(__file__).resolve().parents[1]

#: 默认参数域认证证据（R37 独立 oracle 生成，current-SHA evidence）。
#: 生产启动/执行期通过它装载，禁止空 store 或旧 SHA store 悄悄存在（R39 #29）。
DEFAULT_PARAMETER_DOMAIN_EVIDENCE = FE_ROOT / "docs" / "evidence" / "r37" / "R37_PARAMETER_DOMAIN_STORE.json"


def _current_head() -> str:
    """当前 git HEAD（生成证据时记录；运行时检测旧 SHA 证据）。非 git 部署返回 ""。"""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(FE_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


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
        # R39 #29：装载/freshness/freeze 状态（执行期禁止空 store 或旧 SHA store）。
        self._loaded = False
        self._frozen = False
        self._source_path: str = ""
        self._evidence_hash: str = ""
        self._generated_commit: str = ""

    # -- 证据完整性（R39 #29） --

    @staticmethod
    def _canonical_row_key(cp: "CertifiedPoint") -> str:
        params = ",".join(f"{k}={v!r}" for k, v in cp.key.parameter_point)
        return "|".join([
            cp.key.canonical, cp.key.semantic_version, cp.key.backend,
            cp.key.execution_variant, cp.key.source_context, params,
            cp.key.dtype, cp.key.grain, "1" if cp.passed else "0",
        ])

    def compute_evidence_hash(self) -> str:
        """对全部认证点做确定性 sha256（save/load 双向一致，检测证据损坏）。"""
        with self._lock:
            canon = "\n".join(sorted(self._canonical_row_key(cp)
                                     for cp in self._points.values()))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    # -- 写入 --

    def certify_point(self, key: CertificationKey, passed: bool, *,
                      evidence_hash: str = "", source: str = "",
                      details: dict | None = None) -> None:
        with self._lock:
            self._points[key.to_key()] = CertifiedPoint(
                key=key, passed=passed, evidence_hash=evidence_hash,
                source=source, details=details or {},
            )

    def load_json(self, path: Path | str) -> int:
        """从 audit 输出的 JSON 加载认证点。返回加载数量。

        R39 #29：加载后记录 ``_evidence_hash`` 与 ``_generated_commit``，若 JSON 带
        ``_meta`` 且 hash 不匹配（损坏/篡改）→ 抛 ``ParameterDomainError``（fail
        closed），绝不悄悄保留一份损坏证据。
        """
        from factor_engine.runtime.exceptions import ParameterDomainError

        try:
            p = Path(path)
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._loaded = False
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
        self._evidence_hash = self.compute_evidence_hash()
        self._source_path = str(path)
        self._loaded = n > 0
        meta = data.get("_meta") or {}
        recorded_hash = str(meta.get("evidence_hash", "") or "")
        if recorded_hash and recorded_hash != self._evidence_hash:
            raise ParameterDomainError(
                f"参数域证据 hash 不匹配（损坏/篡改）：recorded={recorded_hash[:16]} "
                f"computed={self._evidence_hash[:16]}（{path}）"
            )
        self._generated_commit = str(meta.get("generated_commit", "") or "")
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
        # R39 #29：记录证据 hash + 生成时的 git HEAD（运行期检测旧 SHA 证据）。
        self._evidence_hash = self.compute_evidence_hash()
        self._generated_commit = self._generated_commit or _current_head()
        path.write_text(json.dumps({
            "certified_points": rows,
            "_meta": {
                "evidence_hash": self._evidence_hash,
                "generated_commit": self._generated_commit,
            },
        }, indent=2, ensure_ascii=False), encoding="utf-8")

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


def get_parameter_domain_store(
    *, ensure_loaded: bool = False,
) -> ParameterDomainCertificationStore:
    """进程级 singleton store。

    R39 #29：``ensure_loaded=True``（production 执行路径）时强制从 R37 证据装载；
    证据缺失/为空/损坏 → 抛 ``ParameterDomainError``（fail closed）——执行期禁止
    空 store 悄悄存在。
    """
    global _GLOBAL_STORE
    with _STORE_LOCK:
        if _GLOBAL_STORE is None:
            _GLOBAL_STORE = ParameterDomainCertificationStore()
        if ensure_loaded:
            _ensure_loaded(_GLOBAL_STORE)
        return _GLOBAL_STORE


def reset_parameter_domain_store() -> None:
    global _GLOBAL_STORE
    with _STORE_LOCK:
        _GLOBAL_STORE = None


#: 已知 invalid 参数点必须**不**被认证（#29 负控）。若证据错误地把它们标成
#: certified → 证据损坏，拒绝启动/执行。
_NEGATIVE_CONTROLS: list[tuple[str, dict[str, Any]]] = [
    ("ts_mean", {"window": 0}),
    ("ts_mean", {"window": -5}),
    ("ts_rank", {"window": 0}),
    ("ts_delay", {"n": 0}),
]


def _ensure_loaded(
    store: ParameterDomainCertificationStore,
    path: str | Path | None = None,
) -> None:
    """装载默认 R37 证据（若已装载则跳过）并跑负控。失败 → fail closed。"""
    from factor_engine.runtime.exceptions import ParameterDomainError

    if getattr(store, "_loaded", False):
        return
    p = path or os.environ.get(
        "FACTOR_ENGINE_PARAMETER_DOMAIN_EVIDENCE", "") or DEFAULT_PARAMETER_DOMAIN_EVIDENCE
    n = store.load_json(p)
    if n == 0:
        raise ParameterDomainError(
            f"参数域认证证据缺失/为空：{p}。production 执行期禁止空 store "
            f"（R39 #29；请运行 scripts/audit_r37_parameter_domains.py 生成）"
        )
    run_negative_controls(store)


def run_negative_controls(
    store: ParameterDomainCertificationStore,
) -> None:
    """#29：已知 invalid 点若被错误 certified → 证据损坏（fail closed）。"""
    from factor_engine.runtime.exceptions import ParameterDomainError

    bad = [
        f"{c}{{k}}".replace("{k}", ",".join(f"{kk}={vv!r}" for kk, vv in kwargs.items()))
        for c, kwargs in _NEGATIVE_CONTROLS
        if store.exact_call_is_certified(c, kwargs)
    ]
    if bad:
        raise ParameterDomainError(
            f"参数域负控失败：invalid 参数点被错误标记 certified {bad}（证据损坏）"
        )


def freeze_parameter_domain_store() -> ParameterDomainCertificationStore:
    """装载并冻结 singleton store（#29）。之后 production 认证要求 store 已装载。"""
    store = get_parameter_domain_store(ensure_loaded=True)
    with _STORE_LOCK:
        store._frozen = True
    return store


def parameter_domain_status() -> dict[str, Any]:
    """执行期可查询的 store 状态（#29：空/旧 SHA 不许悄悄存在）。"""
    store = get_parameter_domain_store()
    with _STORE_LOCK:
        loaded = bool(getattr(store, "_loaded", False))
        return {
            "loaded": loaded,
            "frozen": bool(getattr(store, "_frozen", False)),
            "point_count": len(store._points),
            "evidence_hash": getattr(store, "_evidence_hash", ""),
            "generated_commit": getattr(store, "_generated_commit", ""),
            "source_path": getattr(store, "_source_path", ""),
            "current_head": _current_head(),
        }


def assert_parameter_domain_ready(
    *, strict: bool = True,
) -> ParameterDomainCertificationStore:
    """R39 #29：执行期入口——store 必须已装载（非空）且非旧 SHA。

    - 空 store / 证据缺失 → 抛 ``ParameterDomainError``（fail closed）；
    - ``strict`` 且能确认证据由旧 commit 生成（generated_commit 已知、当前 git
      HEAD 可读、两者不一致）→ 抛（旧 SHA 证据禁止悄悄执行）。
    """
    from factor_engine.runtime.exceptions import ParameterDomainError

    store = get_parameter_domain_store(ensure_loaded=True)
    generated = getattr(store, "_generated_commit", "") or ""
    head = _current_head()
    if strict and generated and head and generated != head:
        raise ParameterDomainError(
            f"参数域证据由 commit {generated[:12]} 生成，当前 HEAD {head[:12]} —— "
            "证据过期（旧 SHA），需重新运行 scripts/audit_r37_parameter_domains.py"
        )
    return store


def _is_production() -> bool:
    try:
        from factor_engine.runtime.production_policy import is_production_mode

        return is_production_mode()
    except Exception:
        return False


def assert_parameter_point_certified(
    canonical: str, kwargs: dict[str, Any], *,
    semantic_version: str = "", backend: str = "pandas_numpy",
    execution_variant: str = "reference", source_context: str = "",
    dtype: str = "float64", grain: str = "daily",
    run_mode: str | None = None,
    store: ParameterDomainCertificationStore | None = None,
) -> None:
    """R37-P0-009 + R39 #26/#29：production compile/execute 前的参数域 membership 门。

    - production + uncertified point => raise ``ParameterDomainError``（fail closed）
    - research + uncertified point => allow + warning（telemetry）
    - ``run_mode``（R39 #26）：认证调用身份的一部分。显式传入时作为**唯一权威**，
      认证层不再自行从环境变量猜 production/research；``None``（无 context 的
      独立/直接调用方）才退回环境变量判定。
    - ``store`` 未显式传入且判定为 production 时强制装载证据（#29）——空 store
      不能悄悄放过。
    """
    from factor_engine.runtime.exceptions import ParameterDomainError

    if run_mode is not None and str(run_mode).strip():
        # #26：认证层不得重猜模式——显式 run_mode 即身份。
        from factor_engine.runtime.production_policy import is_production_mode

        production = is_production_mode(str(run_mode))
    else:
        production = _is_production()
    if store is None:
        store = get_parameter_domain_store(ensure_loaded=production)
    certified = store.exact_call_is_certified(
        canonical, kwargs, semantic_version=semantic_version, backend=backend,
        execution_variant=execution_variant, source_context=source_context,
        dtype=dtype, grain=grain,
    )
    if certified:
        return
    msg = (
        f"parameter point not certified: canonical={canonical!r} kwargs={kwargs!r} "
        f"backend={backend!r} variant={execution_variant!r} "
        f"source_context={source_context!r} dtype={dtype!r} grain={grain!r} "
        f"(production={production}, run_mode={run_mode!r})"
    )
    if production:
        raise ParameterDomainError(msg)
    _log.warning("R37-P0-009 research telemetry: %s", msg)
