"""Atomic reservation（§11.6 / §59）。

canonical_ast_hash / signal_equivalence_id 都 UNIQUE 索引；
reservation 原子性靠 ``INSERT ... ON CONFLICT DO NOTHING`` + 查回 existing，
在单事务里完成（SQLite 串行写天然原子）。冲突时调用方必须立即停止昂贵回测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphaprobe.seen.store import SeenStore


@dataclass(frozen=True)
class IdentityLookupResult:
    """§22/§31 lookup 结果。"""

    verdict: str
    existing_factor_id: str | None = None
    existing_factor_pk: int | None = None
    family_id: str | None = None
    family_member_count: int = 0
    family_saturation: float = 0.0
    orientation: int = 1
    identity_version: str = "1"
    canonical_dsl: str | None = None
    reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReservationResult:
    """§11.6 reservation 结果。"""

    acquired: bool
    factor_id: str | None = None
    factor_pk: int | None = None
    verdict: str = "NEW"
    existing_factor_id: str | None = None
    existing_factor_pk: int | None = None
    family_id: str | None = None
    family_member_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


class ReservationService:
    """Atomic reservation（写路径）：NEW→RESERVED 单事务。

    同 signal 并发 reserve 只允许恰好 1 个 acquired（threading 安全）。
    """

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    # -- 查找 ---------------------------------------------------------------

    def lookup_identity(
        self,
        identity: Any,
        *,
        identity_version: str | None = None,
    ) -> IdentityLookupResult:
        """§22/§31：给定 identity 对象，返回判定。

        判定优先级（§语义）：
        1. VERSION_MISMATCH：identity_version 变了且库里有同 hash 但不同 version
        2. EXACT_DUPLICATE：canonical_ast_hash 相等（同 version）
        3. SIGN_EQUIVALENT_DUPLICATE：signal_equivalence_id 相等（同 version）
        4. FAMILY_SATURATED（非 hard reject，附 family 统计）
        5. NEW
        """
        version = identity_version or getattr(identity, "identity_version", None) or "1"
        canon_hash = getattr(identity, "canonical_ast_hash", None)
        signal_id = getattr(identity, "signal_equivalence_id", None)
        family_id = getattr(identity, "parameter_family_id", None)
        orientation = int(getattr(identity, "orientation", 1) or 1)

        # 先查同 hash 其它 version（§45：不同 version 不互判 duplicate）
        other_version = self._store.find_factor_by_hash_any_version(canon_hash)
        if other_version is not None and other_version["identity_version"] != version:
            return IdentityLookupResult(
                verdict="VERSION_MISMATCH",
                existing_factor_id=other_version["factor_id"],
                existing_factor_pk=other_version["factor_pk"],
                identity_version=version,
                reason=(
                    f"same canonical hash exists under identity_version "
                    f"{other_version['identity_version']} != {version}"
                ),
            )

        existing = self._store.find_factor_by_hash(version, canon_hash)
        if existing is not None:
            return IdentityLookupResult(
                verdict="EXACT_DUPLICATE",
                existing_factor_id=existing["factor_id"],
                existing_factor_pk=existing["factor_pk"],
                family_id=existing["parameter_family_id"],
                identity_version=version,
                canonical_dsl=existing["canonical_dsl"],
            )

        existing_sig = self._store.find_factor_by_signal(version, signal_id)
        if existing_sig is not None:
            # f 与 -f 同 signal_id 但不同 orientation：仍是 SIGN_EQUIVALENT
            return IdentityLookupResult(
                verdict="SIGN_EQUIVALENT_DUPLICATE",
                existing_factor_id=existing_sig["factor_id"],
                existing_factor_pk=existing_sig["factor_pk"],
                family_id=existing_sig["parameter_family_id"],
                orientation=orientation,
                identity_version=version,
                canonical_dsl=existing_sig["canonical_dsl"],
            )

        # family 统计（非 reject）
        member_count = 0
        family_saturation = 0.0
        if family_id:
            members = self._store.find_family_members(version, family_id)
            member_count = len(members)
            family_saturation = min(member_count / 50.0, 1.0)  # §20 soft guidance
            if member_count >= 30:
                return IdentityLookupResult(
                    verdict="FAMILY_SATURATED",
                    family_id=family_id,
                    family_member_count=member_count,
                    family_saturation=family_saturation,
                    identity_version=version,
                    reason="family saturated (soft guidance, not hard reject)",
                )
        return IdentityLookupResult(
            verdict="NEW",
            family_id=family_id,
            family_member_count=member_count,
            family_saturation=family_saturation,
            orientation=orientation,
            identity_version=version,
        )

    # -- 写路径 ---------------------------------------------------------------

    def _insert_new(
        self, identity: Any, *, factor_id: str | None, version: str,
        canon_hash: str, signal_id: str, family_id: str | None,
        orientation: int, canonical_dsl: str | None,
        source_system: str, run_id: str, status: str,
    ) -> int:
        """事务内插入新因子 + family 统计 + meta 计数。返回 factor_pk。

        调用方必须已持有 transaction()（本方法不 commit，由外层统一提交）。
        """
        fid = factor_id or str(canon_hash)[:12]
        try:
            pk = self._store.insert_factor(
                factor_id=fid,
                canonical_ast_hash=canon_hash,
                signal_equivalence_id=signal_id,
                parameter_family_id=family_id,
                orientation=orientation,
                canonical_dsl=canonical_dsl,
                identity_version=version,
                operator_semantics_version=getattr(
                    identity, "operator_semantics_version", None
                ),
                source_system=source_system,
                run_id=run_id,
                status=status,
            )
        except Exception:
            raise
        if family_id:
            self._store.upsert_family(
                family_id=family_id,
                family_template=getattr(identity, "family_template", None),
                factor_pk=pk,
                factor_id=fid,
                parameter_fingerprint=getattr(identity, "parameter_fingerprint", None),
            )
        self._store.incr_meta(f"verdict:NEW:{version}", by=1)
        self._store.incr_meta(f"version:{version}:count", by=1)
        return int(pk)

    def reserve(
        self,
        identity: Any,
        *,
        source_system: str = "alphaprobe",
        run_id: str = "",
        worker_id: str = "",
        identity_version: str | None = None,
        status: str = "RESERVED",
    ) -> ReservationResult:
        """Atomic reservation（单事务，并发恰好 1 个 acquired）。

        Parameters
        ----------
        identity : FactorIdentity
            鸭子类型对象，至少读 canonical_ast_hash / signal_equivalence_id /
            parameter_family_id / orientation / canonical_formula。
        source_system : str
            来源系统（miner / seed_library / ...）。
        run_id : str
            运行 id。
        worker_id : str
            工作线程/进程 id（记录用）。
        identity_version : str
            身份版本（§45）。不同版本不互判。
        status : str
            新因子初始状态，默认 RESERVED。

        Returns
        -------
        ReservationResult
            acquired=True 表示本次调用赢得 reservation；否则
            existing_factor_id 指向已存在因子。
        """
        version = identity_version or getattr(identity, "identity_version", None) or "1"
        canon_hash = getattr(identity, "canonical_ast_hash", None)
        signal_id = getattr(identity, "signal_equivalence_id", None)
        family_id = getattr(identity, "parameter_family_id", None)
        orientation = int(getattr(identity, "orientation", 1) or 1)
        canonical_dsl = getattr(identity, "canonical_formula", None) or canonical_dsl_text(identity)

        # 先查同 hash 其它 version（§45：不同 version 不互判 duplicate）
        other_version = self._store.find_factor_by_hash_any_version(canon_hash)
        if other_version is not None and other_version["identity_version"] != version:
            # 不同 version 不互判：照常新建本 version 的行，但记录 mismatch。
            # factor_id 加 version 后缀避免跨版本碰撞。
            base_fid = getattr(identity, "factor_id", None) or str(canon_hash)[:12]
            namespaced_fid = f"{base_fid}.v{version}"
            with self._store.transaction():
                pk = self._insert_new(
                    identity, factor_id=namespaced_fid, version=version,
                    canon_hash=canon_hash, signal_id=signal_id,
                    family_id=family_id, orientation=orientation,
                    canonical_dsl=canonical_dsl, source_system=source_system,
                    run_id=run_id, status=status,
                )
            self._store.incr_meta(f"verdict:VERSION_MISMATCH:{version}", by=1)
            return ReservationResult(
                acquired=True,
                factor_id=namespaced_fid,
                factor_pk=pk,
                verdict="NEW",
                family_id=family_id,
                stats={
                    "identity_version_mismatch": True,
                    "other_version": other_version["identity_version"],
                    "other_factor_id": other_version["factor_id"],
                },
            )

        with self._store.transaction():
            # 1) 查 existing（同 version）
            existing = self._store.find_factor_by_hash(version, canon_hash)
            if existing is not None:
                self._store.bump_times_seen(existing["factor_pk"])
                if int(existing["orientation"]) != orientation:
                    self._store.update_orientations(existing["factor_pk"], orientation)
                self._store.incr_meta(f"verdict:EXACT_DUPLICATE:{version}", by=1)
                return ReservationResult(
                    acquired=False,
                    verdict="EXACT_DUPLICATE",
                    existing_factor_id=existing["factor_id"],
                    existing_factor_pk=existing["factor_pk"],
                    family_id=existing["parameter_family_id"],
                    stats={"rediscovery": True},
                )

            existing_sig = self._store.find_factor_by_signal(version, signal_id)
            if existing_sig is not None:
                self._store.bump_times_seen(existing_sig["factor_pk"])
                if int(existing_sig["orientation"]) != orientation:
                    self._store.update_orientations(existing_sig["factor_pk"], orientation)
                self._store.incr_meta(f"verdict:SIGN_EQUIVALENT_DUPLICATE:{version}", by=1)
                return ReservationResult(
                    acquired=False,
                    verdict="SIGN_EQUIVALENT_DUPLICATE",
                    existing_factor_id=existing_sig["factor_id"],
                    existing_factor_pk=existing_sig["factor_pk"],
                    family_id=existing_sig["parameter_family_id"],
                    stats={"rediscovery": True},
                )

            # 2) 新因子 INSERT（UNIQUE 约束兜底：并发时只有一个成功）
            identity_factor_id = getattr(identity, "factor_id", None)
            try:
                pk = self._insert_new(
                    identity, factor_id=identity_factor_id, version=version,
                    canon_hash=canon_hash, signal_id=signal_id,
                    family_id=family_id, orientation=orientation,
                    canonical_dsl=canonical_dsl, source_system=source_system,
                    run_id=run_id, status=status,
                )
            except Exception:
                # 并发冲突（UNIQUE）→ 查回 existing
                re_existing = self._store.find_factor_by_hash(version, canon_hash)
                if re_existing is not None:
                    self._store.bump_times_seen(re_existing["factor_pk"])
                    self._store.incr_meta(f"verdict:EXACT_DUPLICATE:{version}", by=1)
                    return ReservationResult(
                        acquired=False,
                        verdict="EXACT_DUPLICATE",
                        existing_factor_id=re_existing["factor_id"],
                        existing_factor_pk=re_existing["factor_pk"],
                    )
                re_sig = self._store.find_factor_by_signal(version, signal_id)
                if re_sig is not None:
                    self._store.bump_times_seen(re_sig["factor_pk"])
                    self._store.incr_meta(f"verdict:SIGN_EQUIVALENT_DUPLICATE:{version}", by=1)
                    return ReservationResult(
                        acquired=False,
                        verdict="SIGN_EQUIVALENT_DUPLICATE",
                        existing_factor_id=re_sig["factor_id"],
                        existing_factor_pk=re_sig["factor_pk"],
                    )
                raise

            return ReservationResult(
                acquired=True,
                factor_id=identity_factor_id or str(canon_hash)[:12],
                factor_pk=pk,
                verdict="NEW",
                family_id=family_id,
                stats={"worker_id": worker_id, "run_id": run_id},
            )


def canonical_dsl_text(identity: Any) -> str | None:
    """从 identity 提取 canonical DSL 文本（鸭子类型兼容多个实现）。"""
    for attr in ("canonical_formula", "canonical_dsl", "dsl", "formula"):
        v = getattr(identity, attr, None)
        if v:
            return str(v)
    return None
