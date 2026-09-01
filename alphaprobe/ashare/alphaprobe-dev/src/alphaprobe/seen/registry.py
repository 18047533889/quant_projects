"""Factor registry + factor_alias + rediscovery（§55/§56 语义）。

注册：跨 Miner 同 hash 同 factor_node 只加 alias。
跨 Round：times_seen+1 不新建。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.seen.store import SeenStore


class RegistryService:
    """Factor registry 管理：alias 注册、rediscovery 统计、seed 导入。"""

    def __init__(self, store: SeenStore) -> None:
        self._store = store

    # -- Alias ---------------------------------------------------------------

    def register_alias(
        self,
        identity: Any,
        canonical_factor_id: str,
        *,
        source_system: str = "alphaprobe",
        source_external_id: str | None = None,
    ) -> bool:
        """跨 Miner 同 hash 同 factor_node 只加 alias。

        Parameters
        ----------
        identity : FactorIdentity
            鸭子类型，至少读 canonical_ast_hash。
        canonical_factor_id : str
            权威 factor_id（已存在库中）。
        source_system : str
            来源系统。
        source_external_id : str | None
            外部 id（如原 miner 的 factor_id）。

        Returns
        -------
        bool
            True 表示新增 alias，False 表示已存在。
        """
        canon_hash = getattr(identity, "canonical_ast_hash", None)
        if not canon_hash:
            raise ValueError("identity missing canonical_ast_hash")
        factor_pk = self._store.factor_pk_by_id(canonical_factor_id)
        if factor_pk is None:
            raise ValueError(f"factor_id not found: {canonical_factor_id}")
        return self._store.insert_alias(
            factor_pk=factor_pk,
            alias_dsl_hash=canon_hash,
            alias_dsl=getattr(identity, "canonical_formula", None),
            source_system=source_system,
            source_external_id=source_external_id or getattr(identity, "factor_id", None),
        )

    def count_aliases(self, factor_id: str) -> int:
        factor_pk = self._store.factor_pk_by_id(factor_id)
        if factor_pk is None:
            return 0
        return self._store.count_aliases_for(factor_pk)

    def total_aliases(self) -> int:
        return self._store.count_aliases()

    # -- Rediscovery ---------------------------------------------------------

    def bump_times_seen(self, factor_id: str) -> None:
        """跨 Round 增加 times_seen（不新建）。"""
        factor_pk = self._store.factor_pk_by_id(factor_id)
        if factor_pk is not None:
            self._store.bump_times_seen(factor_pk)

    # -- Seed ingestion ------------------------------------------------------

    def ingest_seed(
        self,
        identity: Any,
        *,
        source_system: str = "seed_library",
        run_id: str = "",
        fingerprint: bytes | None = None,
        identity_version: str | None = None,
    ) -> dict[str, Any]:
        """种子库导入：如果已存在，跳过 + 返回 existing 信息。

        Returns
        -------
        dict
            {"created": bool, "factor_id": str, "factor_pk": int, "verdict": str}
        """
        from alphaprobe.seen.reservation import ReservationService

        svc = ReservationService(self._store)
        result = svc.reserve(
            identity,
            source_system=source_system,
            run_id=run_id,
            identity_version=identity_version,
            status="RESERVED",
        )
        return {
            "created": result.acquired,
            "factor_id": result.factor_id or result.existing_factor_id,
            "factor_pk": result.factor_pk or result.existing_factor_pk,
            "verdict": result.verdict,
            "existing_factor_id": result.existing_factor_id,
        }