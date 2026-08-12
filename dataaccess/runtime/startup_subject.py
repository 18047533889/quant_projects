"""R32-P0-014/P0-015 —— StartupSubjectDigest：启动环境身份证明。

production 启动证书必须绑定环境主体摘要——build SHA、package version、registry
digest、contract digest、semantic schema、policy digest、calendar snapshot、
credential generation、source profile、worker topology、gate version。

证书复用前验证 current_subject_digest == cert.subject_digest；关键配置变化立即
invalidate。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StartupSubjectDigest:
    """启动环境主体摘要（R32-P0-014）。"""

    build_sha: str = ""
    package_version: str = ""
    registry_digest: str = ""
    contract_digest: str = ""
    semantic_schema_version: str = ""
    policy_digest: str = ""
    calendar_snapshot_ids: tuple[str, ...] = ()
    credential_generation: str = ""
    source_profile: str = ""
    worker_topology: str = ""
    gate_version: str = ""

    def to_digest(self) -> str:
        """计算确定性摘要（R32-P0-014）。"""
        payload = "\n".join([
            f"build_sha={self.build_sha}",
            f"package_version={self.package_version}",
            f"registry_digest={self.registry_digest}",
            f"contract_digest={self.contract_digest}",
            f"semantic_schema_version={self.semantic_schema_version}",
            f"policy_digest={self.policy_digest}",
            f"calendar_snapshot_ids={','.join(sorted(self.calendar_snapshot_ids))}",
            f"credential_generation={self.credential_generation}",
            f"source_profile={self.source_profile}",
            f"worker_topology={self.worker_topology}",
            f"gate_version={self.gate_version}",
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_startup_subject_digest(store: Any) -> StartupSubjectDigest:
    """从 store 构造启动主体摘要（R32-P0-014）。"""
    build_sha = _get_build_sha()
    package_version = _get_package_version()
    registry_digest = _get_registry_digest(store)
    contract_digest = _get_contract_digest(store)
    semantic_schema_version = _get_semantic_schema_version()
    policy_digest = _get_policy_digest()
    calendar_snapshot_ids = _get_calendar_snapshot_ids(store)
    credential_generation = _get_credential_generation()
    source_profile = _get_source_profile()
    worker_topology = _get_worker_topology()
    gate_version = "R32-P0-014"
    return StartupSubjectDigest(
        build_sha=build_sha,
        package_version=package_version,
        registry_digest=registry_digest,
        contract_digest=contract_digest,
        semantic_schema_version=semantic_schema_version,
        policy_digest=policy_digest,
        calendar_snapshot_ids=tuple(calendar_snapshot_ids),
        credential_generation=credential_generation,
        source_profile=source_profile,
        worker_topology=worker_topology,
        gate_version=gate_version,
    )


def _get_build_sha() -> str:
    """获取 build SHA（_build_info.py 或 git）。"""
    try:
        from data_access import _build_info
        return getattr(_build_info, "BUILD_SHA", "")[:16]
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:16]
    except Exception:
        pass
    return "unknown"


def _get_package_version() -> str:
    """获取 package version。"""
    try:
        from data_access import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _get_registry_digest(store: Any) -> str:
    """获取 registry digest（dataset 数量 + names hash）。"""
    try:
        registry = getattr(store, "_registry", None)
        if registry is None:
            return "unknown"
        datasets = list(getattr(registry, "_datasets", {}).keys())
        payload = "\n".join(sorted(datasets))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def _get_contract_digest(store: Any) -> str:
    """获取 contract digest（ContractIR digest）。"""
    try:
        ir = store.contract_ir()
        if ir is None:
            return "unknown"
        digest_fn = getattr(ir, "digest", None)
        if callable(digest_fn):
            return str(digest_fn())[:16]
    except Exception:
        pass
    return "unknown"


def _get_semantic_schema_version() -> str:
    """获取 semantic schema version。"""
    try:
        from data_access.read import read_contract
        return getattr(read_contract, "SCHEMA_VERSION", "unknown")
    except Exception:
        return "unknown"


def _get_policy_digest() -> str:
    """获取 policy digest。"""
    try:
        from data_access.security.policy import current_policy_digest
        return current_policy_digest()[:16]
    except Exception:
        return "unknown"


def _get_calendar_snapshot_ids(store: Any) -> list[str]:
    """获取 calendar snapshot IDs。"""
    from data_access.runtime.startup_gate import calendar_digest
    ids: list[str] = []
    for market in ("ashare", "us"):
        try:
            cal = store.get_calendar(market)
            if cal is not None:
                ids.append(f"{market}:{calendar_digest(cal)}")
        except Exception:
            pass
    return ids


def _get_credential_generation() -> str:
    """获取 credential generation。"""
    try:
        from data_access.security.credentials import _global_credential_provider
        provider = _global_credential_provider()
        if provider is not None:
            gen = getattr(provider, "generation", None)
            if gen is not None:
                return str(gen)[:16]
    except Exception:
        pass
    return "unknown"


def _get_source_profile() -> str:
    """获取 source profile（local/remote/mirror）。"""
    try:
        from data_access.cos.mirror import known_cos_mirror_local_roots
        roots = list(known_cos_mirror_local_roots())
        if roots:
            return f"mirror:{len(roots)}"
        return "remote"
    except Exception:
        return "unknown"


def _get_worker_topology() -> str:
    """获取 worker topology（single/multi）。"""
    import os
    workers = os.environ.get("DATA_ACCESS_WORKERS", "1")
    single = os.environ.get("DATA_ACCESS_SINGLE_WORKER", "").strip().lower() in {
        "1", "true", "yes"
    }
    if single or workers == "1":
        return "single"
    return f"multi:{workers}"
