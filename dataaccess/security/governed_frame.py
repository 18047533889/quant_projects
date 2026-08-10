"""R25 §50 —— GovernedFrame / Provenance Envelope。

裸 pandas DataFrame 不携带可信 provenance。FactorEngine production 接裸
dataframe 时抛 ``UnknownProvenanceError``；research 显式 ``unsafe_external_frame=True``
且不能 publish（§50）。

``GovernedFrame`` 把 table + source_snapshot + lineage + execution_environment +
security_digest 绑成一个对象，是 FactorEngine DataAccessSource 的受管读句柄。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_access.core.exceptions import UnknownProvenanceError


@dataclass(frozen=True)
class ExecutionEnvironmentIdentity:
    """R25 §51：统一执行环境身份。"""

    dataaccess_version: str | None = None
    factorengine_version: str | None = None
    git_commit: str | None = None
    registry_fingerprint: str | None = None
    contract_ir_fingerprint: str | None = None
    semantic_catalog_fingerprint: str | None = None
    calendar_snapshot_id: str | None = None
    source_snapshot_id: str | None = None
    access_policy_digest: str | None = None
    principal_scope_id: str | None = None
    duckdb_version: str | None = None
    polars_version: str | None = None
    pyarrow_version: str | None = None
    backend: str | None = None
    run_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }


@dataclass(frozen=True)
class GovernedFrame:
    """R25 §50：携带可信 provenance 的受管帧。"""

    table_or_frame: Any
    source_snapshot: Any = None
    lineage: Any = None
    execution_environment: ExecutionEnvironmentIdentity | None = None
    security_digest: str = ""

    @property
    def has_provenance(self) -> bool:
        return (
            self.source_snapshot is not None
            and self.lineage is not None
            and bool(self.security_digest)
        )


def require_governed_provenance(
    frame: Any,
    *,
    run_mode: str = "production",
    unsafe_external_frame: bool = False,
    allow_production_publish: bool = False,
) -> GovernedFrame:
    """R25 §50：受管帧入口（FactorEngine DataAccessSource 读数据必须走这里）。

    - production：裸 dataframe（无 provenance envelope）→ ``UnknownProvenanceError``；
    - research：显式 ``unsafe_external_frame=True`` 放行，但**不能 publish**；
    - ``GovernedFrame`` 直接返回（已携带 provenance）。
    """
    if isinstance(frame, GovernedFrame):
        return frame
    if run_mode == "production" and not unsafe_external_frame:
        raise UnknownProvenanceError(
            "production 收到无 provenance 的裸 dataframe（R25 §50）："
            "FactorEngine 必须消费 GovernedFrame（含 source_snapshot + lineage + "
            "execution_environment + security_digest）。裸 frame 无法回答「谁读的 / "
            "读了什么 source / 用什么 PIT」。"
        )
    if not unsafe_external_frame and not isinstance(frame, GovernedFrame):
        # 普通 research 裸 frame：显式标记 unsafe，不能 publish。
        return GovernedFrame(
            table_or_frame=frame,
            source_snapshot=None,
            lineage=None,
            security_digest="unsafe_external_frame",
        )
    return GovernedFrame(
        table_or_frame=frame,
        source_snapshot=None,
        lineage=None,
        security_digest="unsafe_external_frame" if unsafe_external_frame else "",
    )
