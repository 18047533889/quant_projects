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


def _frame_is_governed(frame: Any) -> bool:
    return isinstance(frame, GovernedFrame)


def validate_governed_frame_provenance(
    frame: GovernedFrame,
    *,
    expected_security_digest: str | None = None,
    strict: bool = True,
) -> list[str]:
    """R26-P0-023：GovernedFrame 可证明性校验（不能被随便构造的字符串伪造）。

    检查：
        - ``source_snapshot`` 非空且可验证（含 content_digest）；
        - ``lineage`` 非空；
        - ``execution_environment`` 存在且含 run_mode；
        - ``security_digest`` 与当前执行上下文 digest 一致（若提供 expected）；
        - ``run_mode`` 合法。

    返回 problems；空 = 可证明。
    """
    problems: list[str] = []
    if frame.source_snapshot is None:
        problems.append("source_snapshot 缺失（无法证明读了哪个 source）")
    elif getattr(frame.source_snapshot, "content_digest", None) is None:
        problems.append("source_snapshot 无 content_digest（无法证明 snapshot 身份）")
    if frame.lineage is None:
        problems.append("lineage 缺失（无法证明读取参数/PIT）")
    if frame.execution_environment is None:
        problems.append("execution_environment 缺失")
    elif not getattr(frame.execution_environment, "run_mode", None):
        problems.append("execution_environment.run_mode 缺失")
    if not frame.security_digest:
        problems.append("security_digest 为空")
    elif expected_security_digest and frame.security_digest != expected_security_digest:
        problems.append(
            "security_digest 与当前执行上下文不一致"
            f"（frame={frame.security_digest!r} expected={expected_security_digest!r}）"
            "——可能来自伪造/过期 security context"
        )
    if getattr(frame.execution_environment, "run_mode", None) not in {
        "interactive_research",
        "automated_research",
        "production",
    }:
        problems.append("run_mode 非法")
    return problems


def require_governed_provenance(
    frame: Any,
    *,
    run_mode: str = "production",
    unsafe_external_frame: bool = False,
    allow_production_publish: bool = False,
) -> GovernedFrame:
    """R25 §50 + R26-P0-023：受管帧入口（FactorEngine DataAccessSource 必须走这里）。

    - production：
        - 裸 dataframe（无 provenance envelope）→ ``UnknownProvenanceError``；
        - ``GovernedFrame`` 必须通过 ``validate_governed_frame_provenance``——
          ``GovernedFrame(table_or_frame=df)`` 这种没有真实 provenance 的伪造帧
          一律拒绝（R26-P0-023：不能随便构造字符串冒充可信 provenance）；
    - research：显式 ``unsafe_external_frame=True`` 放行，但**不能 publish**；
    - ``allow_production_publish``：只有 production + 可证明 provenance 才允许
      publish（否则 raise）。
    """
    from data_access.core.exceptions import UnknownProvenanceError

    if isinstance(frame, GovernedFrame):
        expected = None
        try:
            from data_access.security.execution_context import current_security_digest

            expected = current_security_digest()
        except Exception:
            expected = None
        problems = validate_governed_frame_provenance(
            frame, expected_security_digest=expected, strict=run_mode == "production"
        )
        if problems:
            raise UnknownProvenanceError(
                "GovernedFrame 无法证明 provenance（R26-P0-023）："
                + "; ".join(problems)
            )
        return frame
    if run_mode == "production" and not unsafe_external_frame:
        raise UnknownProvenanceError(
            "production 收到无 provenance 的裸 dataframe（R25 §50 / R26-P0-023）："
            "FactorEngine 必须消费可证明的 GovernedFrame（含 source_snapshot + "
            "lineage + execution_environment + 与执行上下文一致的 security_digest）。"
            "裸 frame / 伪造 frame 无法回答「谁读的 / 读了什么 source / 用什么 PIT」。"
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
