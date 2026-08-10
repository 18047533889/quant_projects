# -*- coding: utf-8 -*-
"""RuntimeSecurityContext（R24 P0-S5 §7）。

把 production / strict_semantics / principal / authorizer 统一成一个不可变
上下文——「ServiceSettings(production_mode=True) 但 DataAccess 底层还是 research
语义」的漂移不能再出现。HTTP service 与 DataAccessStore 共享同一个 context。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_access.security.policy import (
    DatasetAuthorizer,
    DefaultAuthorizer,
    get_authorizer,
)
from data_access.security.principal import (
    DEFAULT_ACCESS_POLICY,
    DEFAULT_LOCAL_PRINCIPAL,
    DataPrincipal,
)


@dataclass(frozen=True)
class RuntimeSecurityContext:
    """一次会话的统一安全上下文。

    - ``production``：production 运行模式（env FACTOR_ENGINE_RUN_MODE /
      QUANT_PRODUCTION_MODE 的解析结果，与 DataAccess 底层共享）。
    - ``strict_semantics``：fail-closed 语义开关（production OR strict_read）。
    - ``principal``：当前 DataPrincipal（server 身份）。
    - ``authorizer``：DatasetAuthorizer（逻辑授权）。
    - ``access_policy``：authorizer.policy 的便捷引用。
    """

    production: bool = False
    strict_semantics: bool = True
    principal: DataPrincipal = DEFAULT_LOCAL_PRINCIPAL
    authorizer: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "authorizer", self.authorizer or DefaultAuthorizer()
        )

    @property
    def access_policy(self) -> Any:
        return getattr(self.authorizer, "policy", DEFAULT_ACCESS_POLICY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "production": self.production,
            "strict_semantics": self.strict_semantics,
            "principal": self.principal.to_dict(),
            "access_policy_digest": self.access_policy.digest(),
        }


def resolve_runtime_context(
    *,
    production: bool | None = None,
    principal: DataPrincipal | None = None,
    authorizer: DatasetAuthorizer | None = None,
) -> RuntimeSecurityContext:
    """解析统一运行时安全上下文（HTTP service / store 共用）。

    ``production`` 缺省从 env 判定（FACTOR_ENGINE_RUN_MODE / QUANT_PRODUCTION_MODE）；
    ``strict_semantics`` 恒为 production OR DATA_ACCESS_STRICT_READ。authorizer
    缺省取进程共享授权器（``get_authorizer``）。
    """
    from data_access.read.query_budget import _production_mode, is_strict_semantics

    if production is None:
        production = _production_mode()
    strict = is_strict_semantics()
    authorizer = authorizer or get_authorizer()
    principal = principal or getattr(authorizer, "principal", DEFAULT_LOCAL_PRINCIPAL)
    return RuntimeSecurityContext(
        production=production,
        strict_semantics=strict,
        principal=principal,
        authorizer=authorizer,
    )


__all__ = ["RuntimeSecurityContext", "resolve_runtime_context"]
