# -*- coding: utf-8 -*-
"""DataAccess 安全薄层（R24 §26 推荐的最小 security 模块）。

不要为了安全重新写 DataAccess。这里只提供薄层契约：

    principal.py    DataPrincipal / AccessPolicy（逻辑授权身份与策略）
    policy.py       DatasetAuthorizer / DefaultAuthorizer（授权执行器）
    credentials.py  CredentialProvider / CredentialMaterial（凭证边界）
    redaction.py    Secret / URI / Authorization header 脱敏

DataAccessStore 持有 principal + authorizer + credential_provider，共享给
read / scan / read_joined / read_factors / read_uri / mirror / remote / HTTP
——不要每个 backend 自己猜身份。
"""
from __future__ import annotations

from data_access.security.credentials import (
    CredentialMaterial,
    CredentialProvider,
    EnvCredentialProvider,
    CosCliConfigProvider,
    allow_coscli_config_parse,
)
from data_access.security.policy import (
    DatasetAuthorizer,
    DefaultAuthorizer,
    set_authorizer,
    get_authorizer,
)
from data_access.security.principal import (
    DataPrincipal,
    AccessPolicy,
    DEFAULT_LOCAL_PRINCIPAL,
    DEFAULT_ACCESS_POLICY,
    ALL_ACTIONS,
    DATASET_SCOPED_ACTIONS,
    ACTION_DATASET_LIST,
    ACTION_DATASET_READ,
    ACTION_FACTOR_LIST,
    ACTION_FACTOR_READ,
    ACTION_URI_READ,
    ACTION_METADATA_READ,
    ACTION_FACTOR_METADATA_SENSITIVE,
)
from data_access.security.redaction import (
    redact_secret,
    redact_uri,
    redact_authorization_header,
)
from data_access.security.runtime import (
    RuntimeSecurityContext,
    resolve_runtime_context,
)
from data_access.security.api_principals import (
    hash_api_key,
    ApiPrincipalRegistry,
    get_api_principal_registry,
    reset_api_principal_registry,
)

__all__ = [
    "CredentialMaterial",
    "CredentialProvider",
    "EnvCredentialProvider",
    "CosCliConfigProvider",
    "allow_coscli_config_parse",
    "DatasetAuthorizer",
    "DefaultAuthorizer",
    "set_authorizer",
    "get_authorizer",
    "DataPrincipal",
    "AccessPolicy",
    "DEFAULT_LOCAL_PRINCIPAL",
    "DEFAULT_ACCESS_POLICY",
    "ALL_ACTIONS",
    "DATASET_SCOPED_ACTIONS",
    "ACTION_DATASET_LIST",
    "ACTION_DATASET_READ",
    "ACTION_FACTOR_LIST",
    "ACTION_FACTOR_READ",
    "ACTION_URI_READ",
    "ACTION_METADATA_READ",
    "ACTION_FACTOR_METADATA_SENSITIVE",
    "redact_secret",
    "redact_uri",
    "redact_authorization_header",
    "RuntimeSecurityContext",
    "resolve_runtime_context",
    "hash_api_key",
    "ApiPrincipalRegistry",
    "get_api_principal_registry",
    "reset_api_principal_registry",
]
