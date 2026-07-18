# -*- coding: utf-8 -*-
"""DuckDB httpfs / S3 配置（COS 远程直读）。"""
from __future__ import annotations

import hashlib
import logging
import threading

from .remote import S3Credentials, resolve_s3_credentials
from data_access.core.exceptions import ValidationError

logger = logging.getLogger("data_access.s3_duckdb")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _creds_fingerprint(creds: S3Credentials) -> str:
    """含 secret 哈希，便于密钥轮换后重新注入。"""
    secret_fp = hashlib.sha256(creds.secret_access_key.encode("utf-8")).hexdigest()[:16]
    return (
        f"{creds.access_key_id}:{secret_fp}:{creds.endpoint}:"
        f"{creds.region}:{creds.url_style}:{int(creds.use_ssl)}"
    )


def apply_s3_credentials(conn, creds: S3Credentials) -> None:
    """对 DuckDB 连接注入 S3/COS 访问参数。"""
    try:
        conn.execute("LOAD httpfs;")
    except Exception:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"SET s3_access_key_id={_sql_string(creds.access_key_id)}")
    conn.execute(f"SET s3_secret_access_key={_sql_string(creds.secret_access_key)}")
    conn.execute(f"SET s3_endpoint={_sql_string(creds.endpoint)}")
    conn.execute(f"SET s3_region={_sql_string(creds.region)}")
    conn.execute(f"SET s3_url_style={_sql_string(creds.url_style)}")
    conn.execute(f"SET s3_use_ssl={'true' if creds.use_ssl else 'false'}")
    logger.info(
        "duckdb httpfs configured endpoint=%s region=%s url_style=%s",
        creds.endpoint,
        creds.region,
        creds.url_style,
    )


class S3ConfigState:
    """进程内 httpfs 是否已配置（凭证变更需 reset engine）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configured = False
        self._fingerprint: str | None = None

    def ensure(self, conn) -> None:
        """对共享连接：同指纹跳过；指纹变更则重新注入。"""
        creds = resolve_s3_credentials()
        fp = _creds_fingerprint(creds)
        with self._lock:
            if self._configured and self._fingerprint == fp:
                return
            try:
                apply_s3_credentials(conn, creds)
            except Exception as exc:
                raise ValidationError(
                    f"DuckDB httpfs 配置失败: {exc}. "
                    "请确认已安装 duckdb 且 LOAD httpfs 可用，并检查 COS 凭证/endpoint。"
                ) from exc
            self._configured = True
            self._fingerprint = fp

    def configure_fresh(self, conn) -> None:
        """对新建 ``:memory:`` 连接始终注入凭证（scoped sql 用）。"""
        creds = resolve_s3_credentials()
        fp = _creds_fingerprint(creds)
        try:
            apply_s3_credentials(conn, creds)
        except Exception as exc:
            raise ValidationError(
                f"DuckDB httpfs 配置失败: {exc}. "
                "请确认已安装 duckdb 且 LOAD httpfs 可用，并检查 COS 凭证/endpoint。"
            ) from exc
        with self._lock:
            self._configured = True
            self._fingerprint = fp

    def reset(self) -> None:
        with self._lock:
            self._configured = False
            self._fingerprint = None


_s3_state = S3ConfigState()


def ensure_duckdb_s3(conn) -> None:
    _s3_state.ensure(conn)


def configure_fresh_duckdb_s3(conn) -> None:
    """为独立连接注入 S3（不受共享连接「已配置」短路影响）。"""
    _s3_state.configure_fresh(conn)


def reset_duckdb_s3_state() -> None:
    _s3_state.reset()
