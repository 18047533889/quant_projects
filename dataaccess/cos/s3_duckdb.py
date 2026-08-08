# -*- coding: utf-8 -*-
"""DuckDB httpfs / S3 配置（COS 远程直读）。"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import weakref

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


def _production_mode() -> bool:
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if fe == "production":
        return True
    return os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}


def _load_httpfs(conn) -> None:
    """#P1-3 production 只 ``LOAD httpfs``，禁止 query-time 联网 INSTALL。

    production 启动 health check（``httpfs_probe``）保证扩展已预装——运行时缺
    扩展是 deployment 错误，不是自动安装的时机；dev/research 才允许 INSTALL。
    """
    try:
        conn.execute("LOAD httpfs;")
    except Exception as load_exc:
        if _production_mode():
            raise ValidationError(
                "production 环境缺少 DuckDB httpfs 扩展（LOAD 失败）。"
                "httpfs 必须在部署阶段固定预装，禁止 query-time INSTALL 联网安装。"
            ) from load_exc
        try:
            conn.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception as exc:
            raise ValidationError(
                f"DuckDB httpfs 扩展不可用（LOAD 失败: {load_exc}; "
                f"INSTALL 失败: {exc}）。请检查 duckdb 安装/网络/扩展目录。"
            ) from exc


def apply_s3_credentials(conn, creds: S3Credentials) -> None:
    """对 DuckDB 连接注入 S3/COS 访问参数。

    新版本（>=1.1）用 ``CREATE OR REPLACE SECRET``（Secret Manager，支持
    credential chain / refresh / scope）；老版本回退 ``SET s3_*``。选择由
    DuckDBCapabilities 决定，不硬编码版本号。
    """
    _load_httpfs(conn)

    from data_access.core.duckdb_capabilities import get_duckdb_capabilities

    if get_duckdb_capabilities().supports_create_secret:
        try:
            _apply_s3_secret(conn, creds)
            return
        except Exception as exc:
            logger.warning(
                "CREATE SECRET 失败（%s），回退 SET s3_* 注入", exc,
            )
    _apply_s3_legacy(conn, creds)


def _apply_s3_secret(conn, creds: S3Credentials) -> None:
    """DuckDB Secret Manager 方式：credential 变化自动按指纹重建。"""
    secret_name = "data_access_cos"
    use_ssl = "true" if creds.use_ssl else "false"
    url_style = creds.url_style or "path"
    # secret 名是标识符（非字符串字面量）；常量自持有，安全拼接
    conn.execute(
        f"CREATE OR REPLACE SECRET {secret_name} (TYPE S3,"
        + f"KEY_ID {_sql_string(creds.access_key_id)},"
        + f"SECRET {_sql_string(creds.secret_access_key)},"
        + f"ENDPOINT {_sql_string(creds.endpoint)},"
        + f"REGION {_sql_string(creds.region)},"
        + f"USE_SSL {use_ssl},"
        + f"URL_STYLE {_sql_string(url_style)});"
    )
    logger.info(
        "duckdb httpfs secret configured endpoint=%s region=%s url_style=%s",
        creds.endpoint,
        creds.region,
        url_style,
    )


def _apply_s3_legacy(conn, creds: S3Credentials) -> None:
    """老版本 SET s3_* 注入。"""
    conn.execute(f"SET s3_access_key_id={_sql_string(creds.access_key_id)}")
    conn.execute(f"SET s3_secret_access_key={_sql_string(creds.secret_access_key)}")
    conn.execute(f"SET s3_endpoint={_sql_string(creds.endpoint)}")
    conn.execute(f"SET s3_region={_sql_string(creds.region)}")
    conn.execute(f"SET s3_url_style={_sql_string(creds.url_style)}")
    conn.execute(f"SET s3_use_ssl={'true' if creds.use_ssl else 'false'}")
    logger.info(
        "duckdb httpfs configured (legacy SET) endpoint=%s region=%s url_style=%s",
        creds.endpoint,
        creds.region,
        creds.url_style,
    )


class S3ConfigState:
    """**Per-connection** httpfs/S3 配置状态。

    #P0-5：S3 credential 是 connection 级属性，不是进程全局。`configure_fresh()`
    给 isolated 连接配完凭证**绝不能**设置全局短路——否则共享连接会误以为自己也配了。
    这里用 ``WeakKeyDictionary[conn, fingerprint]`` 按连接记录，连接 GC 自动清理。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_conn: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()

    def ensure(self, conn) -> None:
        """对本连接：同指纹跳过；指纹变更（含从未配置）则重新注入。"""
        creds = resolve_s3_credentials()
        fp = _creds_fingerprint(creds)
        with self._lock:
            if self._by_conn.get(conn) == fp:
                return
            try:
                apply_s3_credentials(conn, creds)
            except Exception as exc:
                raise ValidationError(
                    f"DuckDB httpfs 配置失败: {exc}. "
                    "请确认已安装 duckdb 且 LOAD httpfs 可用，并检查 COS 凭证/endpoint。"
                ) from exc
            self._by_conn[conn] = fp

    def configure_fresh(self, conn) -> None:
        """对新建 ``:memory:`` 连接始终注入凭证（scoped sql 用）。

        只记录**本连接**，绝不影响其他连接状态（#P0-5）。
        """
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
            self._by_conn[conn] = fp

    def reset(self) -> None:
        with self._lock:
            self._by_conn.clear()


_s3_state = S3ConfigState()


def ensure_duckdb_s3(conn) -> None:
    _s3_state.ensure(conn)


def configure_fresh_duckdb_s3(conn) -> None:
    """为独立连接注入 S3（不受共享连接「已配置」短路影响）。"""
    _s3_state.configure_fresh(conn)


def reset_duckdb_s3_state() -> None:
    _s3_state.reset()
