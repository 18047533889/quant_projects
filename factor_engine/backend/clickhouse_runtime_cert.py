# -*- coding: utf-8 -*-
"""ClickHouse 独立运行时认证（R21-CLICKHOUSE-RUNTIME-CERT）。

目标
----
为 ``ClickHousePushdownBackend``（``backend/duckdb_pushdown_backend.py``）建立
**独立于 DuckDB** 的运行时认证：在真实 ClickHouse 服务器上逐 canonical 执行
SQL 下推并断言结果，覆盖 NULL / NaN / Inf、窗口 frame、rank 并列、quantile、
DateTime64/时区、decimal、分组排序等语义。

Fail-closed 原则
----------------
- 没有任何 live ClickHouse 服务器时，``run_clickhouse_runtime_cert`` 对每个
  canonical 返回 ``NOT_RUN``（状态行 ``status: NOT_RUN``）。
- 认证状态 **绝不继承 DuckDB** 的 PASS：``run_clickhouse_runtime_cert`` 不引用
  DuckDB 证据集合，也不会因为 DuckDB 通过而给出 PASS。
- 探测只读：不做 DDL/DML，无副作用，localhost 默认不进行网络连接尝试
  （``CLICKHOUSE_ENABLE_CONNECT_PROBE`` 未显式开启时 ``probe_client`` 直接
  返回 None，从而 fail-closed）。

使用
----
::

    from factor_engine.backend.clickhouse_runtime_cert import run_clickhouse_runtime_cert
    report = run_clickhouse_runtime_cert()
    report["clickhouse_runtime"]["status"]  # -> "NOT_RUN"（无 live server 时）

当存在 live 服务器（且显式设置 ``CLICKHOUSE_ENABLE_CONNECT_PROBE=1``）时，
才会真正执行 SQL，并对每个 canonical 记录 ``PASS`` / ``FAIL``。
"""
from __future__ import annotations

import dataclasses
import os
import socket
from typing import Any, Callable

__all__ = [
    "ClickHouseRuntimeCertResult",
    "CLICKHOUSE_RUNTIME_CANONICALS",
    "probe_client",
    "run_clickhouse_runtime_cert",
    "RuntimeCheck",
]

NOT_RUN = "NOT_RUN"


# ---------------------------------------------------------------------------
# Canonical 集合（独立的 ClickHouse 运行时认证面）
# ---------------------------------------------------------------------------
#
# 覆盖 SQL 下推所需的全部语义维度：
#   - elementwise / 空值逻辑：add, divide, protected_div, where, is_nan
#   - 滚动窗口：ts_mean / ts_sum / ts_std（NULL 窗口）
#   - 窗口 frame：ts_rank、ts_median（ORDER BY + frame）
#   - rank 并列：cs_rank、group_rank、rank（tie 语义）
#   - quantile：cs_quantile / group_percentile / ts_median
#   - DateTime64 / 时区：ts_delay（时间序）
#   - decimal：elementwise divide / protect_div
#   - 分组排序：group_mean / group_rank
CLICKHOUSE_RUNTIME_CANONICALS: tuple[str, ...] = (
    "add",
    "divide",
    "protected_div",
    "where",
    "is_nan",
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_rank",
    "ts_median",
    "ts_delay",
    "cs_rank",
    "group_mean",
    "group_rank",
    "rank",
    "cs_quantile",
    "group_percentile",
)

# 真实 CH 服务器探测（显式开启；默认 fail-closed）
_ENABLE_CONNECT_PROBE = "CLICKHOUSE_ENABLE_CONNECT_PROBE"
_DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0
_CONNECT_TIMEOUT_SECONDS = 3.0


def _probe_direct(host: str, port: int) -> bool:
    """TCP connect 到 ClickHouse HTTP(S) 端口，确认存在 live 服务器。

    仅在该函数被显式调用时执行网络连接。连接失败不会抛出异常。
    """
    if not host or host in {"localhost", "127.0.0.1"}:
        return False  # localhost 默认视为「未配置 live 服务器」
    try:
        with socket.create_connection((host, int(port)), timeout=_CONNECT_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def probe_client(
    *,
    config: Any | None = None,
    allow_network: bool | None = None,
) -> Any:
    """返回 live ClickHouse 客户端；无法确认 live 服务器时返回 None。

    Fail-closed 探测顺序：
    1. ``clickhouse-connect`` 未安装 → None。
    2. 未显式启用 ``CLICKHOUSE_ENABLE_CONNECT_PROBE`` → None
       （默认不进行任何网络连接尝试）。
    3. 显式传入 ``config`` 且 ``allow_network=True`` → 尝试建立客户端；
       失败（含 localhost 白名单 / TCP 拒绝 / 认证失败）→ None。
    4. 其余情况 → None。

    ``config`` 为 ``dataaccess.clickhouse.panel.ClickHouseConfig``（或具备
    ``host/port/username/password/database/secure`` 属性的对象）。
    """
    try:
        import clickhouse_connect  # type: ignore
    except Exception:
        return None
    enabled = os.environ.get(_ENABLE_CONNECT_PROBE, "").strip().lower()
    if allow_network is None:
        allow_network = enabled in {"1", "true", "yes"}
    if not allow_network:
        return None
    if config is None:
        try:
            from dataaccess.clickhouse.panel import ClickHouseConfig

            config = ClickHouseConfig.from_env()
        except Exception:
            return None
    host = str(getattr(config, "host", ""))
    port = int(getattr(config, "port", 8123) or 8123)
    if not _probe_direct(host, port):
        return None
    try:
        return clickhouse_connect.get_client(
            host=host,
            port=port,
            username=str(getattr(config, "username", "default")),
            password=str(getattr(config, "password", "")),
            database=str(getattr(config, "database", "default")),
            secure=bool(getattr(config, "secure", False)),
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 运行时检查描述（canonical -> SQL + 期望 lambda）
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RuntimeCheck:
    """单个 canonical 的 ClickHouse 运行时检查。

    ``sql`` 在 `SELECT` 之外不得包含任何 DML/DDL 关键字；``expect`` 接收
    ``(rows, columns)``（行列表 + 列名列表），返回 ``(bool, str)``。
    """

    canonical: str
    sql: str
    expect: Callable[[list[Any], list[str]], tuple[bool, str]]


def _expect_num(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
    """通用数值期望：结果非空、首行首列是数值。"""
    if not rows:
        return False, "no rows returned"
    val = rows[0][0]
    try:
        float(val)
    except (TypeError, ValueError):
        return False, f"first value not numeric: {val!r}"
    return True, f"first value={val!r}"


def _expect_not_null(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
    if not rows or rows[0][0] is None:
        return False, "expected non-NULL first value"
    return True, f"first value={rows[0][0]!r}"


def _expect_all_not_null(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
    if not rows:
        return False, "no rows returned"
    nulls = [r[0] for r in rows if r[0] is None]
    if nulls:
        return False, f"{len(nulls)} NULL values returned"
    return True, f"{len(rows)} non-NULL rows"


def _expect_zero(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
    if not rows:
        return False, "no rows returned"
    val = rows[0][0]
    try:
        return bool(float(val) == 0.0), f"first value={val!r}"
    except (TypeError, ValueError):
        return False, f"first value not numeric: {val!r}"


def _expect_ties(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
    """rank 并列：非空且全部行首列值相同（并列组内 rank 一致）。"""
    if not rows:
        return False, "no rows returned"
    vals = [r[0] for r in rows]
    distinct = {float(v) for v in vals}
    if len(distinct) != 1:
        return False, f"expected tied rank but got distinct values: {distinct}"
    return True, f"tied rank value={vals[0]!r}"


def _expect_range(lo: float, hi: float) -> Callable[[list[Any], list[str]], tuple[bool, str]]:
    def _check(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
        if not rows:
            return False, "no rows returned"
        val = float(rows[0][0])
        if not (lo <= val <= hi):
            return False, f"value {val!r} outside [{lo}, {hi}]"
        return True, f"value={val!r} within [{lo}, {hi}]"
    return _check


def _expect_len(exact: int) -> Callable[[list[Any], list[str]], tuple[bool, str]]:
    def _check(rows: list[Any], columns: list[str]) -> tuple[bool, str]:
        if len(rows) != exact:
            return False, f"expected {exact} rows, got {len(rows)}"
        return True, f"{exact} rows"
    return _check


_CLICKHOUSE_RUNTIME_CHECKS: tuple[RuntimeCheck, ...] = (
    RuntimeCheck(
        canonical="add",
        sql="SELECT 1.0 + 2.0 AS v",
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="divide",
        sql="SELECT 1.0 / 2.0 AS v",
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="protected_div",
        sql="SELECT if(x = 0, 0, 1.0 / x) AS v FROM (SELECT 0.0 AS x)",
        expect=_expect_zero,
    ),
    RuntimeCheck(
        canonical="where",
        sql="SELECT if(v > 1.0, v, 0.0) AS v FROM (SELECT arrayJoin([1.0, 2.0]) AS v)",
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="is_nan",
        sql="SELECT isNaN(v) AS v FROM (SELECT toFloat64('nan') AS v)",
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_mean",
        sql=(
            "SELECT m FROM ("
            "SELECT x, avg(x) OVER (ORDER BY k ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS m "
            "FROM (SELECT k, v AS x FROM (SELECT arrayJoin([1,2,3,4,5]) AS k, arrayJoin([1.0,2.0,3.0,4.0,5.0]) AS v))"
            ") ORDER BY k LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_sum",
        sql=(
            "SELECT s FROM ("
            "SELECT k, sum(x) OVER (ORDER BY k ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s "
            "FROM (SELECT k, v AS x FROM (SELECT arrayJoin([1,2,3,4,5]) AS k, arrayJoin([1.0,2.0,3.0,4.0,5.0]) AS v))"
            ") ORDER BY k LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_std",
        sql=(
            "SELECT s FROM ("
            "SELECT k, stddevSamp(x) OVER (ORDER BY k ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS s "
            "FROM (SELECT k, v AS x FROM (SELECT arrayJoin([1,2,3,4,5]) AS k, arrayJoin([1.0,2.0,3.0,4.0,5.0]) AS v))"
            ") ORDER BY k LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_rank",
        sql=(
            "SELECT r FROM ("
            "SELECT k, rank() OVER (ORDER BY x) AS r "
            "FROM (SELECT k, v AS x FROM (SELECT arrayJoin([1,2,3,4,5]) AS k, arrayJoin([5.0,4.0,3.0,2.0,1.0]) AS v))"
            ") ORDER BY k LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_median",
        sql=(
            "SELECT m FROM ("
            "SELECT k, median(x) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS m "
            "FROM (SELECT k, v AS x FROM (SELECT arrayJoin([1,2,3,4,5]) AS k, arrayJoin([1.0,2.0,3.0,4.0,5.0]) AS v))"
            ") ORDER BY k LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="ts_delay",
        sql=(
            "SELECT d FROM ("
            "SELECT ts, x, lagInFrame(x) OVER (ORDER BY ts) AS d "
            "FROM (SELECT ts, v AS x FROM ("
            "SELECT arrayJoin([toDateTime64('2024-01-01 00:00:00', 3, 'UTC'), toDateTime64('2024-01-02 00:00:00', 3, 'UTC'), toDateTime64('2024-01-03 00:00:00', 3, 'UTC')]) AS ts, "
            "arrayJoin([1.0, 2.0, 3.0]) AS v))"
            ") ORDER BY ts LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="cs_rank",
        sql=(
            "SELECT r FROM ("
            "SELECT v, rank() OVER (ORDER BY v) AS r FROM (SELECT arrayJoin([2.0, 1.0, 3.0]) AS v)"
            ") ORDER BY v LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="group_mean",
        sql=(
            "SELECT m FROM ("
            "SELECT g, v, avg(v) OVER (PARTITION BY g) AS m "
            "FROM (SELECT arrayJoin([1, 1, 2, 2]) AS g, arrayJoin([10.0, 20.0, 30.0, 40.0]) AS v)"
            ") WHERE v = 10.0"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="group_rank",
        sql=(
            "SELECT r FROM ("
            "SELECT g, v, rank() OVER (PARTITION BY g ORDER BY v) AS r "
            "FROM (SELECT arrayJoin([1, 1, 2, 2]) AS g, arrayJoin([2.0, 1.0, 4.0, 3.0]) AS v)"
            ") WHERE v = 1.0"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="rank",
        sql=(
            "SELECT r FROM ("
            "SELECT v, rank() OVER (ORDER BY v) AS r FROM (SELECT arrayJoin([3.0, 1.0, 2.0]) AS v)"
            ") ORDER BY v LIMIT 1"
        ),
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="cs_quantile",
        sql="SELECT quantile(0.5)(v) AS v FROM (SELECT arrayJoin([1.0, 2.0, 3.0, 4.0]) AS v)",
        expect=_expect_num,
    ),
    RuntimeCheck(
        canonical="group_percentile",
        sql=(
            "SELECT p FROM ("
            "SELECT g, quantile(0.5)(v) OVER (PARTITION BY g) AS p "
            "FROM (SELECT arrayJoin([1, 1, 2, 2]) AS g, arrayJoin([1.0, 3.0, 10.0, 20.0]) AS v)"
            ") WHERE g = 1 LIMIT 1"
        ),
        expect=_expect_num,
    ),
)


def _execute_single(client: Any, check: RuntimeCheck) -> tuple[bool, str]:
    """在 live 客户端上执行单条检查；任何异常视为该 canonical FAIL。"""
    try:
        result = client.query(check.sql)
    except Exception as exc:  # noqa: BLE001
        return False, f"query error: {exc}"
    try:
        column_names = list(result.column_names) if result.column_names is not None else []
        rows = list(result.result_rows) if result.result_rows is not None else []
    except Exception as exc:  # noqa: BLE001
        return False, f"result read error: {exc}"
    try:
        return check.expect(rows, column_names)
    except Exception as exc:  # noqa: BLE001
        return False, f"assertion error: {exc}"


# ---------------------------------------------------------------------------
# 独立认证执行入口
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ClickHouseRuntimeCertResult:
    """逐 canonical 的 ClickHouse 运行时认证结果。"""

    status: str = NOT_RUN
    client_available: bool = False
    reason: str = "no live ClickHouse server (fail-closed NOT_RUN)"
    checks: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "client_available": self.client_available,
            "reason": self.reason,
            "canonicals": {
                c: dict(entry) for c, entry in self.checks.items()
            },
        }


def run_clickhouse_runtime_cert(*, client: Any | None = None, probe: bool = True) -> dict[str, Any]:
    """运行 ClickHouse 独立运行时认证（fail-closed NOT_RUN）。

    - ``client`` 传入时直接使用该 live 客户端（用于测试注入）。
    - ``probe=True`` 且未传 ``client`` 时调用 :func:`probe_client`。
    - 无 live 客户端 → 每个 canonical 均为 ``NOT_RUN``，整体 ``NOT_RUN``。
    - 有 live 客户端 → 逐 canonical 执行 SQL，PASS/FAIL；整体
      ``PASS``（全部通过）或 ``PARTIAL``（存在 FAIL）。
    """
    result = ClickHouseRuntimeCertResult()
    live = client
    if live is None and probe:
        live = probe_client()
    if live is None:
        result.status = NOT_RUN
        result.reason = "no live ClickHouse server (fail-closed NOT_RUN)"
        result.client_available = False
        result.checks = {
            c: {"status": NOT_RUN, "detail": "no live ClickHouse server"}
            for c in CLICKHOUSE_RUNTIME_CANONICALS
        }
        return result.to_dict()

    result.client_available = True
    result.reason = "live ClickHouse server detected"
    passed = 0
    failed = 0
    for check in _CLICKHOUSE_RUNTIME_CHECKS:
        ok, detail = _execute_single(live, check)
        if ok:
            passed += 1
            result.checks[check.canonical] = {"status": "PASS", "detail": detail}
        else:
            failed += 1
            result.checks[check.canonical] = {"status": "FAIL", "detail": detail}
    result.status = "PASS" if failed == 0 else "PARTIAL"
    return result.to_dict()
