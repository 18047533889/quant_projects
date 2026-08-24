# -*- coding: utf-8 -*-
"""ClickHouse 独立运行时认证 harness 回归测试。

验证 :mod:`backend.clickhouse_runtime_cert` 在**没有 live ClickHouse 服务器**时
对每个 canonical 返回 ``NOT_RUN``（fail-closed），并且**绝不**继承 DuckDB 的
PASS/parity 结论。

- 探测默认不发起任何网络连接（``CLICKHOUSE_ENABLE_CONNECT_PROBE`` 未设置）。
- 伪造 DuckDB PASS 证据（monkeypatch 运行时 DuckDB 证据集合含全部 canonical）
  后运行 harness，必须仍得到 ``NOT_RUN``，证明与 DuckDB 证书相互独立。
- 三个测试全部在无服务器环境下执行，串行（见 pyproject pytest 默认）。
"""
from __future__ import annotations

import os
import sys

import pytest

from factor_engine.workspace_paths import quant_projects_root

REPO_ROOT = str(quant_projects_root())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 线程环境变量（串行；见约束）
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")


@pytest.fixture(autouse=True)
def _clear_ch_probe_env(monkeypatch: pytest.MonkeyPatch):
    """确保探测 fail-closed：清空可能启用的 CH 连接探测变量。"""
    monkeypatch.delenv("CLICKHOUSE_ENABLE_CONNECT_PROBE", raising=False)


def test_cert_harness_detects_no_server_and_marks_not_run() -> None:
    """无 live 服务器 → 整体 NOT_RUN 且每个 canonical 均 NOT_RUN。"""
    from factor_engine.backend.clickhouse_runtime_cert import (
        CLICKHOUSE_RUNTIME_CANONICALS,
        run_clickhouse_runtime_cert,
    )

    report = run_clickhouse_runtime_cert()
    assert report["status"] == "NOT_RUN"
    assert report["client_available"] is False
    canonicals = report["canonicals"]
    assert set(canonicals) == set(CLICKHOUSE_RUNTIME_CANONICALS)
    assert all(entry["status"] == "NOT_RUN" for entry in canonicals.values())
    assert all("no live ClickHouse server" in entry["detail"] for entry in canonicals.values())


def test_cert_harness_never_inherits_duckdb_pass() -> None:
    """即使 DuckDB 证据（静态）声称全部 PASS，ClickHouse 认证仍为 NOT_RUN。

    证明 harness 的认证面独立于 DuckDB：不读取 DuckDB 证据集合，
    也不把 DuckDB 的 PASS 映射为 ClickHouse 的 PASS。
    """
    import factor_engine.backend.primitive_evidence as primitive_evidence
    from factor_engine.backend.clickhouse_runtime_cert import (
        CLICKHOUSE_RUNTIME_CANONICALS,
        run_clickhouse_runtime_cert,
    )

    fake = frozenset(CLICKHOUSE_RUNTIME_CANONICALS)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(primitive_evidence, "DUCKDB_REAL_SQL_VERIFIED", fake)
    monkeypatch.setattr(primitive_evidence, "DUCKDB_REFERENCE_PARITY_VERIFIED", fake)
    monkeypatch.setattr(primitive_evidence, "DUCKDB_EDGE_VERIFIED", fake)
    monkeypatch.setattr(primitive_evidence, "DUCKDB_NAN_EDGE_VERIFIED", fake)
    try:
        report = run_clickhouse_runtime_cert()
    finally:
        monkeypatch.undo()

    assert report["status"] == "NOT_RUN"
    assert all(entry["status"] == "NOT_RUN" for entry in report["canonicals"].values())


def test_probe_client_returns_none_without_live_server() -> None:
    """probe_client 在默认（未显式允许网络连接）下返回 None。

    不依赖任何环境：即使存在 CH 配置，只要未显式开启连接探测，就 fail-closed。
    """
    from factor_engine.backend.clickhouse_runtime_cert import probe_client

    assert probe_client() is None


def test_cert_canonicals_cover_required_semantics() -> None:
    """canonical 集覆盖任务要求的语义维度（NULL/NaN/Inf、frame、tie、quantile、DateTime64、decimal、分组排序）。"""
    from factor_engine.backend.clickhouse_runtime_cert import CLICKHOUSE_RUNTIME_CANONICALS

    required = {
        "is_nan",        # NaN
        "protected_div",  # Inf（除零保护）
        "ts_rank",       # 窗口 frame
        "rank",          # rank tie
        "cs_rank",
        "cs_quantile",   # quantile
        "group_percentile",
        "ts_delay",      # DateTime64 / 时区
        "divide",        # decimal
        "group_mean",    # 分组排序
        "group_rank",
    }
    missing = required - set(CLICKHOUSE_RUNTIME_CANONICALS)
    assert not missing, f"missing required canonical dimensions: {sorted(missing)}"
