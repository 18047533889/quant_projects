"""Mining preset / prod profile 与 ``datasets.yaml`` 登记契约校验。

本模块提供递归遍历 ``data_source`` 配置、收集 ``data_access`` 绑定、
校验物理列是否在 registry schema 中，以及 mining preset / prod profile /
PR5 数据集的联合契约审计入口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

# prod profile 双写目标 → 必须存在的 parametric 数据集
_MATERIALIZATION_DATASET_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "local": ("factor_lake",),
    "staging": ("factor_lake_staging",),
    "clickhouse": ("factor_lake",),
    "staging_clickhouse": ("factor_lake_staging", "factor_lake"),
}

_STAGING_TARGETS = frozenset({"staging", "staging_clickhouse"})


def iter_data_access_nodes(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """递归遍历配置树中所有 ``type=data_access`` 节点。

    Parameters
    ----------
    config : dict[str, Any]
        ``data_source`` 或 composite 子树配置。

    Yields
    ------
    dict[str, Any]
        每个 ``type=data_access`` 节点（含 ``dataset``、``fields`` 等）。
    """
    if not isinstance(config, dict):
        return
    if str(config.get("type", "")).lower() == "data_access":
        yield config
    for key in ("sources", "panels", "fields"):
        child = config.get(key)
        if isinstance(child, dict):
            for sub in child.values():
                if isinstance(sub, dict):
                    yield from iter_data_access_nodes(sub)
    for key in ("anchor", "left", "right"):
        child = config.get(key)
        if isinstance(child, dict):
            yield from iter_data_access_nodes(child)


def collect_data_access_bindings(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """``dataset`` → ``{logical_field: physical_column}``（合并同名 dataset 映射）。

    Parameters
    ----------
    config : dict[str, Any]
        完整或部分 ``data_source`` 配置树。

    Returns
    -------
    dict[str, dict[str, str]]
        按 dataset 名聚合的逻辑字段到物理列映射。
    """
    bindings: dict[str, dict[str, str]] = {}
    for node in iter_data_access_nodes(config):
        dataset = str(node.get("dataset") or "").strip()
        if not dataset:
            continue
        fields = node.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        bucket = bindings.setdefault(dataset, {})
        for logical, physical in fields.items():
            bucket[str(logical)] = str(physical)
    return bindings


def _load_registry(config_path: Path | None = None):
    """加载 ``datasets.yaml`` registry（默认路径由 ``data_access`` 决定）。"""
    from data_access.registry import load_registry

    return load_registry(config_path)


def validate_dataset_field_bindings(
    bindings: dict[str, dict[str, str]],
    *,
    registry=None,
) -> list[str]:
    """校验 dataset 已登记且 ``fields`` 物理列在 schema 中（schema 为空则跳过列检）。

    Parameters
    ----------
    bindings : dict[str, dict[str, str]]
        ``collect_data_access_bindings`` 产出的 dataset → fields 映射。
    registry : Any, optional
        已加载的 registry；默认从 ``datasets.yaml`` 读取。

    Returns
    -------
    list[str]
        违规说明列表；空列表表示通过。
    """
    violations: list[str] = []
    reg = registry or _load_registry()
    for dataset, fields in sorted(bindings.items()):
        if dataset not in reg:
            violations.append(f"dataset {dataset!r} 未在 datasets.yaml 登记")
            continue
        ds = reg.get(dataset)
        schema = getattr(ds, "schema", None) or {}
        if not schema:
            continue
        schema_cols = {str(k) for k in schema}
        for logical, physical in sorted(fields.items()):
            if physical not in schema_cols:
                violations.append(
                    f"{dataset}: field {logical!r} → column {physical!r} "
                    f"不在 schema（已声明 {sorted(schema_cols)}）"
                )
    return violations


def audit_mining_dataset_contract(*, registry=None) -> dict[str, Any]:
    """审计 mining 默认 preset 引用的全部 ``data_access`` 数据集。

    Returns
    -------
    dict[str, Any]
        含 ``ok``、``violations``、``datasets_checked``、``by_preset`` 等。
    """
    from api.mining_integration import default_mining_data_source_presets

    presets = default_mining_data_source_presets()
    all_bindings: dict[str, dict[str, str]] = {}
    by_preset: dict[str, list[str]] = {}
    for name, cfg in presets.items():
        bindings = collect_data_access_bindings(cfg)
        by_preset[name] = sorted(bindings)
        for ds, fields in bindings.items():
            merged = all_bindings.setdefault(ds, {})
            merged.update(fields)

    violations = validate_dataset_field_bindings(all_bindings, registry=registry)
    reg = registry or _load_registry()
    datasets_checked = sorted(all_bindings)
    return {
        "ok": not violations,
        "violations": violations,
        "datasets_checked": datasets_checked,
        "datasets_registered": all(ds in reg for ds in datasets_checked),
        "by_preset": by_preset,
    }


def audit_prod_profile_contract(*, profile_name: str = "prod") -> dict[str, Any]:
    """审计 prod profile 的 materialization 目标与 ``datasets.yaml`` 一致。

    Parameters
    ----------
    profile_name : str
        runtime profile 名称（默认 ``"prod"``）。

    Returns
    -------
    dict[str, Any]
        含 ``ok``、``materialization_target``、``violations`` 等。
    """
    from runtime.config import load_profile

    profile = load_profile(profile_name)
    mat = profile.get("materialization") or {}
    target = str(mat.get("target") or "local").strip().lower()
    violations: list[str] = []

    required = _MATERIALIZATION_DATASET_REQUIREMENTS.get(target)
    if required is None:
        violations.append(f"未知 materialization.target={target!r}")
    else:
        reg = _load_registry()
        for ds_name in required:
            if ds_name not in reg:
                violations.append(
                    f"profile {profile_name!r} target={target!r} 需要数据集 {ds_name!r}，"
                    "但未在 datasets.yaml 登记"
                )

    if target in {"clickhouse", "staging_clickhouse"} and not mat.get("clickhouse_table"):
        violations.append(f"target={target!r} 缺少 materialization.clickhouse_table")

    if target in _STAGING_TARGETS and not mat.get("lake_root"):
        violations.append("staging 类 target 建议显式配置 materialization.lake_root")

    ch_table = str(mat.get("clickhouse_table") or "").strip()
    if ch_table and target.endswith("clickhouse"):
        # 与 prod 默认表名一致即可；不强制 datasets.yaml 登记 CH 表
        pass

    return {
        "ok": not violations,
        "profile": profile_name,
        "materialization_target": target,
        "violations": violations,
    }


# PR5/PR6 data_access 登记数据集（读端迁移；schema 契约门禁）
PR5_DATASETS: tuple[str, ...] = (
    "daily_market_summary",
    "stocks_floats",
    "financials_ratios",
    "us_stocks_sip_minute_aggs",
    "fundamentals_balance_sheet",
    "fundamentals_income_statement",
    "fundamentals_cash_flow_statement",
    "fundamentals_short_interest",
    "fundamentals_short_volume",
    "us_stocks_sip_quotes",
    "us_stocks_sip_trades",
)


def audit_pr5_datasets_contract(*, registry=None) -> dict[str, Any]:
    """PR5 新增 datasets.yaml 条目存在且含 schema。

    Returns
    -------
    dict[str, Any]
        含 ``ok``、``violations``、``datasets``（``PR5_DATASETS`` 列表）。
    """
    reg = registry or _load_registry()
    violations: list[str] = []
    for name in PR5_DATASETS:
        if name not in reg:
            violations.append(f"PR5 dataset {name!r} 未在 datasets.yaml 登记")
            continue
        ds = reg.get(name)
        schema = getattr(ds, "schema", None) or {}
        if not schema:
            violations.append(f"PR5 dataset {name!r} 缺少 schema 块")
    return {"ok": not violations, "violations": violations, "datasets": list(PR5_DATASETS)}


def audit_full_datasets_contract(*, profile_name: str = "prod") -> dict[str, Any]:
    """Mining preset + prod profile + PR5 联合契约审计（CI / nightly 入口）。

    Parameters
    ----------
    profile_name : str
        用于 ``audit_prod_profile_contract`` 的 profile 名。

    Returns
    -------
    dict[str, Any]
        聚合 ``mining``、``prod_profile``、``pr5`` 子报告及总 ``violations``。
    """
    mining = audit_mining_dataset_contract()
    prod = audit_prod_profile_contract(profile_name=profile_name)
    pr5 = audit_pr5_datasets_contract()
    violations = (
        list(mining.get("violations") or [])
        + list(prod.get("violations") or [])
        + list(pr5.get("violations") or [])
    )
    return {
        "ok": not violations,
        "mining": mining,
        "prod_profile": prod,
        "pr5": pr5,
        "violations": violations,
    }
