# -*- coding: utf-8 -*-
"""
COS 远程直读（DuckDB httpfs + S3 兼容 API）。

与 ``cos_mirror``（先下载到本地再读）并列，由环境变量 ``DATA_ACCESS_COS_READ_MODE`` 切换：

- ``mirror``（默认）：读前 ``ensure_local_mirror``，DuckDB 读本地 Parquet
- ``remote``：跳过下载，DuckDB ``read_parquet('s3://...')`` 直连 COS
- ``auto``：``time_range`` 内本地文件齐全则用本地，否则走 remote

远程模式需要 COS S3 凭证（``COS_SECRET_ID`` / ``COS_SECRET_KEY`` 或 AWS 风格变量）
以及 ``DATA_ACCESS_COS_S3_ENDPOINT``（如 ``cos.ap-guangzhou.myqcloud.com``）。
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .mirror import (
    DATASET_MIRROR_REGISTRY,
    MirrorSpec,
    _cos_table_uri,
    _iter_dates,
    _iter_years,
    _parse_date,
    mirror_spec_for_dataset,
)
from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    from data_access.registry import Dataset, StaticDataset

logger = logging.getLogger("data_access.cos_remote")

_VALID_MODES = frozenset({"mirror", "remote", "auto"})


@dataclass(frozen=True)
class S3Credentials:
    access_key_id: str
    secret_access_key: str
    endpoint: str
    region: str
    use_ssl: bool = True
    url_style: str = "path"


def cos_read_mode() -> str:
    """``mirror`` | ``remote`` | ``auto``。"""
    raw = os.environ.get("DATA_ACCESS_COS_READ_MODE", "mirror").strip().lower()
    if raw not in _VALID_MODES:
        raise ValidationError(
            f"无效的 DATA_ACCESS_COS_READ_MODE={raw!r}，允许: {sorted(_VALID_MODES)}"
        )
    return raw


def cos_uri_to_s3_uri(uri: str) -> str:
    """``cos://bucket/key`` → ``s3://bucket/key``（DuckDB httpfs 用）。"""
    if uri.startswith("cos://"):
        return "s3://" + uri[len("cos://") :]
    if uri.startswith("s3://"):
        return uri
    raise ValidationError(f"非 COS/S3 URI: {uri!r}")


def allowed_s3_prefixes() -> tuple[str, ...]:
    """已登记 mirror 数据集对应的 S3 路径前缀（白名单）。"""
    prefixes: set[str] = set()
    for spec in DATASET_MIRROR_REGISTRY.values():
        prefixes.add(cos_uri_to_s3_uri(spec.cos_prefix.rstrip("/")) + "/")
        if spec.table:
            prefixes.add(
                cos_uri_to_s3_uri(_cos_table_uri(spec)).rstrip("/") + "/"
            )
    return tuple(sorted(prefixes))


def authorize_s3_path(path: str, extra_prefixes: Sequence[str] | None = None) -> None:
    """远程路径必须落在已登记 COS 前缀下。

    ``extra_prefixes``（#P0-2）：generic ``storage.source`` 数据集把自己的声明根
    （归一化后的 ``s3://`` 前缀）作为授权边界一并校验，不要求手工 mirror 登记。
    """
    normalized = path.rstrip("/")
    prefixes: set[str] = set(allowed_s3_prefixes())
    if extra_prefixes:
        prefixes.update(p.rstrip("/") for p in extra_prefixes if p)
    for prefix in prefixes:
        base = prefix.rstrip("/")
        if normalized == base or normalized.startswith(base + "/"):
            return
    hint = "\n  ".join(sorted(prefixes)[:5])
    raise ValidationError(
        f"S3 路径越界（不在已登记 COS mirror 前缀或声明 storage.source.uri 下）："
        f"{path}\n示例允许前缀：\n  {hint}"
    )


def _read_cos_cli_credentials() -> tuple[str, str]:
    """Fall back to the coscli/clean-cos-ro config (``~/.cos.yaml``).

    The remote path must not silently require env vars when the team's standard
    COS CLI already has credentials configured.  Only ``secretid``/``secretkey``
    from ``cos.base`` are read; nothing is logged.
    """
    import shutil

    explicit = (
        os.environ.get("DATA_ACCESS_COS_YAML")
        or os.environ.get("COS_CONFIG_FILE")
        or ""
    ).strip()
    # 显式指定配置路径时以它为权威（便于测试/CI 覆盖）；否则默认读 ~/.cos.yaml。
    candidates = [explicit] if explicit else [str(Path.home() / ".cos.yaml")]
    for path in candidates:
        if not path or not Path(path).is_file():
            continue
        try:
            import yaml

            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            base = (payload or {}).get("cos", {}).get("base", {}) or {}
            secret_id = str(base.get("secretid") or "").strip()
            secret_key = str(base.get("secretkey") or "").strip()
            if secret_id and secret_key:
                return secret_id, secret_key
        except Exception:
            continue
    return "", ""


def resolve_s3_credentials() -> S3Credentials:
    """解析腾讯云 COS / 通用 S3 凭证。

    优先级：环境变量 → coscli/clean-cos-ro 配置（``~/.cos.yaml``）。
    endpoint 缺省按 region 推断为 ``cos.<region>.myqcloud.com``。
    """
    access = (
        os.environ.get("COS_SECRET_ID")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("S3_ACCESS_KEY_ID")
        or ""
    ).strip()
    secret = (
        os.environ.get("COS_SECRET_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("S3_SECRET_ACCESS_KEY")
        or ""
    ).strip()
    if not access or not secret:
        access, secret = _read_cos_cli_credentials()
    endpoint = (
        os.environ.get("DATA_ACCESS_COS_S3_ENDPOINT")
        or os.environ.get("COS_S3_ENDPOINT")
        or os.environ.get("S3_ENDPOINT")
        or ""
    ).strip().removeprefix("https://").removeprefix("http://")
    region = (
        os.environ.get("DATA_ACCESS_COS_S3_REGION")
        or os.environ.get("COS_REGION")
        or os.environ.get("S3_REGION")
        or "ap-guangzhou"
    ).strip()
    if not access or not secret:
        raise ValidationError(
            "COS 远程直读需要凭证：设置 COS_SECRET_ID + COS_SECRET_KEY，"
            "或 AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY，"
            "或提供 ~/.cos.yaml（coscli/clean-cos-ro 配置）。"
        )
    if not endpoint:
        # Tencent COS 默认 endpoint 由 region 推断。
        endpoint = f"cos.{region}.myqcloud.com"
    use_ssl = os.environ.get("DATA_ACCESS_COS_S3_USE_SSL", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    url_style = os.environ.get("DATA_ACCESS_COS_S3_URL_STYLE", "path").strip() or "path"
    return S3Credentials(
        access_key_id=access,
        secret_access_key=secret,
        endpoint=endpoint,
        region=region,
        use_ssl=use_ssl,
        url_style=url_style,
    )


def _s3_table_base(spec: MirrorSpec) -> str:
    return cos_uri_to_s3_uri(_cos_table_uri(spec)).rstrip("/")


def _remote_daily_paths(spec: MirrorSpec, time_range: tuple[Any, Any] | None) -> list[str]:
    base = _s3_table_base(spec)
    if time_range is None:
        authorize_s3_path(f"{base}/*.parquet")
        return [f"{base}/*.parquet"]
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValidationError(f"remote 模式无法解析 time_range: {time_range}")
    paths = [f"{base}/{day.isoformat()}.parquet" for day in _iter_dates(start, end)]
    for path in paths:
        authorize_s3_path(path)
    return paths


def _remote_single_full(spec: MirrorSpec) -> list[str]:
    path = f"{_s3_table_base(spec)}/{spec.file_name}"
    authorize_s3_path(path)
    return [path]


def _remote_root_file(spec: MirrorSpec) -> list[str]:
    path = cos_uri_to_s3_uri(f"{spec.cos_prefix.rstrip('/')}/{spec.file_name}")
    authorize_s3_path(path)
    return [path]


def _remote_hive_date_paths(spec: MirrorSpec, time_range: tuple[Any, Any] | None) -> list[str]:
    base = cos_uri_to_s3_uri(spec.cos_prefix.rstrip("/"))
    if time_range is None:
        path = f"{base}/date=*/{spec.file_name}"
        authorize_s3_path(base + "/")
        return [path]
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValidationError(f"remote 模式无法解析 time_range: {time_range}")
    paths = [
        f"{base}/date={day.isoformat()}/{spec.file_name}" for day in _iter_dates(start, end)
    ]
    for path in paths:
        authorize_s3_path(path)
    return paths


def _remote_hive_year_paths(spec: MirrorSpec, time_range: tuple[Any, Any] | None) -> list[str]:
    base = cos_uri_to_s3_uri(spec.cos_prefix.rstrip("/"))
    if time_range is None:
        path = f"{base}/year=*/{spec.file_name}"
        authorize_s3_path(base + "/")
        return [path]
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValidationError(f"remote 模式无法解析 time_range: {time_range}")
    paths = [
        f"{base}/year={year}/{spec.file_name}" for year in _iter_years(start, end)
    ]
    for path in paths:
        authorize_s3_path(path)
    return paths


def build_remote_paths(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    ds: "Dataset | None" = None,
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """按 mirror 布局生成 DuckDB 可读的 ``s3://`` 路径列表。

    #29：没有手工 mirror spec、但数据集声明了 ``storage.source.type=cos`` 时，
    从 storage 声明构建（factor lake / model outputs / features 等通用化）。
    """
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        if ds is not None and declares_cos_storage(ds):
            return _remote_paths_from_storage(ds, time_range=time_range, params=params)
        raise ValidationError(
            f"数据集 '{dataset_name}' 未配置 COS mirror 也未声明 storage.source=cos，"
            f"无法 remote 读取"
        )

    if spec.layout == "single_full":
        return _remote_single_full(spec)
    if spec.layout == "root_file":
        return _remote_root_file(spec)
    if spec.layout == "daily_parquet":
        return _remote_daily_paths(spec, time_range)
    if spec.layout == "hive_date":
        return _remote_hive_date_paths(spec, time_range)
    if spec.layout == "hive_year":
        return _remote_hive_year_paths(spec, time_range)
    raise ValidationError(f"未知 mirror layout: {spec.layout}")


def _remote_paths_from_storage(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None,
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """#29 从 ``storage.source`` 声明构建 COS 远程路径（无手工 mirror spec）。

    ``storage.source.uri`` 是 COS 前缀；``layout`` 决定路径形态（daily_parquet /
    hive_date / hive_year / plain）。ParametricDataset 用 params 填 glob 占位符。

    **#P0-2 / #P0-3**：所有远程路径离开 planner 前统一 normalize + authorize，
    生成 canonical ``s3://`` execution URI；params/glob format 错误、missing param、
    unknown layout 一律 ``ValidationError`` fail-closed——禁止宽泛 ``except`` 降级成
    ``**/*.parquet`` 扫描整个数据集。
    """
    from data_access.registry import ParametricDataset

    storage = getattr(ds, "storage", None) or {}
    src = storage.get("source", {})
    uri: str = ""
    layout: str = "plain"
    if isinstance(src, dict):
        uri = str(src.get("uri") or "")
        layout = str(src.get("layout") or getattr(ds, "layout", "plain"))
    elif isinstance(src, str):
        uri = str(src)
    uri = uri.rstrip("/")
    if not uri:
        raise ValidationError(f"数据集 '{ds.name}' 声明 storage.source=cos 但缺 uri")
    layout = str(layout).strip().lower()

    if layout not in {"plain", "daily_parquet", "hive_date", "hive_year"}:
        raise ValidationError(
            f"数据集 '{ds.name}' storage.source.layout={layout!r} 未知；"
            f"允许 plain|daily_parquet|hive_date|hive_year（fail-closed）"
        )

    glob = getattr(ds, "glob", None)
    if isinstance(ds, ParametricDataset):
        validated = dict(params or {})
        if not ds.glob_template:
            raise ValidationError(
                f"数据集 '{ds.name}' 是 ParametricDataset 但未声明 glob_template；"
                "无法推导 glob（fail-closed，禁止扫描整个数据集）"
            )
        try:
            glob = ds.glob_template.format(**validated)
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            raise ValidationError(
                f"数据集 '{ds.name}' glob_template 参数格式化失败: {exc}。"
                f"缺参数或非法参数，禁止降级全库扫描"
            ) from exc
        # 防止 .format 未消费的占位符/残留花括号产生意外路径形态
        if "{" in glob or "}" in glob:
            raise ValidationError(
                f"数据集 '{ds.name}' glob_template 格式化后仍含占位符: {glob!r}"
            )
    if glob is None:
        glob = "**/*.parquet"

    # 授权边界 = 数据集声明的存储根（cos:// → s3:// 归一化后）。
    auth_prefix = cos_uri_to_s3_uri(uri)
    if layout == "daily_parquet" and time_range and time_range[0]:
        start, end = time_range
        lo = _parse_date(start)
        hi = _parse_date(end)
        if lo is not None and hi is not None:
            return _resolve_storage_paths(
                (f"{uri}/{day.isoformat()}.parquet" for day in _iter_dates(lo, hi)),
                authorized_prefix=auth_prefix,
            )
    if layout in {"hive_date", "hive_year"} and time_range and time_range[0]:
        start, end = time_range
        lo = _parse_date(start)
        hi = _parse_date(end)
        if lo is not None and hi is not None:
            if layout == "hive_date":
                return _resolve_storage_paths(
                    (
                        f"{uri}/date={day.isoformat()}/data.parquet"
                        for day in _iter_dates(lo, hi)
                    ),
                    authorized_prefix=auth_prefix,
                )
            return _resolve_storage_paths(
                (f"{uri}/year={y}/data.parquet" for y in _iter_years(lo, hi)),
                authorized_prefix=auth_prefix,
            )
    return _resolve_storage_paths([f"{uri}/{glob}"], authorized_prefix=auth_prefix)


def _resolve_storage_paths(
    paths: Sequence[str],
    *,
    authorized_prefix: str | None = None,
) -> list[str]:
    """#P0-2 / #P0-4：统一 normalize + authorize + canonical s3:// execution URI。

    - ``cos://`` / ``s3://`` 统一为 canonical ``s3://bucket/key``（DuckDB httpfs）；
    - 每个路径经 ``authorize_s3_path`` 白名单校验后才放行。``authorized_prefix``
      是调用方声明的存储根（storage.source.uri 归一化后的 s3:// 前缀），用于
      generic storage 数据集（不在手工 mirror registry 时，uri 本身就是授权边界）；
    - 各模块（generic storage、hybrid）禁止自己拼 scheme——executor 只见到一种
      canonical representation。
    """
    out: list[str] = []
    extra = [authorized_prefix] if authorized_prefix else None
    for raw in paths:
        normalized = cos_uri_to_s3_uri(str(raw)).rstrip("/")
        authorize_s3_path(normalized, extra_prefixes=extra)
        out.append(normalized)
    return out


def _local_daily_complete(spec: MirrorSpec, time_range: tuple[Any, Any]) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    from .mirror import _local_table_dir

    local_dir = _local_table_dir(spec)
    return all((local_dir / f"{day.isoformat()}.parquet").exists() for day in _iter_dates(start, end))


def _local_hive_date_complete(spec: MirrorSpec, time_range: tuple[Any, Any]) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    return all(
        (spec.local_root / f"date={day.isoformat()}" / spec.file_name).exists()
        for day in _iter_dates(start, end)
    )


def _local_hive_year_complete(spec: MirrorSpec, time_range: tuple[Any, Any]) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    return all(
        (spec.local_root / f"year={year}" / spec.file_name).exists()
        for year in _iter_years(start, end)
    )


def local_mirror_complete_for_range(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None,
) -> bool:
    """``auto`` 模式：判断本地是否已有所需文件。"""
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        return False
    if spec.layout == "single_full":
        from .mirror import _local_table_dir

        return (_local_table_dir(spec) / spec.file_name).exists()
    if spec.layout == "root_file":
        return (spec.local_root / spec.file_name).exists()
    if time_range is None:
        return False
    if spec.layout == "daily_parquet":
        return _local_daily_complete(spec, time_range)
    if spec.layout == "hive_date":
        return _local_hive_date_complete(spec, time_range)
    if spec.layout == "hive_year":
        return _local_hive_year_complete(spec, time_range)
    return False


def declares_cos_storage(ds: "Dataset") -> bool:
    """#29 数据集是否声明 COS 作为 storage backend（``storage.source.type == cos``）。

    让 remote 成为 Dataset storage 的通用能力，不再绑定 StaticDataset + 手工
    mirror registry——factor lake / model outputs / features 等 ParametricDataset
    也能走 COS remote。
    """
    storage = getattr(ds, "storage", None)
    if isinstance(storage, dict):
        src = storage.get("source")
        if isinstance(src, dict) and str(src.get("type") or "").lower() in {
            "cos",
            "s3",
            "oss",
        }:
            return True
        if isinstance(src, str) and src.lower().startswith(("cos://", "s3://")):
            return True
    return False


def should_read_cos_remote(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None,
) -> bool:
    """是否对本次读走 COS remote（不拉本地镜像）。"""
    has_mirror = mirror_spec_for_dataset(ds.name) is not None
    has_cos_storage = declares_cos_storage(ds)
    if not has_mirror and not has_cos_storage:
        return False

    mode = cos_read_mode()
    if mode == "mirror":
        return False
    if mode == "remote":
        return True
    # auto: 本地齐全用本地，否则 remote
    if time_range is None:
        return False
    return not local_mirror_complete_for_range(ds.name, time_range=time_range)


def paths_are_remote(paths: Sequence[str]) -> bool:
    return any(str(p).startswith("s3://") for p in paths)


def hybrid_cos_read_paths(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None,
    params: Mapping[str, Any] | None = None,
) -> list[str] | None:
    """#26 local+remote 混合计划（MultiLocationScan 的最小实现）。

    ``auto`` 模式下，本地已有 partition 用本地路径、缺失的 partition 用远程
    ``s3://`` 路径——不再「本地不齐就整个区间 remote」。需要 httpfs backend
    （DuckDB 一条查询可同时读本地与 s3）。

    返回 None 表示不混合（本地齐全 / 非 auto / 非 httpfs / 无可枚举日期），
    调用方走原 all-or-nothing 逻辑。
    """
    from data_access.registry import StaticDataset

    if not isinstance(ds, StaticDataset):
        return None
    spec = mirror_spec_for_dataset(ds.name)
    if spec is None:
        return None
    if cos_read_mode() != "auto":
        return None
    if cos_remote_backend() != "httpfs":
        return None
    if time_range is None or time_range[0] is None:
        return None
    if local_mirror_complete_for_range(ds.name, time_range=time_range):
        return None  # 本地齐全 → 纯本地，不混合

    from .mirror import _expected_dates, _local_table_dir

    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return None
    # #P0-4 hybrid remote fragment 必须用 canonical s3:// execution URI。
    # 不再 _cos_table_uri（返回 cos://）——executor 对 remote 只识别 s3://。
    remote_base = _s3_table_base(spec)
    local: list[str] = []
    remote: list[str] = []
    if spec.layout == "daily_parquet":
        local_dir = _local_table_dir(spec)
        for day in _expected_dates(ds.name, start, end):
            lp = local_dir / f"{day.isoformat()}.parquet"
            if lp.exists():
                local.append(str(lp))
            else:
                rp = f"{remote_base}/{day.isoformat()}.parquet"
                authorize_s3_path(rp)
                remote.append(rp)
    elif spec.layout == "hive_date":
        for day in _expected_dates(ds.name, start, end):
            lp = spec.local_root / f"date={day.isoformat()}" / spec.file_name
            if lp.exists():
                local.append(str(lp))
            else:
                rp = f"{remote_base}/date={day.isoformat()}/{spec.file_name}"
                authorize_s3_path(rp)
                remote.append(rp)
    elif spec.layout == "hive_year":
        for year in _iter_years(start, end):
            lp = spec.local_root / f"year={year}" / spec.file_name
            if lp.exists():
                local.append(str(lp))
            else:
                rp = f"{remote_base}/year={year}/{spec.file_name}"
                authorize_s3_path(rp)
                remote.append(rp)
    else:
        return None
    if not remote:
        return None  # 全部本地都有了（防御；上面 complete 检查应已拦截）
    merged = [*local, *remote]
    logger.info("cos_multi_location: dataset=%s local=%d remote=%d", ds.name, len(local), len(remote))
    return merged


def cos_remote_backend() -> str:
    """``auto`` | ``httpfs`` | ``cli``。

    - ``httpfs``：DuckDB 直读 ``s3://``（需 httpfs 扩展 + S3 凭证 + endpoint）
    - ``cli``：用 ``clean-cos-ro`` 按需拉到独立 cache（不改 COS，也不写永久 mirror 根）
    - ``auto``：httpfs 可用且凭证齐全则 httpfs，否则 cli
    """
    raw = os.environ.get("DATA_ACCESS_COS_REMOTE_BACKEND", "auto").strip().lower()
    if raw not in {"auto", "httpfs", "cli"}:
        raise ValidationError(
            f"无效的 DATA_ACCESS_COS_REMOTE_BACKEND={raw!r}，允许: auto|httpfs|cli"
        )
    if raw != "auto":
        return raw
    if _httpfs_and_creds_ready():
        return "httpfs"
    return "cli"


_httpfs_ext_available: bool | None = None
_httpfs_ext_lock = threading.Lock()


def _httpfs_extension_available() -> bool:
    """进程内缓存 httpfs 扩展探测结果，避免 auto 模式每次 spawn DuckDB。"""
    global _httpfs_ext_available
    with _httpfs_ext_lock:
        if _httpfs_ext_available is not None:
            return _httpfs_ext_available
        try:
            import duckdb

            con = duckdb.connect()
            try:
                con.execute("LOAD httpfs;")
            except Exception:
                try:
                    con.execute("INSTALL httpfs; LOAD httpfs;")
                except Exception:
                    _httpfs_ext_available = False
                    return False
            finally:
                con.close()
            _httpfs_ext_available = True
            return True
        except Exception:
            _httpfs_ext_available = False
            return False


def _httpfs_and_creds_ready() -> bool:
    if not _httpfs_extension_available():
        return False
    # httpfs 用 SigV4 签名；腾讯云 COS 对部分 STS/子账号凭证不接受 SigV4。
    # 仅在用户**显式**配置了 S3 endpoint 时才信任 httpfs；否则 auto 走 cli
    # （coscli 用 COS 原生签名，凭证从 ~/.cos.yaml 解析，稳定可用）。
    if not any(
        os.environ.get(name)
        for name in ("DATA_ACCESS_COS_S3_ENDPOINT", "COS_S3_ENDPOINT", "S3_ENDPOINT")
    ):
        return False
    try:
        resolve_s3_credentials()
        return True
    except ValidationError:
        return False


def reset_httpfs_probe_cache() -> None:
    """测试用：清空 httpfs 扩展探测缓存。"""
    global _httpfs_ext_available
    with _httpfs_ext_lock:
        _httpfs_ext_available = None


def load_cos_cli_credentials_into_env() -> bool:
    """从 ``~/.cos.yaml`` 注入 COS_SECRET_*（若尚未设置）。成功返回 True。"""
    if os.environ.get("COS_SECRET_ID") and os.environ.get("COS_SECRET_KEY"):
        return True
    yaml_path = Path(os.environ.get("DATA_ACCESS_COS_YAML", Path.home() / ".cos.yaml"))
    if not yaml_path.is_file():
        return False
    try:
        import yaml
    except ImportError:
        return False
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        base = (data.get("cos") or {}).get("base") or {}
        sid = str(base.get("secretid") or "").strip()
        skey = str(base.get("secretkey") or "").strip()
        if not sid or not skey:
            return False
        os.environ.setdefault("COS_SECRET_ID", sid)
        os.environ.setdefault("COS_SECRET_KEY", skey)
        buckets = (data.get("cos") or {}).get("buckets") or []
        for b in buckets:
            if not isinstance(b, dict):
                continue
            ep = str(b.get("endpoint") or "").strip()
            region = str(b.get("region") or "").strip()
            if ep:
                os.environ.setdefault("DATA_ACCESS_COS_S3_ENDPOINT", ep)
            if region:
                os.environ.setdefault("DATA_ACCESS_COS_S3_REGION", region)
            break
        return True
    except Exception as exc:
        logger.debug("load ~/.cos.yaml failed: %s", exc)
        return False


def cos_cache_root() -> Path:
    """CLI remote 按需缓存根（与永久 mirror 根分离）。"""
    raw = os.environ.get("DATA_ACCESS_COS_CACHE_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    workspace = os.environ.get("QUANTSOCIETY_WORKSPACE_DATA_ROOT")
    if workspace:
        return (Path(workspace) / ".cos_remote_cache").resolve()
    return Path("/tmp/data_access_cos_cache").resolve()


def _cache_mirror_spec(spec: MirrorSpec) -> MirrorSpec:
    """把 mirror 的本地根改到 cache，COS 前缀不变。"""
    cos = spec.cos_prefix.rstrip("/")
    without_scheme = cos.split("://", 1)[-1]
    parts = without_scheme.split("/", 1)
    rel = parts[1] if len(parts) > 1 else parts[0]
    return MirrorSpec(
        cos_prefix=spec.cos_prefix,
        local_root=cos_cache_root() / rel,
        table=spec.table,
        layout=spec.layout,
        file_name=spec.file_name,
    )


def materialize_remote_via_cli(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    ds: "Dataset | None" = None,
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """用 clean-cos-ro 按需把 COS 对象拉到 cache，返回本地可读路径。

    不写永久 ``ASHARE_PARQUET_ROOT`` / ``US_MASSIVE_ROOT``；不修改 COS 源对象。
    """
    from .mirror import (
        _local_table_dir,
        _sync_daily_file,
        _sync_hive_date,
        _sync_hive_year,
        _sync_root_file,
        _sync_single_full,
    )

    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        # #29 storage 声明型数据集：CLI 不支持逐对象下载远程裸读 → 提示换 httpfs
        if ds is not None and declares_cos_storage(ds):
            raise ValidationError(
                f"数据集 '{dataset_name}' 走 storage.source=cos remote 需要 httpfs "
                f"backend（DATA_ACCESS_COS_REMOTE_BACKEND=httpfs），CLI 仅支持已登记 "
                f"mirror spec 的数据集"
            )
        raise ValidationError(f"数据集 '{dataset_name}' 未配置 COS mirror")
        raise ValidationError(f"数据集 '{dataset_name}' 未配置 COS mirror，无法 remote/cli 读取")
    cache_spec = _cache_mirror_spec(spec)

    if cache_spec.layout == "single_full":
        _sync_single_full(cache_spec)
        path = _local_table_dir(cache_spec) / cache_spec.file_name
        if not path.exists():
            raise ValidationError(f"COS cli 拉取失败: {dataset_name} -> {path}")
        return [str(path)]

    if cache_spec.layout == "root_file":
        _sync_root_file(cache_spec)
        path = cache_spec.local_root / cache_spec.file_name
        if not path.exists():
            raise ValidationError(f"COS cli 拉取失败: {dataset_name} -> {path}")
        return [str(path)]

    if time_range is None:
        raise ValidationError(
            f"COS remote/cli 读 '{dataset_name}' 必须指定 time_range，"
            "避免整表 sync；示例 time_range=('2024-01-02','2024-01-03')"
        )

    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValidationError(f"无法解析 time_range: {time_range}")

    paths: list[str] = []
    if cache_spec.layout == "daily_parquet":
        for day in _iter_dates(start, end):
            _sync_daily_file(cache_spec, day)
            dest = _local_table_dir(cache_spec) / f"{day.isoformat()}.parquet"
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(str(dest))
    elif cache_spec.layout == "hive_date":
        for day in _iter_dates(start, end):
            _sync_hive_date(cache_spec, day)
            dest = cache_spec.local_root / f"date={day.isoformat()}" / cache_spec.file_name
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(str(dest))
    elif cache_spec.layout == "hive_year":
        for year in _iter_years(start, end):
            _sync_hive_year(cache_spec, year)
            dest = cache_spec.local_root / f"year={year}" / cache_spec.file_name
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(str(dest))
    else:
        raise ValidationError(f"未知 layout: {cache_spec.layout}")

    if not paths:
        raise ValidationError(
            f"COS cli 在 time_range={time_range} 未拉到任何文件: {dataset_name}"
        )
    logger.info(
        "cos_remote_cli: dataset=%s files=%d cache=%s",
        dataset_name,
        len(paths),
        cos_cache_root(),
    )
    return paths


def prepare_cos_remote_paths(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    ds: "Dataset | None" = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[list[str], str]:
    """返回 (paths, backend)，backend 为 ``httpfs`` 或 ``cli``。"""
    load_cos_cli_credentials_into_env()
    backend = cos_remote_backend()
    if backend == "httpfs":
        return (
            build_remote_paths(dataset_name, time_range=time_range, ds=ds, params=params),
            "httpfs",
        )
    return (
        materialize_remote_via_cli(
            dataset_name, time_range=time_range, ds=ds, params=params
        ),
        "cli",
    )
