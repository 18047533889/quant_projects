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


def authorize_s3_path(path: str) -> None:
    """远程路径必须落在已登记 COS 前缀下。"""
    normalized = path.rstrip("/")
    for prefix in allowed_s3_prefixes():
        base = prefix.rstrip("/")
        if normalized == base or normalized.startswith(base + "/"):
            return
    hint = "\n  ".join(allowed_s3_prefixes()[:5])
    raise ValidationError(
        f"S3 路径越界（不在已登记 COS mirror 前缀下）：{path}\n"
        f"示例允许前缀：\n  {hint}\n"
        f"如需新增，请在 cos_mirror.DATASET_MIRROR_REGISTRY 登记。"
    )


def resolve_s3_credentials() -> S3Credentials:
    """从环境变量解析腾讯云 COS / 通用 S3 凭证。"""
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
            "或 AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY。"
        )
    if not endpoint:
        raise ValidationError(
            "COS 远程直读需要 DATA_ACCESS_COS_S3_ENDPOINT，"
            "例如 cos.ap-guangzhou.myqcloud.com"
        )
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
) -> list[str]:
    """按 mirror 布局生成 DuckDB 可读的 ``s3://`` 路径列表。"""
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        raise ValidationError(f"数据集 '{dataset_name}' 未配置 COS mirror，无法 remote 读取")

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


def should_read_cos_remote(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None,
) -> bool:
    """是否对本次读走 COS remote（不拉本地镜像）。"""
    from data_access.registry import StaticDataset

    if not isinstance(ds, StaticDataset):
        return False
    if mirror_spec_for_dataset(ds.name) is None:
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
) -> tuple[list[str], str]:
    """返回 (paths, backend)，backend 为 ``httpfs`` 或 ``cli``。"""
    load_cos_cli_credentials_into_env()
    backend = cos_remote_backend()
    if backend == "httpfs":
        return build_remote_paths(dataset_name, time_range=time_range), "httpfs"
    return materialize_remote_via_cli(dataset_name, time_range=time_range), "cli"
