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

import hashlib
import logging
import math
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .mirror import (
    DATASET_MIRROR_REGISTRY,
    MirrorSpec,
    _cos_table_uri,
    _iter_years,
    _local_file_usable,
    _parse_date,
    expected_partitions,
    mirror_spec_for_dataset,
)
from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    from data_access.registry import Dataset, StaticDataset
    from data_access.security.credentials import CredentialMaterial

logger = logging.getLogger("data_access.cos_remote")

_VALID_MODES = frozenset({"mirror", "remote", "auto"})


@dataclass(frozen=True)
class S3Credentials:
    """COS/S3 凭证（R24 P0-S1 §3.4 / §3.5）。

    ``secret_access_key`` / ``session_token`` 带 ``repr=False``——日志、异常、
    repr 绝不能打印（STS/CAM 临时身份没有 session token 支持是不完整的）。

    - ``session_token``：STS 临时凭证 token（可空）
    - ``expires_at``：临时凭证过期时间（可空，provider 已知才填）
    - ``principal_id``：这份凭证对应的 server principal（可空）
    - ``credential_scope_id``：凭证权限范围标识（可空，cache scope 绑定用）
    """

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    endpoint: str
    region: str
    use_ssl: bool = True
    url_style: str = "path"
    session_token: str | None = field(default=None, repr=False)
    expires_at: datetime | None = None
    principal_id: str | None = None
    credential_scope_id: str | None = None
    credential_generation_id: str | None = None

    def __repr__(self) -> str:
        from data_access.security.redaction import redact_secret

        return (
            "S3Credentials("
            f"access_key_id={redact_secret(self.access_key_id)}, "
            "secret_access_key=<redacted>, "
            f"endpoint={self.endpoint!r}, region={self.region!r}, "
            f"use_ssl={self.use_ssl}, url_style={self.url_style!r}, "
            f"session_token={'<redacted>' if self.session_token else None!r}, "
            f"expires_at={self.expires_at!r}, "
            f"principal_id={self.principal_id!r}, "
            f"credential_scope_id={self.credential_scope_id!r}, "
            f"credential_generation_id={self.credential_generation_id!r})"
        )

    @classmethod
    def from_material(
        cls,
        m: "CredentialMaterial",
        *,
        endpoint: str,
        region: str,
        use_ssl: bool = True,
        url_style: str = "path",
    ) -> "S3Credentials":
        return cls(
            access_key_id=m.access_key_id,
            secret_access_key=m.secret_access_key,
            endpoint=endpoint,
            region=region,
            use_ssl=use_ssl,
            url_style=url_style,
            session_token=m.session_token,
            expires_at=m.expires_at,
            principal_id=m.principal_id,
            credential_scope_id=m.credential_scope_id,
            credential_generation_id=m.credential_generation_id,
        )

    @property
    def has_session_token(self) -> bool:
        return bool(self.session_token)

    def to_safe_dict(self) -> dict[str, Any]:
        from data_access.security.redaction import redact_secret

        return {
            "access_key_id": redact_secret(self.access_key_id),
            "endpoint": self.endpoint,
            "region": self.region,
            "has_session_token": self.has_session_token,
            "principal_id": self.principal_id,
            "credential_scope_id": self.credential_scope_id,
            "credential_generation_id": self.credential_generation_id,
        }


def _production_context() -> bool:
    """P0-1：production 判定（remote-first 默认）。

    与 ``local_disk_policy.spill_policy_for()`` 一致：``FACTOR_ENGINE_PRODUCTION``
    （1/true/yes）或 ``FACTOR_ENGINE_LOCAL_DISK_POLICY=STRICT_REMOTE`` 视为
    production。也兼容 ``QUANT_PRODUCTION_MODE`` / ``DATA_ACCESS_STRICT_READ``
    （RuntimeModeIdentity 权威）。
    """
    try:
        from data_access.read.local_disk_policy import (
            LocalDiskPolicy,
            spill_policy_for,
        )

        if spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE:
            return True
    except Exception:
        pass
    try:
        from data_access.runtime.mode_identity import is_production_authority

        if is_production_authority():
            return True
    except Exception:
        pass
    return False


def cos_read_mode() -> str:
    """``mirror`` | ``remote`` | ``auto``。

    **P0-1**：production 上下文默认 ``remote``（remote-first，无 local-mirror
    fallback）——即使 ``DATA_ACCESS_COS_READ_MODE`` 未设置或显式设为 ``mirror``，
    production 也强制 ``remote``。``mirror``/``auto`` 仅 research/dev 允许。
    """
    raw = os.environ.get("DATA_ACCESS_COS_READ_MODE", "").strip().lower()
    if raw not in _VALID_MODES:
        # 未设置或非法值：production 一律 remote-first；research 缺省 mirror。
        if _production_context():
            return "remote"
        if not raw:
            return "mirror"
        raise ValidationError(
            f"无效的 DATA_ACCESS_COS_READ_MODE={raw!r}，允许: {sorted(_VALID_MODES)}"
        )
    if _production_context():
        # production 强制 remote-first：mirror/auto 仅 research/dev 允许。
        return "remote"
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


# ---------------------------------------------------------------------------
# R57：COS CLI 网关 LIST/HEAD（跨账号 bucket 的元数据通路）
# ---------------------------------------------------------------------------
# server C 上 qs-cold 等数据 bucket 归属其他账号（跨 domain）——本机静态密钥
# （~/.cos.yaml / CredentialProvider）对它们 IAM 不可见（boto3/httpfs 一律
# NoSuchBucket/404）。唯一稳定通路是 coscli 网关（clean-cos-ro，COS 原生签名 +
# root-only 凭证）。这两条 best-effort 钩子解析 ``<cli> ls`` 的人类可读表格，
# 给 snapshot resolver 回填 content_length/etag（governor 才能做真实 scan-bytes
# 准入，不再按未知成本保守拒绝）。任何失败返回 []/None → 落回 FileVersion
# fallback（fail-closed 语义不变）。

#: ``ls`` 表格 SIZE 列的乘数（coscli 人类可读单位）。
_CLI_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
    "PB": 1024**5,
}


def _cli_size_to_bytes(raw: str) -> int | None:
    """``"205.82 KB"`` → 210755（向上取整；未知单位返回 None）。"""
    parts = raw.split()
    if not parts:
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].upper() if len(parts) > 1 else "B"
    scale = _CLI_SIZE_UNITS.get(unit)
    if scale is None:
        return None
    return int(math.ceil(value * scale))


def _parse_cos_cli_ls_output(output: str) -> list[dict[str, Any]]:
    """解析 coscli ``ls`` 人类可读表格 → [{key, size, etag, last_modified}]。

    表格形态（coscli 固定输出，管道可解析）::

        KEY | TYPE | LAST MODIFIED | ETAG | SIZE | RESTORESTATUS
        ----+-----+...（分隔线）
        clean_data/.../2024-01-02.parquet | STANDARD | 2026-06-15T16:25:37+08:00 | "abc..." | 205.82 KB |

    只接受「第 1 列像对象 key、ETAG 列是引号包裹 md5」的数据行；分隔线 /
    TOTAL OBJECTS / 空 RESTORESTATUS 一律跳过。解析失败返回 []。
    """
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 5:
            continue
        key, _typ, last_modified, etag, size_raw = cells[:5]
        if not key or key == "KEY" or set(key) <= {"-"}:
            continue
        size = _cli_size_to_bytes(size_raw)
        if size is None:
            continue
        etag_clean = etag.strip().strip('"')
        if not etag_clean or len(etag_clean) < 8 or " " in etag_clean:
            continue
        if "OBJECTS" in line.upper() and not key.endswith(".parquet"):
            continue
        rows.append(
            {
                "key": key,
                "size": size,
                "etag": etag_clean or None,
                "last_modified": last_modified or None,
            }
        )
    return rows


def cos_cli_ls(
    uri: str,
    *,
    cli: str | None = None,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """经 COS CLI 网关列出对象元数据（跨账号 bucket 的 LIST 通路）。

    ``uri`` 是 ``cos://bucket/prefix``（``s3://`` 自动归一）。返回
    ``_parse_cos_cli_ls_output`` 结构；CLI 失败/超时/输出异常 → []（best-effort，
    绝不阻塞读路径——governor 会按未知成本拒绝，语义同 boto3 HEAD 不可用）。
    """
    from data_access.cos.mirror import COS_CLI as DEFAULT_CLI

    cos_uri = str(uri)
    if cos_uri.startswith("s3://"):
        cos_uri = "cos://" + cos_uri[len("s3://"):]
    if not cos_uri.startswith("cos://"):
        return []
    try:
        proc = subprocess.run(
            [cli or DEFAULT_CLI, "ls", cos_uri],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return _parse_cos_cli_ls_output(proc.stdout)


def cos_cli_head(
    uri: str,
    *,
    cli: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any] | None:
    """经 COS CLI 网关取单对象元数据（跨账号 bucket 的 HEAD 通路）。

    复用 ``cos_cli_ls``：CLI 对「精确对象 URI」的 ``ls`` 返回单行表格（实测）。
    未命中/失败 → None。
    """
    cos_uri = str(uri)
    if cos_uri.startswith("s3://"):
        cos_uri = "cos://" + cos_uri[len("s3://"):]
    if not cos_uri.startswith("cos://") or cos_uri.rstrip("/").endswith("/"):
        return None
    rows = cos_cli_ls(cos_uri, cli=cli, timeout_s=timeout_s)
    if len(rows) != 1:
        return None
    return rows[0]


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
    """research-only：从 coscli/clean-cos-ro 配置（``~/.cos.yaml``）读取凭证。

    R24 P0-S1 §3：production/strict 下**禁止**解析 ``~/.cos.yaml`` 获取 base
    credential（会绕过 ``clean-cos-ro`` 的服务器权限分级）。research 也要求显式
    ``DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE=1`` opt-in。返回 ``(secretid, secretkey)``
    或 ``("", "")``。
    """
    from data_access.security.credentials import (
        CosCliConfigProvider,
        allow_coscli_config_parse,
    )

    if not allow_coscli_config_parse():
        return "", ""
    try:
        material = CosCliConfigProvider().resolve()
        return material.access_key_id, material.secret_access_key
    except ValidationError:
        return "", ""


def resolve_s3_credentials() -> S3Credentials:
    """解析腾讯云 COS / 通用 S3 凭证（R24 P0-S1 §3.1）。

    凭证只来自明确 CredentialProvider 链：
        1. 显式注入的 CredentialProvider（``set_credential_provider``）；
        2. 部署注入的 server-scoped env credential（EnvCredentialProvider，§3.1 C）；
        3. research 显式 opt-in 的 coscli 配置（CosCliConfigProvider，§3.4）。

    **P0-6 fail-closed**：当当前执行上下文有已认证 principal 且其类型为
    HUMAN/TEAM（非 SERVICE）时，若没有 principal-scoped credential 解析成功，
    则**必须 fail-closed（抛 ValidationError）**——绝不回退到 server 的
    global/env/service credential（那会绕过 per-team 最小权限）。只有
    principal_type=SERVICE（后台任务）才允许走 service/env/global 链。

    **production/strict 下绝不自动解析 ``~/.cos.yaml -> cos.base.secret*``**
    （A/B/C/D 之外的来源全部拒绝）。任何 provider 都不写 ``os.environ``。
    endpoint 缺省按 region 推断为 ``cos.<region>.myqcloud.com``。
    """
    from data_access.security.credentials import (
        EnvCredentialProvider,
        _global_credential_provider,
    )
    from data_access.security.execution_context import (
        current_credential_provider,
        current_principal_type,
    )

    material = None
    # R28-17：优先 **request-scoped** credential provider（HTTP/任务执行上下文的
    # 每 principal 凭证），再回退进程级全局 provider。之前只认全局——并发 API key
    # A/B 各自带自己的受限凭证时，嵌套读（read_joined / cache miss / stream）会
    # 拿全局 base credential，绕过 per-principal 权限分级。
    provider = current_credential_provider() or _global_credential_provider()

    # P0-6：已认证 HUMAN/TEAM principal 只允许用 **principal-scoped** credential。
    # 若没有 request-scoped provider → fail-closed，绝不回退 server 的
    # global/env/service credential（那会绕过 per-team 最小权限）。
    ptype = current_principal_type()
    if ptype is not None and ptype.upper() != "SERVICE":
        if current_credential_provider() is None:
            raise ValidationError(
                f"principal_type={ptype!r} 的已认证请求没有 principal-scoped COS/S3 "
                "凭证，且不允许回退到 server 的 global/env/service credential "
                "（P0-6 fail-closed，per-team 最小权限）。请为该 principal 显式注入 "
                "CredentialProvider 或 scoped env credential。"
            )
        # 有 request-scoped provider：用它，绝不回退 global/env。
        material = provider.resolve()
    elif provider is not None:
        material = provider.resolve()
    else:
        try:
            material = EnvCredentialProvider().resolve()
        except ValidationError:
            material = None

    if material is None:
        # research 显式 opt-in：coscli/clean-cos-ro 配置（不影响 production）。
        access, secret = _read_cos_cli_credentials()
        if access and secret:
            from data_access.security.credentials import CredentialMaterial

            material = CredentialMaterial(
                access_key_id=access,
                secret_access_key=secret,
                source="coscli-research",
            )
    if material is None:
        raise ValidationError(
            "COS 远程直读需要凭证：请通过部署注入 COS_SECRET_ID + COS_SECRET_KEY"
            "（或 AWS_/S3_ 前缀），或显式配置 CredentialProvider。"
            "production 禁止解析 ~/.cos.yaml（见 DATA_ACCESS_ALLOW_COSCLI_CONFIG_PARSE）。"
        )
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
    if not endpoint:
        # Tencent COS 默认 endpoint 由 region 推断。
        endpoint = f"cos.{region}.myqcloud.com"
    use_ssl = os.environ.get("DATA_ACCESS_COS_S3_USE_SSL", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    url_style = os.environ.get("DATA_ACCESS_COS_S3_URL_STYLE", "path").strip() or "path"
    return S3Credentials.from_material(
        material,
        endpoint=endpoint,
        region=region,
        use_ssl=use_ssl,
        url_style=url_style,
    )


def _s3_table_base(spec: MirrorSpec) -> str:
    return cos_uri_to_s3_uri(_cos_table_uri(spec)).rstrip("/")


def _remote_daily_paths(
    spec: MirrorSpec,
    dataset_name: str,
    time_range: tuple[Any, Any] | None,
) -> list[str]:
    base = _s3_table_base(spec)
    if time_range is None:
        authorize_s3_path(f"{base}/*.parquet")
        return [f"{base}/*.parquet"]
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        raise ValidationError(f"remote 模式无法解析 time_range: {time_range}")
    # #P0 收官（0.9.5）：期望 partition 走共享 expected_partitions（trade_day 用
    # 交易日历，跨周末/节假日不再生成不存在的对象路径）。
    # R25 P0-002：文件名走共享 locator（_daily_filename，StockCapital shares_ 前缀），
    # 不再 mirror/remote 各自拼。
    from .mirror import _daily_filename

    paths = [
        f"{base}/{_daily_filename(spec, day)}"
        for day in expected_partitions(dataset_name, start, end, layout=spec.layout)
    ]
    for path in paths:
        authorize_s3_path(path)
    return paths


def _remote_period_files(
    spec: MirrorSpec,
    dataset_name: str,
    time_range: tuple[Any, Any] | None,
) -> list[str]:
    """R25 P0-001：period_files 布局（US finance）的远程路径。

    partition_clock=period_end（StockIncome/2024-03-31.parquet），predicate 时钟是
    filing_date——**不能**按 request time_range 展开物理文件名。用目录 glob
    ``{base}/*.parquet``（宁可多扫不能漏），snapshot resolver 再枚举 exact objects
    （P0-009）并严格按 filing_date 过滤。time_range 只用于提示（诊断日志）。
    """
    base = _s3_table_base(spec)
    authorize_s3_path(f"{base}/")
    return [f"{base}/*.parquet"]


def _remote_prefixed_date(
    spec: MirrorSpec,
    dataset_name: str,
    time_range: tuple[Any, Any] | None,
) -> list[str]:
    """R25 P0-002 / R26-P0-010：prefixed_date_file 布局（StockCapital split/shares）远程路径。

    与 daily 相同走共享 ``_daily_filename``（file_selector 区分 shares_ 前缀），
    但 expected_partitions 对该布局返回空（不按 request 时间轴枚举）。

    **R26-P0-010**：绝不用「空 prefix + *」表达排除逻辑——split 的
    ``*.parquet`` 会同时命中 ``shares_*.parquet``（两类 schema 混读）。改用共享
    ``FileSelector`` IR 生成**精确 date 文件名 glob**（字符类限定，不吞 shares_）。
    snapshot resolver 再 LIST exact objects 按 ``FileSelector.regex`` 过滤。
    """
    from data_access.contract.file_selector import file_selector_for_layout
    from data_access.contract.physical_partition import PhysicalLayout

    base = _s3_table_base(spec)
    authorize_s3_path(f"{base}/")
    selector = file_selector_for_layout(
        PhysicalLayout.PREFIXED_DATE_FILE,
        file_selector=getattr(spec, "file_selector", None),
    )
    return [selector.glob_pattern(base)]


def _remote_single_full(spec: MirrorSpec) -> list[str]:
    path = f"{_s3_table_base(spec)}/{spec.file_name}"
    authorize_s3_path(path)
    return [path]


def _remote_root_file(spec: MirrorSpec) -> list[str]:
    path = cos_uri_to_s3_uri(f"{spec.cos_prefix.rstrip('/')}/{spec.file_name}")
    authorize_s3_path(path)
    return [path]


def _remote_hive_date_paths(
    spec: MirrorSpec,
    dataset_name: str,
    time_range: tuple[Any, Any] | None,
) -> list[str]:
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
        f"{base}/date={day.isoformat()}/{spec.file_name}"
        for day in expected_partitions(dataset_name, start, end, layout="hive_date")
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
        return _remote_daily_paths(spec, dataset_name, time_range)
    if spec.layout in {"period_files", "event_files"}:
        return _remote_period_files(spec, dataset_name, time_range)
    if spec.layout == "prefixed_date_file":
        return _remote_prefixed_date(spec, dataset_name, time_range)
    if spec.layout == "hive_date":
        return _remote_hive_date_paths(spec, dataset_name, time_range)
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

    # #P0-final closure 4：registry 保存 typed StorageSpec（旧 raw dict 兼容），
    # 统一走 core.storage 的 declared_* 读取，不再各自解析 dict。
    from data_access.core.storage import declared_storage_layout, declared_storage_uri

    uri: str = str(declared_storage_uri(ds) or "")
    layout: str = str(declared_storage_layout(ds) or getattr(ds, "layout", "plain"))
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
        if not ds.glob_template:
            raise ValidationError(
                f"数据集 '{ds.name}' 是 ParametricDataset 但未声明 glob_template；"
                "无法推导 glob（fail-closed，禁止扫描整个数据集）"
            )
        # #P0 收官（0.9.5）：remote 必须复用与 local 完全相同的 compiled param IR。
        # 旧代码 ``glob_template.format(**dict(params or {}))`` 绕过了 ``validate_params``
        # （type/range/allowed/regex/路径遍历防护）——同一 dataset local 拒参数、
        # remote 却放行，严重时参数能把 glob 扩成 ``**/*.parquet`` 全库扫描。
        # ``validate_params`` 是 resolve_paths 前半段的唯一事实源，这里复用同一份。
        from data_access.registry.params_validation import ParamSpec, validate_params

        specs = ds.param_specs or {
            k: ParamSpec(name=k, type=t) for k, t in ds.params_schema.items()
        }
        validated = validate_params(ds.name, specs, dict(params or {}))
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
                (
                    f"{uri}/{day.isoformat()}.parquet"
                    for day in expected_partitions(ds.name, lo, hi, layout="daily_parquet")
                ),
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
                        for day in expected_partitions(ds.name, lo, hi, layout="hive_date")
                    ),
                    authorized_prefix=auth_prefix,
                )
            return _resolve_storage_paths(
                (f"{uri}/year={y}/data.parquet" for y in _iter_years(lo, hi)),
                authorized_prefix=auth_prefix,
            )
    return _resolve_storage_paths([f"{uri}/{glob}"], authorized_prefix=auth_prefix)


def resolve_remote_paths(
    ds: "Dataset",
    *,
    time_range: tuple[Any, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """public 入口：storage.source=cos 数据集的远程路径（无手工 mirror spec）。

    #P0 收官（0.9.5）：与 ``_remote_paths_from_storage`` 相同——ParametricDataset
    的 glob 复用 ``validate_params``（type/range/allowed/regex/路径遍历防护），
    local/remote 参数 parity 一致，不允许 remote 绕过校验。
    """
    return _remote_paths_from_storage(ds, time_range=time_range, params=params)


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


def _local_daily_complete(
    spec: MirrorSpec, dataset_name: str, time_range: tuple[Any, Any]
) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    from .mirror import _local_table_dir

    local_dir = _local_table_dir(spec)
    return all(
        _local_file_usable(local_dir / f"{day.isoformat()}.parquet")
        for day in expected_partitions(dataset_name, start, end, layout="daily_parquet")
    )


def _local_hive_date_complete(
    spec: MirrorSpec, dataset_name: str, time_range: tuple[Any, Any]
) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    return all(
        _local_file_usable(spec.local_root / f"date={day.isoformat()}" / spec.file_name)
        for day in expected_partitions(dataset_name, start, end, layout="hive_date")
    )


def _local_hive_year_complete(
    spec: MirrorSpec, dataset_name: str, time_range: tuple[Any, Any]
) -> bool:
    start = _parse_date(time_range[0])
    end = _parse_date(time_range[1])
    if start is None or end is None:
        return False
    return all(
        _local_file_usable(spec.local_root / f"year={year}" / spec.file_name)
        for year in expected_partitions(dataset_name, start, end, layout="hive_year")
    )


def local_mirror_complete_for_range(
    dataset_name: str,
    *,
    time_range: tuple[Any, Any] | None,
) -> bool:
    """``auto`` 模式：判断本地是否已有所需文件。

    #P0 收官（0.9.5）：统一走 ``_local_file_usable``（verified/corrupt 三态）
    而不是裸 ``.exists()``——本地一个截断/checksum 错/无 manifest 的 parquet
    **只要文件还在**，exists 就会把它当「本地齐全」而选它，而不是 resync/remote。
    """
    spec = mirror_spec_for_dataset(dataset_name)
    if spec is None:
        return False
    if spec.layout == "single_full":
        from .mirror import _local_table_dir

        return _local_file_usable(_local_table_dir(spec) / spec.file_name)
    if spec.layout == "root_file":
        return _local_file_usable(spec.local_root / spec.file_name)
    if spec.layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/016：period/event/prefixed 布局按完整目录判断（不能按
        # time_range 枚举）。目录有任意 verified parquet 视为本地已同步（完整的
        # 周期文件集，后续由 snapshot/read 层严格过滤）。
        from .mirror import _local_file_fresh

        local_dir = _local_table_dir(spec) if spec.table else spec.local_root
        if not local_dir.exists() or not local_dir.is_dir():
            return False
        for p in local_dir.glob("*.parquet"):
            if _local_file_fresh(p):
                return True
        return False
    if time_range is None:
        return False
    if spec.layout == "daily_parquet":
        return _local_daily_complete(spec, dataset_name, time_range)
    if spec.layout == "hive_date":
        return _local_hive_date_complete(spec, dataset_name, time_range)
    if spec.layout == "hive_year":
        return _local_hive_year_complete(spec, dataset_name, time_range)
    return False


def declares_cos_storage(ds: "Dataset") -> bool:
    """#29 数据集是否声明 COS 作为 storage backend（``storage.source.type == cos``）。

    让 remote 成为 Dataset storage 的通用能力，不再绑定 StaticDataset + 手工
    mirror registry——factor lake / model outputs / features 等 ParametricDataset
    也能走 COS remote。
    """
    # #P0-final closure 4：typed StorageSpec / 旧 raw dict 都支持。
    from data_access.core.storage import StorageSpec

    storage = getattr(ds, "storage", None)
    if isinstance(storage, StorageSpec):
        return storage.type in {"cos", "s3", "oss"} or str(storage.uri or "").startswith(
            ("cos://", "s3://")
        )
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
    if spec.layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/016：period/event/prefixed 布局禁止 local/remote 混源——
        # 本地周期文件与远程 glob 无法按 partition_clock 对齐（不混 epoch）。
        return None
    # R25 P0-013：auto hybrid 必须先能证明 local/remote 同一 source generation。
    # 无 publisher manifest 证明同 generation 时，production/strict 禁止混源
    # （fallback 全部 remote target generation，不混）。research 可降级（返回 None
    # 表示不混合，走 all-or-nothing remote）。
    if _strict_mode():
        try:
            from data_access.read.query_budget import is_strict_semantics

            if is_strict_semantics():
                # strict：不允许 local/remote 混 source epoch（无法证明同 generation）。
                # 返回 None → 调用方走纯 remote（all target generation），不混。
                return None
        except Exception:
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
            if _local_file_usable(lp):
                local.append(str(lp))
            else:
                rp = f"{remote_base}/{day.isoformat()}.parquet"
                authorize_s3_path(rp)
                remote.append(rp)
    elif spec.layout == "hive_date":
        for day in _expected_dates(ds.name, start, end):
            lp = spec.local_root / f"date={day.isoformat()}" / spec.file_name
            if _local_file_usable(lp):
                local.append(str(lp))
            else:
                rp = f"{remote_base}/date={day.isoformat()}/{spec.file_name}"
                authorize_s3_path(rp)
                remote.append(rp)
    elif spec.layout == "hive_year":
        for year in _iter_years(start, end):
            lp = spec.local_root / f"year={year}" / spec.file_name
            if _local_file_usable(lp):
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


def _strict_mode() -> bool:
    """#P1-final closure 17：唯一严格模式判定（production OR strict_read）。"""
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        return True


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
                # #P1-final closure 17：strict/production 禁止 query-time 联网
                # INSTALL（扩展是部署阶段的事，缺扩展是 deployment 错误）——
                # LOAD 失败即视为不可用，绝不自动联网装。
                if _strict_mode():
                    _httpfs_ext_available = False
                    return False
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
    """**已废弃**（R24 P0-S1 §3.3）：绝不把 secret 写入全局环境。

    - production/strict：直接抛 ``ValidationError``——进程存在解析 coscli 配置
      注入 env 的架构能力本身就是安全债（子进程继承 / crash dump / 长驻 worker
      其他模块可读 / 测试进程污染）。
    - research：不再写入 ``os.environ``。返回 False 并告警，调用方应改用
      ``resolve_s3_credentials``（research 显式 ``DATA_ACCESS_ALLOW_COSCLI_
      CONFIG_PARSE=1`` 时由 provider 链兜底）。
    """
    from data_access.read.query_budget import is_strict_semantics

    if is_strict_semantics():
        raise ValidationError(
            "load_cos_cli_credentials_into_env() 已废弃：禁止把 COS secret 写入"
            "全局 os.environ（production fail-closed）。请改用 resolve_s3_credentials()"
            "（凭证只来自 CredentialProvider / 部署注入 env）。"
        )
    logger.warning(
        "load_cos_cli_credentials_into_env() 已废弃且不再写入 os.environ；"
        "请改用 resolve_s3_credentials()"
    )
    return False


def _cache_principal_scope() -> str:
    """当前 principal 的 cache scope id（R24 P0-S3 §5.1）。

    高权限服务器下载的数据不能因为落盘就失去权限保护——cache 必须按
    user/principal 隔离，不同 principal 不能共用同一份 cache。

    来源：``DATA_ACCESS_PRINCIPAL_ID``（部署注入）→ 否则 uid。只取脱敏的
    principal 标识，不参与权限判定本身（那属于 CAM/IAM + AccessPolicy）。
    """
    pid = os.environ.get("DATA_ACCESS_PRINCIPAL_ID", "").strip()
    if pid:
        # Never put an untrusted principal name in a filesystem path.  The old
        # character replacement made distinct identities (for example
        # ``tenant/a`` and ``tenant_a``) share a cache directory, and allowed
        # arbitrarily long identity values to create unbounded path components.
        # A domain-separated SHA-256 digest gives a stable, bounded scope while
        # keeping the principal itself out of cache paths and logs.
        digest = hashlib.sha256(f"dataaccess-cos-principal\\0{pid}".encode()).hexdigest()
        return f"p{digest[:32]}"
    try:
        return f"u{os.getuid()}"
    except (AttributeError, OSError):
        return "uunknown"


def cos_cache_root() -> Path:
    """CLI remote 按需缓存根（与永久 mirror 根分离，R24 P0-S3 §5.1 按 principal 隔离）。

    优先 ``$XDG_CACHE_HOME/quant-dataaccess/cos/<principal_scope>/``；
    否则 ``/tmp/data_access_cos_cache_<uid>/<principal_scope>/``。
    显式 ``DATA_ACCESS_COS_CACHE_ROOT`` 时在其下再挂 ``<principal_scope>/`` 子目录
    （仍按 principal 隔离，不允许跨 principal 共享）。
    """
    raw = os.environ.get("DATA_ACCESS_COS_CACHE_ROOT")
    if raw:
        return (Path(raw).expanduser().resolve() / _cache_principal_scope()).resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and xdg.strip():
        return (Path(xdg).expanduser() / "quant-dataaccess" / "cos" / _cache_principal_scope()).resolve()
    workspace = os.environ.get("QUANTSOCIETY_WORKSPACE_DATA_ROOT")
    if workspace:
        return (Path(workspace) / ".cos_remote_cache" / _cache_principal_scope()).resolve()
    return (Path(f"/tmp/data_access_cos_cache_{_cache_principal_scope()}")).resolve()


def ensure_cache_root_secure(root: Path | None = None) -> Path:
    """R24 P0-S3 §5.2 / §5.4 / T-S07 / T-S08：缓存根的权限安全保证。

    - 默认 directory=0700；显式 ``DATA_ACCESS_COS_CACHE_MODE``（如 ``0750``）可用
      group 模式（须显式配置，不默认 world-readable）；
    - production/strict 下 root 权限过宽（world-readable / group-world writable）、
      root 是 symlink → fail-closed；
    - root 必须落在当前 principal scope 下（防跨 principal 复用高权限缓存）。
    """
    import stat as _stat

    from data_access.read.query_budget import is_strict_semantics

    root = root or cos_cache_root()
    # 若已存在，先做安全校验（T-S07）。
    if root.exists():
        if root.is_symlink() or root.is_symlink():
            if is_strict_semantics():
                raise ValidationError(
                    f"COS cache root 是 symlink，拒绝使用（T-S08 fail-closed）: {root}"
                )
        lstat = root.lstat()
        mode = _stat.S_IMODE(lstat.st_mode)
        if mode & _stat.S_IWOTH:
            if is_strict_semantics():
                raise ValidationError(
                    f"COS cache root 对 world 可写（{oct(mode)}），production fail-closed: {root}"
                )
        if mode & _stat.S_IROTH:
            # world-readable 只在 strict 下拒绝；非 strict 也收敛到 0700。
            if is_strict_semantics():
                raise ValidationError(
                    f"COS cache root 对 world 可读（{oct(mode)}），production fail-closed: {root}"
                )
    root.mkdir(parents=True, exist_ok=True)
    # 默认目录 0700（或显式 DATA_ACCESS_COS_CACHE_MODE=0750/0710 group 模式）。
    _ALLOWED_CACHE_MODES = {0o700, 0o750, 0o710}
    try:
        mode = int(os.environ.get("DATA_ACCESS_COS_CACHE_MODE", "700"), 8)
    except (ValueError, TypeError):
        mode = 0o700
    if mode not in _ALLOWED_CACHE_MODES:
        # 显式 group 模式只允许 0700/0750/0710；其余（含 world-readable）拒绝。
        if is_strict_semantics():
            raise ValidationError(
                f"DATA_ACCESS_COS_CACHE_MODE={os.environ.get('DATA_ACCESS_COS_CACHE_MODE')} "
                "非法（只允许 0700 / 0750 / 0710；world-readable 拒绝）"
            )
        mode = 0o700
    try:
        os.chmod(root, mode)
    except OSError:
        pass
    return root


def _cache_mirror_spec(spec: MirrorSpec) -> MirrorSpec:
    """把 mirror 的本地根改到 cache，COS 前缀不变。

    R24 P0-S3 §5.1/§5.4：先 ``ensure_cache_root_secure``（root 权限校验 + 建目录）
    再返回——cache 根永远按 principal scope 隔离，权限过宽 fail-closed。
    """
    ensure_cache_root_secure()
    cos = spec.cos_prefix.rstrip("/")
    without_scheme = cos.split("://", 1)[-1]
    parts = without_scheme.split("/", 1)
    rel = parts[1] if len(parts) > 1 else parts[0]
    # R61-P1 #56：local_root 在**每次调用**时基于 ``cos_cache_root()`` 现算——
    # 不能用模块 import 时的快照。调用方（测试）可在每次请求前改
    # ``DATA_ACCESS_COS_CACHE_ROOT`` 指向独立 cache（每档冷启动）。
    base = cos_cache_root()
    return MirrorSpec(
        cos_prefix=spec.cos_prefix,
        local_root=base / rel,
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
    if cache_spec.layout in {"period_files", "prefixed_date_file", "event_files"}:
        # R25 P0-001/016：period/event/prefixed 布局 CLI 缓存也走完整目录 sync
        # （不能按 time_range 展开物理文件名），并收集命中 file_selector 的文件。
        from .mirror import _sync_dir_full

        _sync_dir_full(cache_spec)
        selector = getattr(cache_spec, "file_selector", None) or ""
        base_dir = _local_table_dir(cache_spec) if cache_spec.table else cache_spec.local_root
        for p in sorted(base_dir.glob(f"{selector}*.parquet")):
            if p.exists() and p.stat().st_size > 0:
                paths.append(str(p))
    elif cache_spec.layout == "daily_parquet":
        for day in expected_partitions(dataset_name, start, end, layout=cache_spec.layout):
            _sync_daily_file(cache_spec, day, dataset_name=dataset_name)
            from .mirror import _daily_filename

            dest = _local_table_dir(cache_spec) / _daily_filename(cache_spec, day)
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(str(dest))
    elif cache_spec.layout == "hive_date":
        for day in expected_partitions(dataset_name, start, end, layout=cache_spec.layout):
            _sync_hive_date(cache_spec, day)
            dest = cache_spec.local_root / f"date={day.isoformat()}" / cache_spec.file_name
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(str(dest))
    elif cache_spec.layout == "hive_year":
        for year in expected_partitions(dataset_name, start, end, layout=cache_spec.layout):
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
    """返回 (paths, backend)，backend 为 ``httpfs`` 或 ``cli``。

    R24 P0-S1：不再调用 ``load_cos_cli_credentials_into_env``（禁止 secret 写
    入 os.environ）；httpfs 凭证由 ``cos_remote_backend`` → ``resolve_s3_credentials``
    按 CredentialProvider 链解析。
    """
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
