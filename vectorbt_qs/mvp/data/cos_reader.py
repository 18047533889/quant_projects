"""
COS（腾讯云对象存储）读取器

支持 cos://bucket/key 协议，自动选择后端：
  1. clean-cos / raw-cos CLI（Server A/B 标准方式，优先）
  2. Python COS SDK（需密钥或 CAM 角色，fallback）

同时兼容本地文件系统路径。

用法:
    from mvp.data.cos_reader import CosReader
    reader = CosReader()
    files = reader.list_objects("cos://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/")
    df = reader.read_parquet("cos://qs-cold/.../2025-01-02.parquet")
"""

import os
import re
import subprocess
import tempfile
import shutil
import hashlib
from typing import List, Optional
from pathlib import Path


# ============================================================
# 路径解析
# ============================================================

def is_cos_path(path: str) -> bool:
    return path.startswith("cos://")


def parse_cos_uri(uri: str):
    """解析 cos://bucket/key"""
    if not uri.startswith("cos://"):
        raise ValueError(f"非 COS URI: {uri}")
    path = uri[6:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# ============================================================
# CLI 后端 — 使用 clean-cos / raw-cos 命令
# ============================================================

def _find_cos_cli(key_prefix: str) -> Optional[str]:
    """根据 key 前缀找到对应的 CLI: raw_data→raw-cos, 其他→clean-cos"""
    for prefix, cli in [("raw_data", "raw-cos"), ("clean_data", "clean-cos")]:
        if key_prefix.startswith(prefix) and shutil.which(cli):
            return cli
    for cli in ["clean-cos", "raw-cos"]:
        if shutil.which(cli):
            return cli
    return None


def _cli_list(bucket: str, prefix: str, cli: str = "clean-cos") -> List[str]:
    """clean-cos ls → 解析表格输出为 URI 列表"""
    cos_path = f"cos://{bucket}/{prefix}"
    result = subprocess.run([cli, "ls", cos_path], capture_output=True, text=True, timeout=60)

    uris = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 2:
            key = parts[0].strip()
            # 跳过表头和无意义的汇总行
            if not key or key == "KEY" or key.startswith("TOTAL") or key.startswith("---"):
                continue
            if not key.endswith("/"):
                uris.append(f"cos://{bucket}/{key}")
    return sorted(uris)


def _cli_download(bucket: str, key: str, local: str, cli: str = "clean-cos") -> bool:
    """clean-cos cp cos://bucket/key local"""
    os.makedirs(os.path.dirname(local), exist_ok=True)
    result = subprocess.run(
        [cli, "cp", f"cos://{bucket}/{key}", local],
        capture_output=True, text=True, timeout=120
    )
    return result.returncode == 0 and os.path.exists(local)


# ============================================================
# SDK 后端 — 使用 cos-python-sdk-v5
# ============================================================

_cos_client = None

def _get_sdk_client():
    global _cos_client
    if _cos_client is not None:
        return _cos_client
    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(
        Region=os.environ.get("COS_REGION", "ap-guangzhou"),
        SecretId=os.environ.get("COS_SECRET_ID"),
        SecretKey=os.environ.get("COS_SECRET_KEY"),
        Token=os.environ.get("COS_TOKEN") or None,
        Scheme="https",
    )
    _cos_client = CosS3Client(config)
    return _cos_client


def _sdk_list(bucket: str, prefix: str, max_keys: int = 10000) -> List[str]:
    client = _get_sdk_client()
    uris, marker = [], ""
    while len(uris) < max_keys:
        resp = client.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker, MaxKeys=min(1000, max_keys - len(uris)))
        if "Contents" not in resp:
            break
        for obj in resp["Contents"]:
            if not obj["Key"].endswith("/"):
                uris.append(f"cos://{bucket}/{obj['Key']}")
        if resp.get("IsTruncated") == "false":
            break
        marker = resp.get("NextMarker", "")
    return sorted(uris)


def _sdk_download(bucket: str, key: str, local: str) -> bool:
    client = _get_sdk_client()
    os.makedirs(os.path.dirname(local), exist_ok=True)
    resp = client.get_object(Bucket=bucket, Key=key)
    resp["Body"].get_stream_to_file(local)
    return os.path.exists(local)


# ============================================================
# CosReader — 统一接口
# ============================================================

class CosReader:
    """COS 文件读取器。自动优先使用 CLI，不可用时降级 SDK。"""

    def __init__(self, cache_dir: Optional[str] = None, backend: str = "auto"):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "cache"
            )
        self._cache_dir = cache_dir
        self._backend = backend
        os.makedirs(self._cache_dir, exist_ok=True)

    def _detect(self, key: str) -> str:
        if self._backend != "auto":
            return self._backend
        return "cli" if _find_cos_cli(key) else "sdk"

    def _cache_path(self, uri: str) -> str:
        """将 cos:// URI 映射到本地缓存路径（含原文件名）"""
        if not is_cos_path(uri):
            return uri
        bucket, key = parse_cos_uri(uri)
        h = hashlib.md5(f"{bucket}/{os.path.dirname(key)}".encode()).hexdigest()[:8]
        return os.path.join(self._cache_dir, f"{h}_{os.path.basename(key)}")

    def _download(self, uri: str) -> str:
        local = self._cache_path(uri)
        if os.path.exists(local):
            return local
        bucket, key = parse_cos_uri(uri)
        backend = self._detect(key)
        ok = False
        if backend == "cli":
            cli = _find_cos_cli(key) or "clean-cos"
            ok = _cli_download(bucket, key, local, cli)
        else:
            ok = _sdk_download(bucket, key, local)
        if not ok:
            raise RuntimeError(f"下载失败: {uri}")
        return local

    def list_objects(self, prefix: str, max_keys: int = 10000) -> List[str]:
        if not is_cos_path(prefix):
            import glob
            return sorted(glob.glob(os.path.join(prefix, "*")))

        bucket, key_prefix = parse_cos_uri(prefix)
        if key_prefix and not key_prefix.endswith("/"):
            key_prefix += "/"

        backend = self._detect(key_prefix)
        if backend == "cli":
            cli = _find_cos_cli(key_prefix) or "clean-cos"
            return _cli_list(bucket, key_prefix, cli)[:max_keys]
        return _sdk_list(bucket, key_prefix, max_keys)

    def glob(self, pattern: str, suffix: str = ".parquet") -> List[str]:
        """glob 匹配。COS 模式下前缀过滤 + 后缀过滤。"""
        if not is_cos_path(pattern):
            import glob
            return sorted(glob.glob(pattern, recursive=True))

        bucket, key_pattern = parse_cos_uri(pattern)
        prefix = key_pattern
        match_prefix = ""
        if "*" in key_pattern:
            idx = key_pattern.index("*")
            prefix = key_pattern[:idx].rstrip("/")
            match_prefix = os.path.basename(key_pattern[:idx])

        all_objs = self.list_objects(f"cos://{bucket}/{prefix}")
        result = []
        for uri in all_objs:
            _, key = parse_cos_uri(uri)
            fname = os.path.basename(key)
            if suffix and not fname.endswith(suffix):
                continue
            if match_prefix and not fname.startswith(match_prefix):
                continue
            result.append(uri)
        return result

    def read_parquet(self, path: str, **kwargs):
        import pandas as pd
        local = self._download(path) if is_cos_path(path) else path
        return pd.read_parquet(local, **kwargs)

    def read_parquet_batch(self, paths: List[str], **kwargs):
        import pandas as pd
        frames = [self.read_parquet(p, **kwargs) for p in paths]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def clear_cache(self, days: Optional[int] = None):
        if days:
            import time
            t = time.time() - days * 86400
            for f in Path(self._cache_dir).glob("*"):
                if f.stat().st_mtime < t:
                    f.unlink()
        else:
            if os.path.exists(self._cache_dir):
                shutil.rmtree(self._cache_dir)
                os.makedirs(self._cache_dir, exist_ok=True)


# ============================================================
# 全局单例
# ============================================================

_cos_reader: Optional[CosReader] = None

def get_cos_reader() -> CosReader:
    global _cos_reader
    if _cos_reader is None:
        _cos_reader = CosReader()
    return _cos_reader
