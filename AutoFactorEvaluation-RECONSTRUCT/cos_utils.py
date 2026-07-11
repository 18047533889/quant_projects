"""
COS 路径工具集

提供对 COS（腾讯云对象存储）路径的：
- 目录结构检查（ls）
- 写入权限验证（cp + rm 探针）
- factor_pool 合法性校验
- 缺失目录自动创建
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from path_convention import FACTOR_TREE, CANDIDATE_TREE, expected_subdirs


def _cos_parent(path: str) -> tuple[str, str]:
    """解析 COS URI 的父级 URL 和当前名称。

    Args:
        path: 形如 cos://bucket/prefix/dir

    Returns:
        (父级URL, 当前名称)
    """
    stripped = path.rstrip("/")
    if "/" not in stripped[6:]:
        return stripped + "/", stripped.split("/")[-1]
    parent_url = stripped[:stripped.rfind("/")] + "/"
    name = stripped.split("/")[-1]
    return parent_url, name


class CosPath:
    """封装一个 COS 路径的便捷操作。"""

    def __init__(self, path: str):
        self.path = path.rstrip("/")

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"CosPath({self.path!r})"

    def __truediv__(self, other: str) -> "CosPath":
        return CosPath(f"{self.path}/{other}")

    @property
    def is_cos(self) -> bool:
        """是否为 COS 路径（以 cos:// 开头）。"""
        return self.path.startswith("cos://")

    def list_dir(self) -> list[dict]:
        """执行 admin-cos ls，返回目录条目列表。

        Returns:
            每个条目为 {"key": str, "type": "DIR"|"FILE"}。
            空路径或不存在时返回 []。
        """
        if not self.is_cos:
            return []
        try:
            result = subprocess.run(
                ["admin-cos", "ls", f"{self.path}/"],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        if result.returncode != 0:
            return []

        entries = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if "|" not in line or "KEY" in line or "---" in line or "TOTAL" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                key = parts[0]
                typ = parts[1] if len(parts) > 1 else ""
                if key and typ:
                    dir_name = key.rstrip("/").split("/")[-1]
                    entries.append({"key": dir_name, "type": typ})
        return entries

    def exists(self) -> bool:
        """检查路径是否存在。"""
        if not self.is_cos:
            return Path(self.path).exists()

        try:
            # 方案一: 直接 ls 目标路径（空目录也返回 0 exit code）
            result = subprocess.run(
                ["admin-cos", "ls", f"{self.path}/"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return True

            # 方案二: 检查父级目录
            parent_url, current_name = _cos_parent(self.path)
            result2 = subprocess.run(
                ["admin-cos", "ls", parent_url],
                capture_output=True, text=True, timeout=30,
            )
            if result2.returncode != 0:
                return False
            for line in result2.stdout.splitlines():
                if current_name in line and "DIR" in line:
                    return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def ensure_dir(self) -> bool:
        """确保目录存在（COS 上传 .empty 标记文件以保持目录可见）。

        COS 对象存储以 key-prefix 模拟目录，空目录不会在 ls 中显示。
        因此通过保留一个 .empty 空文件来确保目录始终可见。

        Returns:
            True=已存在或已创建, False=创建失败。
        """
        if self.exists():
            return True

        marker = f"{self.path}/.empty"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".marker") as f:
                tmp = f.name
                f.write(b"")

            cp = subprocess.run(
                ["admin-cos", "cp", tmp, marker],
                capture_output=True, text=True, timeout=30,
            )
            Path(tmp).unlink(missing_ok=True)
            return cp.returncode == 0
        except Exception:
            return False

    def check_writable(self) -> tuple[bool, str]:
        """测试路径是否可写（上传临时文件再删除）。

        Returns:
            (可写: bool, 消息: str)
        """
        if not self.is_cos:
            return os.access(self.path, os.W_OK), "本地路径权限检查"

        probe_name = f"._probe_{id(self)}"
        probe_path = f"{self.path}/{probe_name}"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".probe") as f:
                tmp_path = f.name
                f.write(b"probe")

            cp = subprocess.run(
                ["admin-cos", "cp", tmp_path, probe_path],
                capture_output=True, text=True, timeout=30,
            )
            Path(tmp_path).unlink(missing_ok=True)

            if cp.returncode != 0:
                return False, f"上传失败: {cp.stderr.strip() or cp.stdout.strip()}"

            rm = subprocess.run(
                ["admin-cos", "rm", probe_path],
                capture_output=True, text=True, timeout=30,
                input="y\n",
            )
            return True, "可写"
        except subprocess.TimeoutExpired:
            return False, "超时"
        except FileNotFoundError:
            return False, "admin-cos 命令不可用"
        except Exception as e:
            return False, str(e)


# ============================================================
# 自动创建缺失的 COS 目录
# ============================================================


def _ensure_tree_dirs(base_path: str, tree: dict[str, str], label: str) -> list[str]:
    """通用：确保某基座下指定目录树的所有路径都存在。

    Args:
        base_path: 基座路径。
        tree: 目录树字典（如 FACTOR_TREE）。
        label: 日志标签（如 "factor_pool"）。

    Returns:
        操作记录列表。
    """
    records: list[str] = []
    base = CosPath(base_path)

    if not base.exists():
        if base.ensure_dir():
            records.append(f"✅ 创建 {label} 基座: {base}")
        else:
            records.append(f"❌ {label} 基座创建失败: {base}")
            return records
    else:
        records.append(f"✅ {label} 基座已存在: {base}")

    all_rel = sorted(set(tree.values()))

    # 先创建 tier 父目录（如有）
    tier_dirs: set[str] = set()
    for rel in all_rel:
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0].startswith("tier"):
            tier_dirs.add(parts[0])
    for tier in sorted(tier_dirs):
        tier_path = base / tier
        if not tier_path.exists():
            tier_path.ensure_dir()
            records.append(f"  └─ 创建 {tier}/")

    # 再创建每个子目录
    for rel in all_rel:
        full_path = base / rel
        if not full_path.exists():
            if full_path.ensure_dir():
                records.append(f"  └─ 创建 {rel}/")
        else:
            records.append(f"  ✔️  {rel}/ 已存在")

    return records


def ensure_factor_pool_dirs(factor_pool: str) -> list[str]:
    """确保 factor_pool 下所有 FACTOR_TREE 目录存在（tier1~tier4）。"""
    return _ensure_tree_dirs(factor_pool, FACTOR_TREE, "factor_pool")


def ensure_candidate_pool_dirs(candidate_pool: str) -> list[str]:
    """确保 candidate_pool 下所有 CANDIDATE_TREE 目录存在（candidate/, archive/）。"""
    return _ensure_tree_dirs(candidate_pool, CANDIDATE_TREE, "candidate_pool")


# ============================================================
# 校验器
# ============================================================


class BaseValidator:
    """COS 基座路径合法性校验器（通用）。"""

    def __init__(self, base_path: str, tree: dict[str, str], label: str = "",
                 auto_fix: bool = False):
        self.base = CosPath(base_path)
        self.tree = tree
        self.label = label or base_path
        self.auto_fix = auto_fix
        self._issues: list[str] = []

    @property
    def path(self) -> str:
        return str(self.base)

    def validate_all(self) -> list[str]:
        self._issues = []
        self._check_base_exists()
        self._check_structure()
        self._check_writable()
        return list(self._issues)

    def _check_base_exists(self):
        exists = self.base.exists()
        if not exists:
            self._issues.append(f"❌ {self.label} 基座路径不存在: {self.base}")
        else:
            self._issues.append(f"✅ {self.label} 基座路径存在: {self.base}")

    def _check_structure(self):
        # 检查 tier 结构（如有）
        expected = expected_subdirs(self.tree)
        for tier, required_subdirs in sorted(expected.items()):
            tier_path = self.base / tier
            actual_entries = tier_path.list_dir()
            actual_dirs = {e["key"] for e in actual_entries if e["type"] == "DIR"}
            required_set = set(required_subdirs)
            missing = required_set - actual_dirs
            extra = actual_dirs - required_set
            if not missing and not extra:
                self._issues.append(
                    f"✅ {tier}/: 子目录完整 ({len(required_subdirs)}个)"
                )
            else:
                if missing:
                    msg = f"⚠️  {tier}/ 缺少子目录: {', '.join(sorted(missing))}"
                    if self.auto_fix:
                        for sub in sorted(missing):
                            sub_path = tier_path / sub
                            if sub_path.ensure_dir():
                                msg += f" → 已创建 {sub}/"
                            else:
                                msg += f" → ❌ 创建失败 {sub}/"
                    self._issues.append(msg)
                if extra:
                    self._issues.append(
                        f"ℹ️  {tier}/ 额外子目录: {', '.join(sorted(extra))}"
                    )

        # 检查根级子目录（如 candidate/, archive/ 等非 tier 目录）
        root_dirs = sorted(set(
            rel for rel in self.tree.values() if "/" not in rel
        ))
        if root_dirs:
            actual_entries = self.base.list_dir()
            actual_dirs = {e["key"] for e in actual_entries if e["type"] == "DIR"}
            for d in root_dirs:
                if d in actual_dirs:
                    self._issues.append(f"✅ {d}/ 已存在")
                else:
                    msg = f"⚠️  缺少子目录: {d}/"
                    if self.auto_fix:
                        sub_path = self.base / d
                        if sub_path.ensure_dir():
                            msg += " → 已创建"
                        else:
                            msg += " → ❌ 创建失败"
                    self._issues.append(msg)

    def _check_writable(self):
        writable, msg = self.base.check_writable()
        if writable:
            self._issues.append(f"✅ 写入权限: {msg}")
        else:
            self._issues.append(f"❌ 写入权限: {msg}")


def validate_factor_pool(factor_pool: str, auto_fix: bool = False) -> list[str]:
    """校验 factor_pool 路径合法性。"""
    v = BaseValidator(factor_pool, FACTOR_TREE, "factor_pool", auto_fix=auto_fix)
    return v.validate_all()


def validate_candidate_pool(candidate_pool: str, auto_fix: bool = False) -> list[str]:
    """校验 candidate_pool 路径合法性。"""
    v = BaseValidator(candidate_pool, CANDIDATE_TREE, "candidate_pool", auto_fix=auto_fix)
    return v.validate_all()


def validate_factor_pool(factor_pool: str, auto_fix: bool = False) -> list[str]:
    """便利函数: 校验 factor_pool 路径合法性。

    Args:
        factor_pool: factor_pool 基座路径。
        auto_fix: 是否自动创建缺失目录。

    Returns:
        校验结果列表。
    """
    validator = FactorPoolValidator(factor_pool, auto_fix=auto_fix)
    return validator.validate_all()


# ── CLI 入口 ──

def main():
    """python -m cos_utils <type> <path> [--fix]

    type: factor_pool | candidate_pool
    """
    args = sys.argv[1:]
    auto_fix = "--fix" in args
    positional = [a for a in args if not a.startswith("--")]

    if len(positional) < 2:
        print("用法: python -m cos_utils <factor_pool|candidate_pool|candidate_pool_all> <path> [--fix]")
        print("  --fix   自动创建缺失目录")
        print("示例:")
        print("  python -m cos_utils factor_pool cos://qs-cold/factor_pool/us_stock/ --fix")
        print("  python -m cos_utils candidate_pool cos://qs-cold/candidate_pool/us_stock/ --fix")
        print("  python -m cos_utils candidate_pool_all cos://qs-cold/candidate_pool/us_stock/ --fix")
        sys.exit(1)

    kind = positional[0]
    path = positional[1]

    if kind == "factor_pool":
        results = validate_factor_pool(path, auto_fix=auto_fix)
    elif kind == "candidate_pool":
        results = validate_candidate_pool(path, auto_fix=auto_fix)
    elif kind == "candidate_pool_all":
        results = validate_candidate_pool(path, auto_fix=auto_fix)
        results.append("")
        results += validate_factor_pool(path.replace("candidate_pool", "factor_pool"), auto_fix=auto_fix) if "candidate_pool" in path else ["(跳过 factor_pool)"]
    else:
        print(f"未知类型: {kind}")
        sys.exit(1)

    print("\n".join(results))
    if any(r.startswith("❌") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
