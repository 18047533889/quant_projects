"""R40 #80: wheel package layout — [tool.setuptools.packages.find] include
must cover market*/security*/mining*/semantic* (they were missing → those
packages never entered the wheel)."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _packages_find_include() -> list[str]:
    pyproject = _REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    section = text.split("[tool.setuptools.packages.find]", 1)[1]
    include_block = section.split("include =", 1)[1].split("]", 1)[0]
    return [line.strip().strip('",') for line in include_block.splitlines() if line.strip()]


def test_packages_find_include_contains_all_packages():
    include = _packages_find_include()
    for pattern in ("api*", "backend*", "runtime*", "service*", "storage*"):
        assert pattern in include, f"missing baseline include {pattern!r}"
    # R40 #80: these were missing from the wheel
    for pattern in ("market*", "security*", "mining*", "semantic*"):
        assert pattern in include, f"missing include {pattern!r}"


def test_market_security_mining_semantic_packages_exist():
    # 这些包目录必须存在且可被 find_packages 的 include 匹配（semantic 当前
    # 缺 __init__.py，属独立打包问题 —— 这里只验证目录存在 + 核心包有 __init__）。
    for pkg in ("market", "security", "mining", "semantic"):
        assert (_REPO_ROOT / pkg).is_dir(), f"{pkg} package dir missing"
    for pkg in ("market", "security", "mining"):
        assert (_REPO_ROOT / pkg / "__init__.py").is_file(), f"{pkg} package missing __init__.py"
