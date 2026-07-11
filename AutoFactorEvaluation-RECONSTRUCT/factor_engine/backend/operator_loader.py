"""外部算子库加载器。

负责将用户提供的外部算子库（绝对路径下的 .py 文件）预编译为 .pyc，
并加载到解释器环境以供 ``operator_registry`` 使用。
"""

from __future__ import annotations

import importlib
import importlib.util
import py_compile
import sys
from pathlib import Path
from typing import Any

from logging_utils import get_logger

logger = get_logger("backend.operator_loader")


def load_external_operators(operators_path: str) -> list[str]:
    """加载外部算子库。

    流程:
      1. 扫描目标路径下的所有 .py 文件
      2. 预编译为 .pyc（进度展示）
      3. 逐个加载到 sys.modules

    Args:
        operators_path: 外部算子库的绝对路径。

    Returns:
        已加载的模块名列表。
    """
    base = Path(operators_path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"外部算子库路径不存在: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"外部算子库路径不是目录: {base}")

    py_files = sorted(base.rglob("*.py"))
    if not py_files:
        logger.warning("外部算子库路径下未找到 .py 文件: %s", base)
        return []

    loaded: list[str] = []
    total = len(py_files)

    # ── 阶段 1: 预编译 ──
    logger.info("=" * 50)
    logger.info("阶段 1/2: 预编译外部算子 (%d 个文件)", total)
    logger.info("=" * 50)
    compiled_count = 0
    for idx, py_file in enumerate(py_files, 1):
        rel_path = py_file.relative_to(base)
        try:
            py_compile.compile(py_file, doraise=True, cfile=str(py_file) + "c")
            compiled_count += 1
            logger.info("  [%d/%d] ✅ 编译: %s", idx, total, rel_path)
        except py_compile.PyCompileError as e:
            logger.warning("  [%d/%d] ⚠️  编译失败 %s: %s", idx, total, rel_path, e)

    logger.info("预编译完成: %d/%d 成功", compiled_count, total)

    # ── 阶段 2: 加载到解释器 ──
    logger.info("=" * 50)
    logger.info("阶段 2/2: 加载外部算子到解释器 (%d 个文件)", total)
    logger.info("=" * 50)
    for idx, py_file in enumerate(py_files, 1):
        rel_path = py_file.relative_to(base)
        try:
            module_name = _pyfile_to_modname(py_file, base)
            if module_name in sys.modules:
                logger.info("  [%d/%d] ⏭️  已存在: %s (%s)", idx, total, module_name, rel_path)
                loaded.append(module_name)
                continue

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("  [%d/%d] ⚠️  无法创建 spec: %s", idx, total, rel_path)
                continue

            module = importlib.util.module_from_spec(spec)
            # 注入父包路径以确保相对导入正常
            _ensure_parent_package(module_name, base)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(module_name)
            logger.info("  [%d/%d] ✅ 加载: %s (%s)", idx, total, module_name, rel_path)

        except Exception as e:
            logger.warning("  [%d/%d] ⚠️  加载失败 %s: %s", idx, total, rel_path, e)

    logger.info("外部算子加载完成: %d/%d 成功", len(loaded), total)
    return loaded


def _pyfile_to_modname(py_file: Path, base: Path) -> str:
    """将 .py 文件路径转换为 Python 模块名。

    例如: /path/to/ops/sub/foo.py → ops.sub.foo
    """
    rel = py_file.relative_to(base)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


def _ensure_parent_package(module_name: str, base: Path):
    """确保父级包路径在 sys.modules 中，以支持相对导入。"""
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            parent_path = base / "/".join(parts[1:i])
            parent_module = type(sys)(parent)
            parent_module.__path__ = [str(parent_path)]
            parent_module.__package__ = parent
            sys.modules[parent] = parent_module


# ── 便捷函数: 从引擎配置加载 ──

def maybe_load_operators_from_config(config: Any) -> bool:
    """如果配置指定了外部算子库，则加载之。

    Args:
        config: FactorEngineConfig 对象或类似结构。

    Returns:
        是否加载了外部算子。
    """
    operators_path = getattr(config, "factor_engine_operators", None) or (
        getattr(config, "engine", None) and getattr(config.engine, "factor_engine_operators", None)
    )
    perf_path = getattr(config, "factor_engine_operators", None)

    path = operators_path or perf_path
    if not path:
        return False

    loaded = load_external_operators(str(path))
    return len(loaded) > 0
