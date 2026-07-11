"""初始化核心入口 — 服务启动时执行一次。

初始化内容:
  1. 加载 config.yaml（通过 ConfigManager）
  2. 从 local_tmp 推导 factor_engine_cache 路径
  3. 加载因子引擎算子库（内建或外部）
  4. 通过 DeepSeek 生成算子复杂度评分表
  5. 返回初始化状态（后续模块直接使用，无需重复初始化）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .operator_loader import load_operators, get_registered_operators
from .complexity_builder import build_complexity_table


def get_cache_path(cache_root: str) -> Path:
    """获取去重缓存文件路径。"""
    return Path(cache_root) / "gateway_dedup_cache.json"


_init_state: dict[str, Any] | None = None


def init_service(
    config_dir: str | Path | None = None,
    factor_engine_operators: str | None = None,
    build_complexity: bool = True,
) -> dict[str, Any]:
    """服务启动初始化 — 仅执行一次。

    Args:
        config_dir: config.yaml 所在目录。None=使用 AFVCONFIG 或默认路径。
        factor_engine_operators: 外部算子库路径。None=使用内建。
        build_complexity: 是否生成复杂度评分表。

    Returns:
        {config, local_tmp, cache_path, complexity_table, operator_count, ...}
    """
    global _init_state
    if _init_state is not None:
        return _init_state

    print("=" * 60)
    print("AutoFactorEvaluation 服务初始化")
    print("=" * 60)

    # 1. 加载配置
    print("\n[1/4] 加载配置文件...")
    from config_manager import ConfigManager
    config = ConfigManager(config_dir=config_dir)
    local_tmp = config.local_tmp
    cache_path = str(Path(local_tmp) / "cache" / "factor_engine")
    print(f"  local_tmp: {local_tmp}")
    print(f"  cache_path: {cache_path}")

    Path(cache_path).mkdir(parents=True, exist_ok=True)
    Path(local_tmp).mkdir(parents=True, exist_ok=True)

    # 2. 加载算子库
    print("\n[2/4] 加载算子库...")
    operator_count = load_operators(factor_engine_operators)
    print(f"  → {operator_count} 个算子已注册")

    # 3. 生成复杂度评分表
    complexity_table: dict[str, float] = {}
    if build_complexity:
        print("\n[3/4] 生成算子复杂度评分表 (DeepSeek)...")
        ops = get_registered_operators()
        complexity_table = build_complexity_table(ops)
        print(f"  → {len(complexity_table)} 个算子已评分")
        if complexity_table:
            sample = dict(list(complexity_table.items())[:5])
            print(f"  示例: {sample}")
    else:
        print("\n[3/4] 跳过复杂度评分表生成")
        complexity_table = {op: 1.0 for op in get_registered_operators()}

    # 4. 确保本地路径结构
    print("\n[4/4] 确保本地路径结构...")
    from path_convention import LOCAL_TREE, resolve_local
    for key in LOCAL_TREE:
        p = resolve_local(local_tmp, key)
        p.mkdir(parents=True, exist_ok=True)
    print(f"  → {len(LOCAL_TREE)} 个本地子路径已就绪")

    result = {
        "config": config,
        "local_tmp": local_tmp,
        "cache_path": cache_path,
        "operator_count": operator_count,
        "complexity_table": complexity_table,
        "initialized": True,
    }
    _init_state = result

    print("\n" + "=" * 60)
    print("初始化完成")
    print("=" * 60)
    return result


def get_init_state() -> dict[str, Any] | None:
    return _init_state
