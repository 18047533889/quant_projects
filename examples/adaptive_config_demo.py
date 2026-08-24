#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""演示自适应资源配置系统的使用。

这个示例展示如何在不同服务器环境下自动优化配置。
"""
import sys
sys.path.insert(0, '.')

from factor_engine.runtime.adaptive_config import auto_configure, get_adaptive_config


def demo_auto_configure():
    """演示 auto_configure() 的基本使用。"""
    print("=" * 70)
    print("1. 基本使用：自动配置并应用环境变量")
    print("=" * 70)

    # 启动时调用一次，自动检测系统资源并设置环境变量
    config = auto_configure(apply_env=True)

    print(f"\n检测到系统配置：")
    print(f"  - 总内存: {config['system_memory_gb']:.1f} GB")
    print(f"  - CPU 核心: {config['system_cpu_cores']} 个")

    print(f"\n自动优化的配置：")
    print(f"  DuckDB:")
    print(f"    - 线程数: {config['duckdb_threads']}")
    print(f"    - 内存限制: {config['duckdb_memory_limit']}")

    print(f"  Polars:")
    print(f"    - 线程数: {config['polars_threads']}")
    print(f"    - 流式块大小: {config['polars_streaming_chunk_size']:,}")

    print(f"  批处理:")
    print(f"    - 批量大小: {config['batch_size']:,} 行")
    print(f"    - DAG 编译批量: {config['dag_chunk_size']:,} 因子")
    print(f"    - 编译块大小: {config['compile_chunk_size']:,}")

    print(f"  并发:")
    print(f"    - 计算 workers: {config['max_workers_compute']}")
    print(f"    - I/O workers: {config['max_workers_io']}")

    print(f"  内存管理:")
    print(f"    - 单块最大: {config['block_abs_max_bytes'] / 1024**3:.1f} GB")
    print(f"    - 缓存大小: {config['cache_size_bytes'] / 1024**3:.1f} GB")
    print(f"    - 硬内存限制: {config['hard_memory_limit_bytes'] / 1024**3:.1f} GB")

    return config


def demo_different_environments():
    """演示不同服务器环境下的配置差异。"""
    print("\n" + "=" * 70)
    print("2. 不同服务器环境的配置对比")
    print("=" * 70)

    environments = [
        ("小型服务器 (8GB, 4核)", 8.0, 4),
        ("开发机 (30GB, 8核)", 30.0, 8),
        ("生产服务器 (128GB, 32核)", 128.0, 32),
        ("高性能服务器 (500GB, 64核)", 500.0, 64),
    ]

    print(f"\n{'环境':<25} {'DuckDB内存':<12} {'批量大小':<12} {'DAG块':<10} {'Workers':<10}")
    print("-" * 70)

    for name, memory_gb, cpu_cores in environments:
        cfg = get_adaptive_config(force_memory_gb=memory_gb, force_cpu_cores=cpu_cores)
        print(f"{name:<25} {cfg.duckdb_memory_limit:<12} {cfg.batch_size:<12,} "
              f"{cfg.compile_chunk_size:<10} {cfg.max_workers_compute:<10}")


def demo_environment_variable_override():
    """演示环境变量覆盖。"""
    print("\n" + "=" * 70)
    print("3. 环境变量覆盖示例")
    print("=" * 70)

    import os

    print("\n场景：用户希望手动控制 DuckDB 线程数")
    print("设置: export DUCKDB_THREADS=16")

    # 模拟用户设置环境变量
    os.environ["DUCKDB_THREADS"] = "16"

    # auto_configure 会尊重已有的环境变量
    cfg = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

    print(f"\n结果:")
    print(f"  - 自动检测建议: 8 线程")
    print(f"  - 用户环境变量: 16 线程")
    print(f"  - 实际使用: {cfg.duckdb_threads} 线程 (尊重用户设置)")
    print(f"  - 配置来源: {cfg.config_source['duckdb_threads']}")

    # 清理
    del os.environ["DUCKDB_THREADS"]


def demo_programmatic_usage():
    """演示程序化使用。"""
    print("\n" + "=" * 70)
    print("4. 程序化使用示例")
    print("=" * 70)

    # 获取配置但不应用环境变量
    config = auto_configure(apply_env=False, force_memory_gb=30.0, force_cpu_cores=8)

    print("\n在代码中直接使用配置值：")
    print(f"""
# 配置 DuckDB 连接
import duckdb
conn = duckdb.connect()
conn.execute(f"SET threads = {config['duckdb_threads']}")
conn.execute(f"SET memory_limit = '{config['duckdb_memory_limit']}'")

# 配置 Polars
import polars as pl
pl.Config.set_streaming_chunk_size({config['polars_streaming_chunk_size']})

# 使用批量大小
batch_size = {config['batch_size']}
for batch in data_iterator(batch_size=batch_size):
    process(batch)

# 配置并发 executor
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers={config['max_workers_compute']})
""")


def demo_memory_scaling():
    """演示内存缩放策略。"""
    print("\n" + "=" * 70)
    print("5. 内存缩放策略")
    print("=" * 70)

    print("\n配置如何随内存大小缩放：")

    memory_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\n{'内存':<10} {'DuckDB内存':<15} {'批量大小':<15} {'编译块':<15}")
    print("-" * 55)

    for mem_gb in memory_sizes:
        cfg = get_adaptive_config(force_memory_gb=float(mem_gb), force_cpu_cores=8)
        print(f"{mem_gb}GB{'':<6} "
              f"{cfg.duckdb_memory_limit:<15} "
              f"{cfg.batch_size:<15,} "
              f"{cfg.compile_chunk_size:<15}")

    print(f"\n策略说明：")
    print(f"  - DuckDB 内存: 固定 50% 系统内存（线性）")
    print(f"  - 批量大小: 平方根缩放（小内存保守，大内存充分利用）")
    print(f"  - 编译块: 平方根缩放，有最小/最大值限制")


def demo_integration_example():
    """演示集成到实际项目。"""
    print("\n" + "=" * 70)
    print("6. 集成到项目示例")
    print("=" * 70)

    print("""
在项目的 __init__.py 或主入口添加：

```python
# factor_engine/__init__.py

from factor_engine.runtime.adaptive_config import auto_configure

# 启动时自动配置
_adaptive_config = auto_configure(apply_env=True)

# 可选：导出配置供其他模块使用
RUNTIME_CONFIG = _adaptive_config

print(f"FactorEngine initialized for {_adaptive_config['system_memory_gb']:.1f}GB system")
```

然后在其他模块中：

```python
# 方式1：从环境变量读取（推荐）
import os
batch_size = int(os.environ.get('FE_BATCH_SIZE', 100000))

# 方式2：从全局配置读取
from factor_engine.runtime.adaptive_config import get_global_adaptive_config
config = get_global_adaptive_config()
batch_size = config.batch_size

# 方式3：使用便捷函数
from factor_engine.runtime.adaptive_config import get_compile_chunk_size
chunk_size = get_compile_chunk_size()
```

这样，代码在任何服务器上都能自动优化配置！
""")


def main():
    """运行所有演示。"""
    print("\n" + "=" * 70)
    print("自适应资源配置系统演示")
    print("=" * 70)

    try:
        demo_auto_configure()
        demo_different_environments()
        demo_environment_variable_override()
        demo_programmatic_usage()
        demo_memory_scaling()
        demo_integration_example()

        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        print("\n主要特性：")
        print("  ✓ 自动检测系统资源（内存、CPU）")
        print("  ✓ 智能缩放配置（8GB 到 500GB+ 都能优化）")
        print("  ✓ 环境变量覆盖（用户控制优先）")
        print("  ✓ 零配置启动（开箱即用）")
        print("  ✓ 跨服务器一致性（同一份代码，自动适配）")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
