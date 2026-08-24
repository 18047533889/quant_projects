"""因子落盘系统演示：完整的 写入 → 元数据查询 → 读取 流程。

运行前提
--------
- ``pip install pyarrow polars``  （polars 可选，退化为 Pandas）
- 无需真实行情数据，本脚本使用内存模拟数据

运行
----
::

    cd society
    python examples/materialize_demo.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.datasource import DataSource
from factor_engine.storage import ParquetMaterializer, build_result_store

# ============================================================================
# 1. 准备：内存数据源 + 因子定义
# ============================================================================


class InMemorySeriesSource(DataSource):
    """最简内存数据源。"""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self.data = data

    def load_column(self, name: str):
        return self.data[name]


# 构造 5 天 × 3 只标的 的模拟收盘价
dates = pd.to_datetime([
    "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
])
assets = ["000001.SZ", "000002.SZ", "600000.SH"]
idx = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
close = pd.Series(
    [10.0, 20.0, 30.0,
     11.0, 19.0, 31.0,
     12.0, 21.0, 29.0,
     10.5, 20.5, 32.0,
     13.0, 18.0, 28.0],
    index=idx,
)

# ============================================================================
# 2. 引擎计算因子
# ============================================================================

factor = Factor(
    name="mom_rank_v1",
    expr=rank(ts_mean(col("close"), 3)),
    freq="1d",
    description="3日均价截面排名",
)

engine = FactorEngine(
    backend=PandasBackend(),
    data_source=InMemorySeriesSource({"close": close}),
)
result_dict = engine.run(factor)
series = result_dict["result"]
ir_node = result_dict["analysis"].ir

print("=" * 60)
print("1. 引擎计算结果")
print("=" * 60)
print(series)
print()

# ============================================================================
# 3. 落盘
# ============================================================================

mat = ParquetMaterializer(lake_root="./demo_factor_lake")

summary = mat.materialize(
    factor_id="mom_rank_v1",
    result=series,
    ir_node=ir_node,          # 自动计算 AST Hash
    frequency="1d",
    description="3日均价截面排名 v1",
    expression='rank(ts_mean(col("close"), 3))',
)

print("=" * 60)
print("2. 落盘摘要")
print("=" * 60)
print(f"  factor_id:    {summary['factor_id']}")
print(f"  rows_written: {summary['rows_written']}")
print(f"  partitions:   {summary['partitions']}")
print(f"  watermark:    {summary['watermark']}")
print()

# ============================================================================
# 4. 查看因子目录
# ============================================================================

print("=" * 60)
print("3. 因子目录")
print("=" * 60)
for f in mat.list_factors():
    print(f"  {f['factor_id']:25s}  作者={f['author'] or '?':10s}  "
          f"频率={f['frequency']:5s}  "
          f"Hash={f['ast_hash'][:12]}…  "
          f"区间={f.get('start_date', 'N/A')} → {f.get('end_date', 'N/A')}")
print()

# ============================================================================
# 5. 读取（自动选择 Polars/Pandas 后端）
# ============================================================================

store = build_result_store("./demo_factor_lake")
print("=" * 60)
print(f"4. 读取后端: {type(store).__name__}")
print("=" * 60)

df = store.to_pandas(["mom_rank_v1"])
print(df.to_string(index=False))
print()

# ============================================================================
# 6. Hash 防呆演示
# ============================================================================

print("=" * 60)
print("5. Hash 防呆演示")
print("=" * 60)

try:
    mat.materialize(
        factor_id="mom_rank_v1",
        result=series,
        ast_hash="FAKE_DIFFERENT_HASH",  # 模拟公式被修改
    )
except Exception as e:
    print(f"  ✅ 防呆生效: {type(e).__name__}")
    print(f"     {e}")
print()

# ============================================================================
# 7. 清理演示数据
# ============================================================================

mat.delete_factor("mom_rank_v1", delete_files=True)
# 关闭所有 SQLite 连接后再删目录（Windows 文件锁）
mat.catalog.close()
store._catalog.close() if hasattr(store, '_catalog') else None
import shutil, os
if os.path.exists("./demo_factor_lake"):
    shutil.rmtree("./demo_factor_lake")
print("已清理演示数据。")
