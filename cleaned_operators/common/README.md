# `cleaned_operators/common` — 通用算子

时序、截面、分组、清洗、统计等 **跨品类共用** 的 Polars / Pandas 实现。

## 子模块

| 文件 | 内容 |
|------|------|
| `time_series.py` | 滚动均值/标准差/排名、delay、decay 等 |
| `cross_sectional.py` | rank、zscore、neutralize、winsorize |
| `elementwise.py` | 四则、log、abs、clip |
| `group.py` | group_neutralize、group_rank |
| `data_cleaning.py` | 缺失填充、inf 处理 |
| `statistics.py` | 相关、回归辅助 |
| `polars_ops.py` | Polars 核心算子入口 |
| `polars_extended.py` / `polars_math_extended.py` | 扩展数学函数 |

注册：各模块 `@register` → [`../registry.py`](../registry.py)。

语义详见 [`../../docs/operators_semantics.md`](../../docs/operators_semantics.md)。
