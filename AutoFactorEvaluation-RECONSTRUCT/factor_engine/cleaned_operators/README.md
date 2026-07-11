# cleaned_operators — Runtime 算子库

本目录是 factor_engine 的 **唯一算子实现层**：`OperatorRegistry` 注册、`SeriesOperator.calculate()` 在 **宽表 panel**（index=时间, columns=标的）上执行。

- **DSL / 工厂**：[`api/`](../api/README.md)（`from api import rank` → `CleanedCall`）  
- **执行桥**：[`backend/cleaned_bridge.py`](../backend/cleaned_bridge.py)（MultiIndex ↔ panel）  
- **挖掘侧**：一般 **不 import 本目录**；查 [`operators_catalog.md`](operators_catalog.md) 与 [`_aliases.py`](_aliases.py) 即可。

>  monorepo 外还有一份源码副本：[`quantsociety/cleaned_operators/`](../../../../cleaned_operators/)。改算子时请 **同步** 到 `factor_engine/cleaned_operators/`。

---

## 1. 与 `api` 的分工（不重复）

| 问题 | 找谁 |
|------|------|
| 公式怎么写、怎么 import | `api` |
| 算子叫什么、有哪些别名 | `operators_catalog.md`、`_aliases.py` |
| 算子怎么算（rolling、rank 语义） | **本目录** `.py` 实现 |
| 字符串能否 parse | `api.operator_registry.build_dsl_allowlist()` |

```text
api.make_cleaned_call_factory("rank")(col("x"))
    → expr.CleanedCall
    → ir (op="rank")
    → backend → OperatorRegistry.get("rank").calculate(panel)
```

---

## 2. 目录结构

| 模块 | 内容 |
|------|------|
| [`registry.py`](registry.py) | `OperatorRegistry`：register / get / catalog / 别名 |
| [`base.py`](base.py) | `Operator`、`SeriesOperator`、`@register_operator` |
| [`_aliases.py`](_aliases.py) | DSL 别名 → canonical（**维护别名只改此文件**） |
| [`time_series.py`](time_series.py) | `ts_mean`、`ts_delay`、`SMA` 等 |
| [`cross_sectional.py`](cross_sectional.py) | `rank`、`zscore`、`neutralize` 等 |
| [`elementwise_math.py`](elementwise_math.py) | 四则、比较、`log`、`exp` 等 |
| [`technical_signal.py`](technical_signal.py) | `MACD`、`RSI`、`ADX` 等 |
| [`statistics_regression.py`](statistics_regression.py) | 相关、回归、统计矩 |
| [`group_neutralization.py`](group_neutralization.py) | `group_rank` 等 |
| [`intraday_microstructure.py`](intraday_microstructure.py) | 部分 LQTP 实现 + **未注册的 stub 函数** |
| [`operators_catalog.md`](operators_catalog.md) | 三源合并审计目录（含历史来源标注） |

加载入口：[`__init__.py`](__init__.py) 的 `load_all()`（import 全部模块并应用 `_aliases`）。

---

## 3. 数据形态约定

- factor_engine 中间结果为 **`(timestamp, instrument)` MultiIndex Series**。  
- `cleaned_bridge` 会 **unstack** 成宽表再调用 `calculate()`，结果 **stack** 回 MultiIndex。  
- 时序 rolling 在 **列方向**（每个 instrument 一列）上滚动；与旧版 per-groupby MultiIndex kernel 在边界上可能略有差异。

---

## 4. 新增 / 修改算子

1. 在合适模块添加 `SeriesOperator` 子类，使用 `@register_operator(...)`。  
2. 在 [`_aliases.py`](_aliases.py) 增加 DSL 别名（若需要）。  
3. 运行测试：  
   ```bash
   cd factor_engine
   PYTHONPATH=. pytest tests/test_cleaned_operators_comprehensive.py -q
   ```  
4. 更新 [`operators_catalog.md`](operators_catalog.md)（若由脚本生成则重跑生成脚本）。  
5. Gateway **`operator_whitelist.json`** 与 `build_dsl_allowlist()` 对齐（评估侧）。

**注册后自动生效的路径**：`build_dsl_allowlist()` → DSL；新建 `PandasBackend()` → 执行 kernel。

---

## 5. stub 与未实现算子

- `intraday_microstructure.py` 内大量 `*_stub` **普通函数**，多数 **未** `register_operator`，**不会**进入 DSL 白名单。  
- 仅 **catalog**、无 pandas runtime 的名字不会出现在 `build_dsl_allowlist()`。  
- 仅 **polars** backend 的算子（如部分 `ts_argmax`）同样不进 pandas DSL 白名单。

投递 manifest 时只使用 **`parse_expr` 能解析** 且 **本地试算不报错** 的算子名。

---

## 6. 冒烟示例

```bash
cd factor_engine
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -c "
from api.dsl_parser import parse_expr
from api.factor import Factor
from api.columns import col
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource
import pandas as pd

idx = pd.MultiIndex.from_product(
    [pd.date_range('2024-01-01', periods=5), ['A','B']],
    names=['timestamp','instrument'],
)
close = pd.Series(range(10), index=idx, dtype=float)
src = InMemorySeriesSource(data={'close': close})
expr = parse_expr(\"rank(SMA(col('close'), 3))\")
f = Factor(name='demo', expr=expr)
print(FactorEngine(backend=PandasBackend(), data_source=src).run(f)['result'])
"
```

---

## 7. 延伸阅读

- [`docs/算子与导入教程.md`](../docs/算子与导入教程.md)  
- [`docs/operators_semantics.md`](../docs/operators_semantics.md)  
- [`api/README.md`](../api/README.md)
