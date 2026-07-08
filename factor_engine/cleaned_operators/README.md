# cleaned_operators — Runtime 算子库

本目录是 factor_engine 的 **唯一算子实现层**：`OperatorRegistry` 注册、`SeriesOperator.calculate()` 在 **宽表 panel**（index=时间, columns=标的）上执行。

- **DSL / 工厂**：[`api/`](../api/README.md)（`from api import rank` → `CleanedCall`）  
- **执行桥**：[`backend/cleaned_bridge.py`](../backend/cleaned_bridge.py)（MultiIndex ↔ panel）  
- **算子文档**：[`docs/`](docs/README.md)（全览、catalog、语义公式 — **与代码分离**）  
- **挖掘侧**：一般 **不 import 本目录**；查 [`docs/operators_catalog.md`](docs/operators_catalog.md) 与 [`_aliases.py`](_aliases.py) 即可。

> monorepo 外还有一份源码副本：[`quantsociety/cleaned_operators/`](../../../../cleaned_operators/)。改算子时请 **同步** 到 `factor_engine/cleaned_operators/`。

---

## 1. 与 `api` 的分工（不重复）

| 问题 | 找谁 |
|------|------|
| 公式怎么写、怎么 import | `api` |
| 算子叫什么、有哪些别名 | [`docs/operators_catalog.md`](docs/operators_catalog.md)、[`_aliases.py`](_aliases.py) |
| 算子含义与公式 | [`docs/算子全览.md`](docs/算子全览.md) |
| 算子怎么算（rolling、rank 语义） | **本目录** `*.py` 实现 |
| 字符串能否 parse | `api.operator_registry.build_dsl_allowlist()` |

```text
api.make_cleaned_call_factory("rank")(col("x"))
    → expr.CleanedCall
    → ir (op="rank")
    → backend → OperatorRegistry.get("rank").calculate(panel)
```

---

## 2. 目录结构

```text
cleaned_operators/
├── registry.py, base.py, base_polars.py   # 注册中心与基类
├── _aliases.py, _dedupe.py              # 别名与去重
├── _numpy_kernels.py, _rolling_fast.py   # 内部加速内核
├── common/                              # 共通运算（不绑特定数据域）
│   ├── elementwise.py                   # 四则、log、矩阵、clip …
│   ├── time_series.py                   # ts_mean、ts_corr、ts_decay_linear …
│   ├── shift_cum.py                     # 滞后、差分、累计
│   ├── cross_sectional.py               # rank、zscore、scale …
│   ├── group.py                         # group_neutralize、group_rank …
│   ├── data_cleaning.py                 # fillna、winsorize、window_* …
│   ├── statistics.py                    # 相关、回归、假设检验
│   └── polars_ops.py                    # polars 高频算子扩展
├── price_volume/ops.py                  # 收益、波动、夏普、beta …
├── technical/signal.py                  # MACD、RSI、trade_when …
├── fundamental/ops.py                   # ttm、yoy、quarter …
├── microstructure/ops.py                # 日内微观结构
└── docs/                                # 算子文档（与代码分离）
```

根目录仍保留 **兼容 shim**（如 `elementwise_math.py` → `common/elementwise.py`），旧 import 路径可继续使用。

| 位置 | 内容 |
|------|------|
| [`registry.py`](registry.py) | `OperatorRegistry`：register / get / catalog / 别名 |
| [`common/`](common/) | 元素数学、时序、截面、分组、清洗、统计 |
| [`price_volume/`](price_volume/) | 量价衍生 |
| [`technical/`](technical/) | 技术指标与信号 |
| [`fundamental/`](fundamental/) | 财报衍生 |
| [`microstructure/`](microstructure/) | 微观结构 |
| [`docs/`](docs/README.md) | 算子全览、catalog、语义公式 |

加载入口：[`__init__.py`](__init__.py) 的 `load_all()`（import 全部模块并应用 `_aliases` + `_dedupe`）。

---

## 2.1 命名规范（canonical vs 别名）

| 类型 | canonical 示例 | 常见别名（仍可计算） |
|------|----------------|----------------------|
| 时序滚动 | `ts_mean`, `ts_var`, `ts_pct` | `SMA`, `m_var`, `returns`, `pct_change` |
| 截面 | `rank`, `zscore` | `CS_RANK`, `standardize`, `panel_zscore` |
| 分组中性 | `group_neutralize` | `neutralize`, `group_demean`, `industry_neutralize` |
| 裁剪 | `clip` | `cap`, `clamp`, `CLIP` |
| 扩展统计 | `expanding_mean`, `expanding_zscore` | `cum_avg`, `cum_standardize` |
| 技术指标 | `RSI`, `MACD`, `WMA` | `ts_rsi`, `ts_macd`, `ts_wma` |

新增算子时优先选用上表 canonical 风格；旧 DSL 名在 [`_aliases.py`](_aliases.py) 登记即可。

---

## 2.2 执行 backend（哪个快用哪个）

- 默认 **`FACTOR_ENGINE_OPERATOR_BACKEND=auto`**：某算子有 polars 实现则走 polars，否则回退 pandas。
- 强制 pandas：``FACTOR_ENGINE_OPERATOR_BACKEND=pandas_numpy``
- 强制 polars（无实现则回退）：``FACTOR_ENGINE_OPERATOR_BACKEND=polars``
- 使用 ``build_backend("polars")`` 的 ``FactorEngine`` 与 pandas 路径数值对齐；polars 实现见 ``common/time_series.py``、``common/cross_sectional.py``、``common/polars_ops.py``。

---

## 3. 数据形态约定

- factor_engine 中间结果为 **`(timestamp, instrument)` MultiIndex Series**。  
- `cleaned_bridge` 会 **unstack** 成宽表再调用 `calculate()`，结果 **stack** 回 MultiIndex。  
- 时序 rolling 在 **列方向**（每个 instrument 一列）上滚动；与旧版 per-groupby MultiIndex kernel 在边界上可能略有差异。

---

## 4. 新增 / 修改算子

1. 在合适子目录添加 `SeriesOperator` 子类（共通 → `common/`，量价 → `price_volume/`，等），使用 `@register_operator(...)`。  
2. 在 [`_aliases.py`](_aliases.py) 增加 DSL 别名（若需要）。  
3. 运行测试：  
   ```bash
   cd factor_engine
   PYTHONPATH=. pytest tests/test_cleaned_operators_comprehensive.py -q
   ```  
4. 若语义变化，更新 [`docs/operator_doc_semantics.py`](docs/operator_doc_semantics.py) 并重跑 [`generate_operators_guide.py`](../scripts/generate_operators_guide.py)。  
5. Gateway **`operator_whitelist.json`** 与 `build_dsl_allowlist()` 对齐（评估侧）。

**注册后自动生效的路径**：`build_dsl_allowlist()` → DSL；新建 `PandasBackend()` → 执行 kernel。

---

## 5. stub 与未实现算子

- `microstructure/ops.py` 内部分 `*_stub` **普通函数** 可能 **未** `register_operator`，**不会**进入 DSL 白名单。  
- 仅 **catalog**、无 pandas runtime 的名字不会出现在 `build_dsl_allowlist()`。  
- 仅 **polars** backend 的算子同样不进 pandas DSL 白名单。

投递 manifest 时只使用 **`parse_expr` 能解析** 且 **本地试算不报错** 的算子名。

---

## 6. 延伸阅读

- [`docs/算子全览.md`](docs/算子全览.md)  
- [`docs/算子与导入教程.md`](../docs/算子与导入教程.md)  
- [`docs/operators_semantics.md`](../docs/operators_semantics.md)  
- [`api/README.md`](../api/README.md)
