# R57 接手（WorkBuddy）第一轮：ADX 收口、缺失算子注册、R28 收敛与两处新发现

日期：2026-09-19（Asia/Hong_Kong）。基线 HEAD `ad216adf`。本文只记录**本轮有证据**的结论。

## 1. ADX：engine 级 polars/pandas 不一致已定位并修复

### 根因（此前未定位）

`factor_engine/cleaned_operators/layer_composite_fixes.py::pl_adx_strict` 的 `true_range_valid`
守卫里多了一个 `pl.col("c").is_not_null()`：

```python
true_range_valid = (
    pl.col("h").is_not_null()
    & pl.col("l").is_not_null()
    & pl.col("c").is_not_null()      # <-- 多余
    & previous_close.is_not_null()
)
```

pandas 参考 `pd_adx`（`cleaned_operators/overhaul/technical.py`）里
`tr[t] = max(h-l, |h-c[t-1]|, |l-c[t-1]|)` **只需要 `c[t-1]`**，不需要 `c[t]`。
因此当某根 bar 自身 close 缺失（例如停牌/缺失填充）时，polars 版会把该 bar 的 tr 误置为 null，
多出一个"缺失观测"，把 Wilder EWM 的 `min_samples` 门控整体推后一根，
进而 `atr → pdi/mdi → dx → adx` 全线错位。

### 证据（逐列对照，同一样本）

对同一份 2 股票 × 50 日样本，逐列比较 pandas 与 polars：

| 中间量 | 修复前 | 修复后 |
| --- | --- | --- |
| `tr` | 一致 | 一致 |
| `plus` / `minus` | 一致 | 一致 |
| `atr` first_valid | pandas 15 / polars **16** | 一致 |
| `dx` | 一致（maxdiff 2.5e-14） | 一致 |
| `adx` first_valid | pandas 28 / polars **29**，首个重叠值 45.860404 vs 38.606550 | 一致 |

另有独立验证：`recursive_kernel.adx_segment()` 与 `pd_adx` **逐值完全相同**（47.338971, 45.860404, …），
说明修复后的 polars 版与"单一权威递归核"和 pandas 参考三方一致。

### 落地改动

- `factor_engine/cleaned_operators/layer_composite_fixes.py`：移除多余的当前 close 守卫（附原因注释）。
- `factor_engine/backend/operator_semantic_version.py`：显式声明 `"ADX": 2`
  （`_canonical_semantic_version` 会渲染成 `"2.0"`，与既有派生值一致，但现在属于**已声明**）。
- 新测试 `factor_engine/tests/operators/test_r56_adx_physical_contract.py`（前一轮子代理留下，本轮纳入提交）。

### 验证

`factor_engine/tests/operators/test_polars_phase2_parity.py -k adx`：**1 passed**
（修复前 1 failed）。

## 2. 那两条 `unknown ... lacks PhysicalImplementationSpec` warning：**不是 ADX 引起**

交接文档把这两条 warning 归因于"ADX reconcile"，经查**不成立**：

- ADX 的 polars 槽现在**已经有**显式 `PhysicalImplementationSpec`，且 digest 已由
  `complete_physical_specs()` 绑定到当前代码（`implementation_source_hash` 等为本轮真实哈希）。
- 警告实际产生于 `cleaned_operators/polars_gap_coverage.py::_reconcile_delegate_metadata()`
  第 146 行调用 `canonical_polars_kind(canonical)`（默认 `production_mode=True`）。
- 触发的算子类既没有 `.canonical` 属性（所以文案是 "Operator unknown"），也没有 `_physical_spec`：
  `PolarsFundamental_fin_beat_streak`、`PolarsFundamental_fin_miss_streak`、
  `PolarsFundamental_fin_actual_expectation_divergence`、
  `PolarsFundamental_fin_days_since_expectation_revision`、
  `PolarsFundamental_fin_expectation_revision_count`、
  `PolarsFundamental_fin_expectation_revision_magnitude`、`TSReturnsNative`。

### 顺带查出的大面积问题（未修，需用户决策）

全量 polars 槽中**有 713 个没有显式 `PhysicalImplementationSpec`**。
由于 `polars_backend_kind` 在 `production_mode=True` 下 fail-closed，
`_reconcile_delegate_metadata()` 对这些槽一律判为 `UNSUPPORTED`，
**从不执行 `_stamp_delegate_meta()`** —— 也就是说这个"重新标注 delegate 元数据"的环节
对其声称要覆盖的绝大多数槽实际上是空转。这是设计层面的问题，不在本轮改动范围，故未改。

## 3. 34 个缺失公共算子：已接通，并经独立 A/B 验证契约零变化

- 新增 `factor_engine/cleaned_operators/_missing_public_bootstrap.py`，
  并在 `cleaned_operators/__init__.py` 的 `BOOTSTRAP_MODULE_SPECS` 末尾追加该模块。
- 该模块在 building 窗口导入 11 个源模块注册 34 个名字，然后**逐字段回滚**对既有算子的附带改动。

### 独立验证（非子代理自测）

用 fresh 进程（无 pytest conftest 预热）对**同一个工作树**做 A/B：
临时 `git apply -R` 撤下 `__init__.py` 的 14 行 hook，dump 全量注册表结构指纹，再恢复。

| 指标 | 无 hook（baseline） | 有 hook（candidate） |
| --- | ---: | ---: |
| catalog 算子数 | 1789 | 1823 |
| alias 数 | 307 | 307 |
| `_operators` 键数 | 1789 | 1823 |
| 既有算子字段变化 | — | **0** |
| alias 新增/删除/改指向 | — | **0 / 0 / 0** |
| 移除的算子 | — | **0** |
| 新增集合 == 允许清单 | — | **True（34/34）** |

即唯一净效果就是新增那 34 个名字。3 个 `research_only`
（`ts_deviation_from_mean`、`ts_jump_bipower`、`ts_lag1_autocorr`）未进入生产注册表。

### 未解决：status 分布与交接描述不符

交接文档称这 34 项中"23 个原状态为 production/implemented/extended"。实测为
**33 个 `experimental` + 1 个 `deprecated`（`ts_returns`）**，与各源模块自身的 `@register_operator`
标注一致。本轮**没有**把它们提升为 production（no-thaw），因此这 34 个名字虽然"可见"，
但在 `mode="production"` 下仍受准入限制。若确实需要提级，属于独立治理改动，需要单独证据。

## 4. R28 全算子合法样本测试

- 子代理本轮收敛结果：A（3 项未修好 fixture）、B（`ts_gap_fill_ratio` 重跑）、C（remaining26 后 6 项）
  共 **10 项，10 passed**（本人已独立复跑确认：`10 passed, 2 warnings`）。
- 仍在的失败（lower 0..899 段实测 900 例）：**796 passed / 104 failed**，
  其中 99 个 allNaN、5 个运行时/契约错误
  （`fin_cash_earnings_gap`、`holder_concentration_change`、`holder_count_change_rate`、
  `intraday_volume_clock_path_efficiency`、`intraday_volume_clock_roughness`）。**尚未收敛。**
- 文件当前 sha256：见提交。

## 5. 新发现：`test_polars_phase2_parity.py` 有 35 个**既有**失败，与本体代码无关

A/B 实测（撤下/加上第 3 节的 hook，结果完全相同）：

```
35 failed, 5 passed        # 有 hook
35 failed, 5 passed        # 无 hook
```

失败原因是 `KeyError: unknown operator canonical: 'ts_rsi'`。
进一步查证：

- `OperatorRegistry._catalog` 中**不存在** `RSI`，也不存在 `BBANDS`；`resolve_canonical_strict("RSI")` 直接 KeyError。
- `_aliases["ts_rsi"]` 是 `None`（悬空），而 `_aliases["ts_rsi_wilder"] = "RSI_WILDER"` 正常。
- 这些名字的实现在 `cleaned_operators/technical/polars_phase2_indicators.py`，
  而该模块是**未接入的原型**：注册段整段被注释掉，且全仓库**没有任何地方 import 它**。

因此这 35 个失败不是回归，而是**测试文件在测一批从未注册的 canonical**。
修复方向需要用户决策，二选一：
(a) 把这批 polars 实现按模块内注释的指引接入既有 canonical（`canonical="RSI"` 而不是 `"ts_rsi"`）——属于新功能；
(b) 认定该原型已废弃，把该测试文件按明确理由标记为 stale/skip，并修掉 `ts_rsi` 这类悬空 alias。

本轮**未**擅自选择任一路线。该结论也提示："1700+ 算子已逐个可用"这类说法需要用**当前注册面**重新核，不能沿用历史计数。

## 6. 本轮改动的文件

| 文件 | 说明 |
| --- | --- |
| `factor_engine/cleaned_operators/layer_composite_fixes.py` | ADX tr 守卫修复 |
| `factor_engine/backend/operator_semantic_version.py` | 显式声明 `"ADX": 2` |
| `factor_engine/cleaned_operators/__init__.py` | 追加 R56 缺失算子 bootstrap 模块 spec |
| `factor_engine/cleaned_operators/_missing_public_bootstrap.py` | 新增：34 个公共名的无副作用注册 |
| `factor_engine/tests/operators/test_r56_missing_public_bootstrap.py` | 新增：fresh-subprocess 契约不变验收 |
| `factor_engine/tests/operators/test_r56_adx_physical_contract.py` | 新增：ADX 物理契约与独立参考核 |
| `factor_engine/tests/operators/r28/test_all_canonicals_execute.py` | R28 合法样本收敛（+643/-72 基础上再 +98） |

未提交/未处理：11 万因子的当前代码重编译与执行清单、lower 段 104 个失败、713 个无物理说明的 polars 槽、
35 个 stale phase2 parity 用例、以及第 3 节的 status 提级问题。
