# QE 能力与文档同步说明（2026-09-24）

> **文档维护规则（对所有后续改动生效）**：任何代码行为变更（新指标、绑定、
> 口径、后端、API）都必须在同一轮里同步受影响的 docs/ 文档；每份被改动
> 的文档在文首加/更新「最后更新」时间戳；本文档是多轮变更的编年索引。

最后更新：2026-09-27（Asia/Shanghai）

## 文档清单与状态

| 文档 | 最后更新 | 状态 |
|---|---|---|
| docs/METRIC_REFERENCE.md | 2026-09-24 | 生成物（build_metric_reference.render()）；164 指标公式；metrics/ 改动后必须重生成 |
| docs/formulas_*.json | 2026-09-24 | 手工口径段落（生成器保留段），10 个重写 id 的口径决定在此 |
| docs/METRIC_CONVENTIONS.md | 2026-09-24 | 已补多周期与快路径口径（见下） |
| docs/PUBLIC_RUNTIME_CAPABILITIES.md | 2026-09-24 | 已补可求值指标数、probe_pnl 通道、numba/polars 后端状态 |
| docs/METRIC_CAPABILITY_MATRIX.csv | 2026-09-24 | 由 coverage_compiler 生成；绑定后状态随之更新 |
| docs/BOOTSTRAP_PERFORMANCE_20260922.md | 2026-09-22 | 历史，仍有效 |
| docs/VECTORIZED_PERFORMANCE_20260922.md | 2026-09-22 | 第一轮向量化与等价性记录 |
| docs/ROUND2_AUDIT_AND_AB_20260923.md | 2026-09-24 | 二/三轮：全指标审计、接线修复、绑定、sweep |
| docs/MULTI_HORIZON_20260924.md | 2026-09-24 | 本轮新增：多周期评估、快路径、后端（下文） |

## 2026-09-24 能力变更（MULTI_HORIZON_20260924.md 摘要）

### 多周期因子评估（rank_ic / ic / icir / ir 的 3/5/7/10/20 天等）

- `api.horizons.build_forward_return_label_bundles(daily_returns, horizons=(1,3,5,7,10,20), ...)`：
  从 (T,N) 单期收益面板构建任意 H 的复利前向收益标签
  `label_H(t) = Π_{k=1..H}(1+r(t+k)) − 1`，窗口 [t+1, t+1+H]，NaN/越界传播，
  因果（H=1 恒等 r(t+1)，无浮点往返）。与 `evaluate_horizons` 的校验契约对齐。
- `api.horizons.summarize_horizons(bundle)`：每周期 mean IC / IC std(ddof=1) /
  rank ICIR（复用 compute_icir）/ 年化 IR（ICIR×√252，等间隔 252 交易日假设）/
  **累计 IC**（nancumsum，NaN 跳过不重置）/ IC 正值比例 / 有效样本数 /
  t-stat 与 HAC t-stat（复用唯一 HAC 内核 compute_hac_tstat）。
- `api.horizons.evaluate_factor_multi_horizon(factors, daily_returns, horizons=…)`：
  一站式入口。多空曲线水下时长等多空指标按周期分别调用既有
  `evaluate(metrics=("max_underwater_duration","mean_underwater_duration",
  "time_to_recovery"), portfolio_returns=…)` 后注入
  `MultiHorizonSummary.per_horizon_portfolio_metrics`（结构化承载，不重复实现）。
- vwap_to_vwap 精确口径的 H01/H05/H10/H20 仍走 `adapters/data_access.py`
  的 DataAccess target（H03/H07 需要 DA 侧物理列，QE 不越界造数据；
  复利口径与 vwap 口径的差异已写进 docstring）。

### 10 个缺内核 id 的处置（_DECLARED_UNBOUND 26 → 1）

- 新内核 + 绑定 9 个：ic_summary（mean 口径）、ic_stability（标量适配）、
  quantile_stability（日对排序 Spearman 均值，文档口径）、hhi_concentration
  （逐日 HHI 有限日均值，口径已写入文档）、hhi_effective_n（mean(1/HHI_t)，
  倒数的均值）、return_coverage（前向收益有限占比）、turnover_adjusted_ic
  （mean IC / mean τ，τ 复用 estimate_turnover_from_ranks）、
  turnover_stability（有限期换手总体方差 ddof=0）、autocorrelation_ic
  （复用 output_type="vector" 通道，(21,F) lag 轴）。
- ic_decay：新内核 `compute_ic_decay_from_mean_ics` 在 API 层可用（配合
  summarize_horizons）；facade 自动分发需未来"多标签请求"契约，未造假通路，
  是 `_DECLARED_UNBOUND` 仅存 id。

### 常用指标性能（rank_ic / pearson_ic / ic_ir 上游）

- 默认 exact 路径**位级不变**，压榨后 L 档（T=1250,N=300）：
  spearman 81.7→51.1ms（1.6×）、pearson 55.0→31.0ms（1.8×）、
  F=10 面板 1.7-1.9×；端到端 rank_ic+pearson_ic+ic_ir：152→95ms。
- opt-in 快路径：`compute_daily_ic(..., backend="numba")`，JIT 单遍内核，
  实测与 exact 最大偏差 3.33e-16（ulp 级；**证据哈希与 exact 不同**，已
  写进 docstring 与测试），NaN 位置完全一致，L 档 spearman 51→19ms、
  F=10 441→57ms（10×+）。JIT 冷编译 ~2-4s（磁盘缓存），默认路径不受影响。
- polars 后端审计：`polars_ic_batch` 等为**未接线代码**（selector 无消费方）；
  已补 4 个 parity 测试（家规容差，实测 ≤2.2e-16），唯一语义分歧（<min_obs
  行的 valid_counts 置 0 vs 如实计数）已钉进测试。IC 族暂不切 polars 的
  理由（长表物化 O(TNF) + 位级差异）已记录。
- 删除死代码 `metrics/quantile_optimized.py`（全仓库仅 metamorphic 测试
  引用，已清理；备份在 /home/sunhaiwei/qe_opt/oc_work/backup/）。

### 测试总量

QE 默认套件测试数（2026-09-24）：**约 2470+**（本轮新增 22+48+23=93；
四轮累计新增约 950）。全量回执见 /tmp/wb_r4.txt（本轮定版运行）。

## 历史轮次索引

- 2026-09-22：IC/quantile/回测族向量化 + 位级等价套件（VECTORIZED_PERFORMANCE_20260922.md）
- 2026-09-23：全指标审计、11 个 probe_pnl 指标接线修复、AB 契约套件（ROUND2_AUDIT_AND_AB_20260923.md）
- 2026-09-24：16 个编目绑定、12 个 fixture 打通、剩余模块 sweep、多周期、缺内核、快路径（本文档）

---

## 2026-09-25 更新：polars 后端接线（IC 族第三条 opt-in 路径）

最后更新：2026-09-25（Asia/Hong_Kong）

上一轮审计结论"polars 未接线"已处置完毕，AB 实测后按用户规则接入：

1. **修复 `polars_ic_batch` 实现硬伤**：分组键里的逐行字符串 `factor_id`
   列（Python 列表推导 O(T*N*F) 次字符串操作，纯冗余——`factor_idx` 已是
   同一分组键）已移除；spearman 的 rank+corr 合并为同一 lazy plan。
2. **AB 实测**（T=1250, N=300，单线程，3 次取最优，含正确性对拍
   max|Δ|≤5.6e-17、NaN 位置一致）：

   | 形状 | exact（默认，位级不变） | polars（修复后） | numba（opt-in） |
   |---|---|---|---|
   | F=1 | 47.7ms | **28.7ms**（1.7×） | 12.0ms |
   | F=10 | 426.7ms | **208.6ms**（2.0×） | 43.5ms |
   | F=50 | 2209.8ms | **1496.4ms**（1.5×） | 142.9ms |

3. **接线**：`compute_daily_ic(..., backend="polars")`——修复后的 polars
   在所有形状上快于 exact（1.5–2.0×），故按"更快就一定要用上"接入为
   opt-in 路径；但 **numba 在所有形状上仍快 polars 2.4–10×**，因此默认
   路径仍是 exact（位级证据哈希契约），纯速度选 numba，polars 管线选
   polars。三条路径的 NaN 位置完全一致。
4. **已声明的语义分歧**：polars 路径对低于 `min_obs` 的组把
   `valid_counts` 记 0（组被 plan 过滤）；facade 边界把不可求值行的
   counts 恢复为 exact 口径，调用方可见的 eligible 行计数一致。已钉进
   tests/metrics/test_polars_backend_wiring.py（8 个新测试：对拍 exact
   与逐日 oracle、确定性、常数截面 NaN、未知 backend 拒绝）。
5. 结论表：**默认 exact（哈希契约）→ 提速选 numba → polars 管线选
   polars**。selector 未改动（IC 族不建议自动路由，理由不变：长表物化
   O(TNF) 与位级差异）。

---

## 2026-09-25 追加：polars 纯原生化重写与最终 AB 结论

最后更新：2026-09-25（Asia/Hong_Kong）

针对"必须纯正原生 polars、不能有逐行 Python/pandas"的要求，做了两种
纯 polars 公式的原型 AB（均在 T=1250/N=300 实测，数值对拍 ≤5.6e-17）：

1. **宽表 + 水平表达式**（零 long 表，按列构建 N 项水平求和表达式）：
   pearson 183ms —— polars 水平聚合非 SIMD 强项，远差于 exact，弃用。
2. **long 表 + 分组求和聚合**（Σv²/Σy²/Σvy 一次 groupby + 算术后处理，
   替代每组 pl.corr；并用原生 `mean().over()` 窗口居中消除原始矩公式的
   偏移灾难性抵消；偏移压力实测与数据固有 float64 行为一致）：
   F=1 pearson 原型 11.4ms（vs pl.corr 版 28.7ms，2.5×）。
   已将该公式重写进 `polars_ic_batch`（动态分组键：F==1 退化为单键；
   组合长表摊薄多因子构建成本）。

最终 AB（重写后，`polars_ic_batch` 端到端）：

| 形状 | exact | polars（纯原生重写） | numba |
|---|---|---|---|
| F=1 | 46.7ms | 28.2ms | **8.0ms** |
| F=10 | 433.4ms | 205.9ms | **38.4ms** |
| F=50 | 2165.8ms | 1262.0ms | **122.8ms** |

**结论**：纯原生 polars 已压到极限（比首版快 4-5×），但对这种稠密 (T,N)
面板逐日归约，**numba 仍快 2.3-3.5×**——连续内存布局 + prange 覆盖全
(T,F) 网格是算法与布局的正配，polars 的长表物化与 hash groupby 是结构性
开销，无法消除。接线状态不变：backend="polars" 可用（比 exact 快），
默认 exact（位级哈希契约），纯速度选 numba。所有相关测试 48 个全绿。

## 2026-09-27：注册 ic_decay 的显式多期限入口

`evaluate_horizons(factors, labels, as_of=..., ic_method="pearson")`
接收至少两份已明确标注 horizon、时间轴及结束时点的 `LabelBundle`。
它按 `as_of` 过滤未成熟标签，并根据 `sample_policy` 选择共同样本
或各期限样本。随后调用 `compute_ic_decay(bundle)`，得到只读
`ICDecayResult.values`，形状为 (期限, 因子)，每格是该期限有限逐日
Pearson IC 的均值；无有效日时为 NaN。

结果绑定有序 horizon、factor ID、标签内容哈希、as_of、样本策略和
成熟/样本计数。默认 `evaluate_horizons` 仍使用 Spearman，不能作为
注册 `ic_decay` 的 Pearson 输入；`compute_ic_decay` 会明确拒绝。
统一 `evaluate(metrics=("ic_decay",))` 仍只有一个标签输入，
因此继续拒绝自动分发，不能据此声称该 ID 已接入单标签 facade。
