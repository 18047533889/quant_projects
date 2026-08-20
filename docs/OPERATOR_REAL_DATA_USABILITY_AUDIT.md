# 算子真实数据可用性审计报告(2026-08-08)

针对 §18 市场状态描述语言新增的 39 个算子,逐一用**真实形态的 A 股数据**
(随机游走收盘、跳空开盘、高低价包络、涨跌停、3% 停牌 NaN、对数正态成交额/
换手、30 个行业分组漂移、forward-filled 财报 + 真实更新日)在 160 股 × 260 交易日
面板上实测。报告每个算子的真实字段接线、稳态覆盖率、值域,并记录发现与修复。

结论先行:**39 个算子全部可保留,无删除项**。其中 3 处真实"数据理解偏差"已在本次修复
(update_clock 窗口、roll spread 参数名、event 算子默认窗口);1 个 P2 算子
(`ts_extremal_dependence_decay`)是**本质稀疏**的诊断量(保留,文档说明);其余算子在
正确接线下稳态覆盖率 ≥ 97%。

---

## 一、真实字段接线(输入参数 → 目录字段)

以下参数名按 `fields/catalog.py` 的真实字段给出推荐接线:

| 参数名 | 推荐真实字段 | 说明 |
|---|---|---|
| `x` / `target` / `y` / `a` / `b` | `ret` / `turnover_ratio` / `volume` / `amount` | 通用连续序列 |
| `close` / `high` / `low` | `close` / `high` / `low` | 日线 OHLC(未复权,全引擎统一) |
| `f1` `f2` `f3` | `ret`, `turnover_ratio`, `volume` | 多字段核:价格/流动性/量 |
| `condition` | `pe_ratio` / `pb_ratio` | 估值状态条件 |
| `group_id` / `group` | `industry_code` | 行业分组(role=group_key,可解析) |
| `scale`(DC) | 派生 ATR/波动代理 | `|ret|.rolling(20).mean()` 等 |
| `update_event` | **派生**:`pub_date` 出现日 | 目录无此字段,须用户构造 |
| `event` / `mark` | **派生**:涨停事件、异常收益、其幅度 | 目录无事件/标记字段,须构造 |
| `returns`(日内) | 派生分钟收益 `log(minute_close)` | 走分钟执行路径 |

**真实目录中没有的字段**:`update_event`、`event`(布尔事件)、`mark`(事件强度)。
目录仅有的布尔字段是 `is_suspend`。这 3 类输入须由用户用表达式派生(下详),
不是算子缺陷——算子本身保持通用。

---

## 二、逐算子审计结果

覆盖率 = 输出非 NaN 占比;稳态 = 后 20% 行(冷启动后)。P1=extended 表面,P2=research。

### A. Quantile-hit / extreme dependence(6 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ts_quantilogram` | P1 | x=ret | 98.0% | 100% | ✅ 可用 |
| `ts_cross_quantilogram` | P1 | target/source | 98.0% | 100% | ✅ 可用 |
| `ts_quantile_crossing_spectral_concentration` | P2 | x=ret | 97.3% | 100% | ✅ 可用 |
| `ts_extremogram` | P1 | x=ret | 98.3% | 100% | ✅ 可用 |
| `ts_cross_extremogram` | P1 | target/source | 98.3% | 100% | ✅ 可用 |
| `ts_extremal_dependence_decay` | P2 | x=ret | 23.6% | 27.3% | ⚠️ **本质稀疏**(下详) |

`ts_extremal_dependence_decay`:拟合 log-linear 指数衰减需要窗口内 ≥2 个正 excess
点。真实数据上常无正 excess(极值不衰减或数据不足)→ 大量 NaN。这是该量的定义性质
(fail-closed by design),不是 bug。**用途**:极端状态记忆长度的 regime 诊断,不当作
稠密因子使用;低覆盖率是预期行为。

### B. Expectile(2 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ts_expectile` | P1 | x=ret | 99.2% | 100% | ✅ 可用 |
| `ts_expectile_beta` | P1 | y=ret, x=vol | 98.8% | 100% | ✅ 可用(斜率量级取决于 y/x 单位) |

### C. Directional Change(4 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ts_dc_overshoot_ratio` | P1 | x=close, scale=ATR | 92.7% | 100% | ✅ 可用 |
| `ts_dc_event_rate` | P1 | 同上 | 92.7% | 100% | ✅ 可用 |
| `ts_dc_duration_asymmetry` | P1 | 同上 | 92.7% | 100% | ✅ 可用 |
| `ts_dc_overshoot_asymmetry` | P1 | 同上 | 92.7% | 100% | ✅ 可用 |

DC 是**价格**方向变化语义:必须接 `close` + 波动尺度,不能接收益率。文档已明确。

### D. 多字段协方差几何(4 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ts_feature_mode_share` | P1 | f1=ret f2=turnover f3=vol | 98.4% | 100% | ✅ 可用 |
| `ts_feature_effective_rank` | P1 | 同上 | 98.4% | 100% | ✅ 可用 |
| `ts_feature_subspace_rotation` | P1 | 同上 | 54.2% | 100% | ✅ 可用(需 recent+prior 双窗口冷启动) |
| `ts_beta_break_score` | P1 | y=ret x=vol | 54.2% | 100% | ✅ 可用(同上) |

### E. 条件依赖(2 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ts_conditional_transfer_entropy` | P2 | t=ret s=vol c=pe | 77.9% | 99.2% | ✅ 可用(bins≤3 默认窗口内可行) |
| `ts_modwt_band_corr` | P2 | x=ret y=vol | 97.6% | 100% | ✅ 可用 |

`ts_conditional_transfer_entropy`:默认 window=60、bins=3 时要求 54 个转移,日频数据
可满足;bins≥4 需加大窗口,已 fail-loud(ValueError 提示)。

### F. Spread 估算(2 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `ohlc_corwin_schultz_spread` | P1 | high/low | 99.5% | 100% | ✅ 可用(0.0055 均量级,符合 A 股) |
| `ts_roll_effective_spread` | P2 | **price=close** | 97.9% | 100% | ✅ **已修复**(原参数名 `x`) |

> **修复 A**:`ts_roll_effective_spread` 原参数名为 `x`,但语义要求**正价格序列**(内部
> log 差分)。用户若误传收益率(log(负值)→ NaN)会得到 23.5% 覆盖率的静默坏输出。
> 已将参数重命名为 `price`,使契约显式。签名、审计 defaults 同步更新。

### G. 日内(5 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `intraday_subsampled_rv_dispersion` | P1 | 分钟 returns | 100% | 100% | ✅ 可用 |
| `intraday_volatility_signature_slope` | P1 | 分钟 returns | 100% | 100% | ✅ 可用 |
| `intraday_realized_power_variation` | P2 | 分钟 returns | 100% | 100% | ✅ 可用 |
| `intraday_profile_surprise_energy` | P1 | 分钟 panel | 92.3% | 100% | ✅ 可用 |
| `intraday_profile_phase_shift` | P2 | 分钟 panel | 92.2% | 99.8% | ✅ 可用 |

**数据要求**:分钟级 panel(DatetimeIndex 分钟时间戳 × 标的列),走
`intraday_daily` 执行路径(分钟→日频聚合器已就绪,A 股 COS 分钟源已认证)。
**不是**日频面板算子——用日频数据会全部 NaN(实测 0%),这是接线错误而非算子缺陷。

### H. 局部非线性横截面(5 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `cs_knn_local_linear_residual` | P1 | target/f1/f2/f3 | 97.0% | 97.2% | ✅ 可用 |
| `cs_knn_tangent_residual` | P2 | f1/f2/f3 | 97.0% | 97.2% | ✅ 可用 |
| `cs_knn_local_gradient_norm` | P2 | target/f1/f2/f3 | 97.0% | 97.2% | ✅ 可用 |
| `cs_rank_copula_mi` | P2 | a/b | 100% | 100% | ✅ 可用(需足够横截面) |
| `cs_rank_copula_entropy` | P2 | a/b | 100% | 100% | ✅ 可用(同上) |

**数据要求**:
- `cs_knn_*` 需要每日 ≥ k 个有效样本(k≥4)。全市场/中大盘都没问题。
- `cs_rank_copula_*` 需要每日 ≥ 2·grid² 个有效股票(grid=8 时 128 只)。全 A
  (~5000)与沪深 300(300)均满足;**30 只以下的小组合会整日 NaN**(fail-closed,
  语义诚实)。设计为全市场 regime 特征(`global_state` tag),不是小组合个股 alpha。
- 性能提示:`cs_knn_*` 为 O(交易日 × 股票数² × k),全市场 5000 只规模下日频运行
  成本高,建议在中小盘/指数成分上使用或降低频次。

### I. 尾部系统性(3 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `group_tail_centrality` | P1 | x + industry | 100% | 100% | ✅ 可用 |
| `group_tail_lead_score` | P2 | x + industry | 99.6% | 100% | ✅ 可用 |
| `relation_diffusion_score` | P2 | x + industry | 100% | 100% | ✅ 可用 |

### J. Marked event(2 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `event_mark_autocorr` | P1 | event/mark(派生) | 44.4% | 99.2% | ✅ **已修复**(默认窗口) |
| `event_interval_mark_coupling` | P1 | event/mark(派生) | 44.4% | 99.2% | ✅ **已修复**(默认窗口) |

> **修复 B**:原默认 `history_window=120`,对 3% 频率事件窗口内平均仅 3.6 次、凑不齐
> 需要的 5 次 → 覆盖率 15.7%。已把默认窗口提到 **252(1 年)**,3% 事件频率下稳态覆盖
> 率 22% → 99%。事件频率更低(如涨停 1%)时仍需用户加大 `history_window`。

### K. Update clock(4 个)

| 算子 | 表面 | 接线 | 覆盖率 | 稳态 | 结论 |
|---|---|---|---|---|---|
| `update_path_efficiency` | P1 | x + update_event | — | **100%**(3 年历史) | ✅ **已修复**(窗口) |
| `update_acceleration` | P1 | 同上 | — | 100% | ✅ **已修复** |
| `update_surprise` | P1 | 同上 | — | 100% | ✅ **已修复** |
| `update_direction_persistence` | P1 | 同上 | — | 100% | ✅ **已修复** |

> **修复 C(最重要的数据理解错误)**:原内核扫描窗口硬编码为 `2×n_updates`(默认 10 天)。
> 真实财报按季度更新(~1.5% 交易日),10 天窗口永远集不齐 n_updates=5 → **4 个算子全
> NaN,覆盖率 0%**。修复:
> 1. 扫描窗口改为 **3 个交易日历年(756 天)**,让最近 n 次真实更新可达;
> 2. 内核重写为**预计算每列 update 索引 + 二分搜索**,O(rows) 而非 O(rows×window),
>    生产规模(数千股 × 全历史)可行。
> 实测:季度密度(1.6%/日)下稳态覆盖率 0% → **100%**。
>
> **使用要求**:需 ≥2~3 年历史才能累积 5 次季度更新;`update_event` 由用户构造
> (如 `pub_date` 出现日)。

---

## 三、本次修复清单

| # | 文件 | 修复 | 证据 |
|---|---|---|---|
| 1 | `cleaned_operators/update_clock.py` | 扫描窗口 `2*n` → 756 天 + O(rows) 二分内核;删死代码 `_last_updates` | 季度密度稳态覆盖率 0% → 100%;1600 天×20 股 ~1s |
| 2 | `cleaned_operators/spread_estimators.py` + `backend/operator_signatures_phase2.py` | `ts_roll_effective_spread` 参数 `x` → `price`(显式正价格契约) | 误接收益率时 23.5% → 接 close 后 97.9% |
| 3 | `cleaned_operators/marked_event.py` | `event_mark_autocorr`/`event_interval_mark_coupling` 默认窗口 120 → 252 | 3% 事件稳态覆盖率 22% → 99% |

另:`scripts/audit_new_ops_real_data.py` 为本审计的复现脚本(真实形态面板 + 逐算子
覆盖率/值域),保留在仓库供日后回归。

## 四、需要你注意的使用前提(非缺陷)

1. **`update_event` / `event` / `mark` 无目录字段**——须由表达式派生
   (`update_event` 建议 `pub_date==当日`;`event` 建议涨停/异常收益;`mark` 建议事件
   幅度),算子接受任意 DataFrame 输入,属正常用法。
2. **`cs_rank_copula_*` 需 ≥2·grid² 只股票/日**;小组合请降低 grid 或用于全市场
   regime 特征。
3. **5 个 intraday 算子走分钟路径**(`intraday_daily` 数据源),不是日频面板算子;
   面分类 extended/research(与既有 `intra_*` 一致)。
4. **`update_clock` 需 ~2~3 年历史**(对季度更新字段);冷启动段 NaN 属预期。
5. **全部 39 个算子在 extended/research 表面**(非基础 daily 表面)。按既有治理,
   扩展算子包注册在 extended 表面;`factor_production_targets()` 已含 extended+
   research,六门认证(evidence overlay)才是生产门槛,表面分类不阻塞生产目标。
6. OHLC 全引擎统一为未复权价(`adjusted=False, unverified`)。Corwin-Schultz 用
   高低比(尺度无关)不受影响;Roll 用 log 价格差分,除息跳空日会有失真——这是全引擎
   级数据注记,非本包特有。

## 五、验证与认证状态

- **42 个 market-language 测试全过**(含 golden 值、PIT、polars parity),advanced
  套件 35 个全过。
- 本报告所有覆盖率数字由 `scripts/audit_new_ops_real_data.py` 在固定随机种子
  (seed=7)的真实形态面板上产出,可复现。

**production 认证状态(2026-08-08 17:xx,需如实说明)**:

`audit_all_factor_production.py --runtime-only` 当前报告 1103 个
"PIT policy is not causal"——**覆盖所有生产目标**(含 zscore/ADX 等未改动的存量算子),
且全日志除该单一原因外**零**运行时错误。根因链:

1. 本轮修改了 3 个算子源码(`update_clock.py`/`marked_event.py`/`spread_estimators.py`)
   且并发会话新增了 18 个算子 → `evidence/factor_operator_verified.json`(16:11,1160 个
   算子)的 source hashes 与 operator-set 均失效;
2. evidence 全局失效 → `apply_evidence_certification_overlay` 对全部生产目标
   fail-closed(`production_certified=False` → `_EXPLICIT_POLICIES[canon].pit_safe=False`);
3. 审计的 PIT 门因此全红(这是 evidence 失效的级联,不是算子语义缺陷);
4. certifier 先跑审计、审计不过不写 evidence,形成"审计要 evidence 通过、evidence 要
   审计通过"的临时死锁——按既有惯例,待并发批次落定后由收尾方执行
   **factor → primitive → recipe → manifests → catalog** 的 evidence 重生成链即可恢复。

本轮的 39 个算子在运行时层面已确认干净:审计日志中它们只有 evidence 级联导致的
PIT 一行,**无**确定性/轴保持/前缀因果错误;修复后的 42 个测试全部通过。认证恢复是
evidence 重生成的既有流程,与算子代码无关。
