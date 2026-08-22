# FactorEngine 第四批整改报告(2026-08-08)

针对外部 AI 评审提出的第四批问题清单,已在本机代码库全部整改。
**未删除任何算子**;所有被指出的"数学错误 / 缺失语义错误 / 搜索空间污染 /
命名不一致"均以修改实现或新增治理的方式修复,个别深层基础设施项已文档化。

---

## 一、P0 修复(14 项全部完成)

| 编号 | 问题 | 修复 | 文件 |
|---|---|---|---|
| P0-01 | Huber IRLS 权重被平方(`wdesign=X*w` 最小化 Σw²e²) | 改 `sqrt(w)` 加权(与 `daily_panel._ols_residual` 同约定) | `regression_models.py` |
| P0-02 | 旧 `period_lag` 按首次出现顺序 | 改为 fiscal-ordinal 有序插入,滞后用**精确财政期序**(缺失期 → NaN),与 `transforms_v2` 一致 | `common/daily_panel.py` |
| P0-03 | `_values()` 跨过缺失财政期取更老报告 | 新增 `require_consecutive=True`,`fin_ttm`/`fin_average_balance` 遇财政期 gap → NaN | `fundamental/transforms_v2.py` |
| P0-04 | 财务 AR(1) cov(ddof=1)/var(ddof=0) 放大 n/(n-1) | 统一总体协方差/方差(同一 ddof) | `fundamental/quality_v2.py` `_ar1_slope` |
| P0-05 | 递归 EMA/Wilder(DMI/DX/NATR/PPO/PVO/Keltner/TSI/DEMA/TEMA/Chaikin/ForceIndex)未标 full_replay | 全部标记 `stateful`+`full_replay`(pandas/polars 双后端),并入 `FULL_HISTORY_REPLAY_CANONICALS`(分段执行 fail-closed) | `technical/indicators_v2.py`、`polars_tech_misc.py`、`price_volume/liquidity_v2.py`、`production_hardening.py` |
| P0-06 | 整数参数名白名单太窄,`fast_window=5.9` 静默截断成 5 | 白名单扩展(全部窗口/期数/滞后/结构参数)+ `OperatorMetadata.param_types` 成为权威来源 | `base.py` |
| P0-07 | `ts_new_high/low` warmup/缺失输出 0 | 当前值或基线缺失 → NaN(pandas+polars 双后端) | `price_volume/technical_extensions.py`、`technical/polars_misc_v2.py` |
| P0-08 | `candlestick_pattern` warmup 输出 0 | 每个 pattern 计算 `required_valid_mask`,历史/基线不足 → NaN | `price_volume/candle_pattern_engine_v2.py`(+repairs) |
| P0-09 | `3_inside/3_outside` bar 索引错位(多读 t-3) | 重写为标准 bar1=t-2 / bar2=t-1 / confirmation=t,不再复用移位后的多 lag mask | 同上 |
| P0-10 | candlestick 死参数污染搜索空间 | 新增 `candlestick_active_params(pattern)` + 每个 pattern 的 `_CANDLE_PARAM_DEPS`;`penetration` 仅 `abandoned_baby`/`evening_doji_star` 生效 | 同上 |
| P0-11 | 排列转移熵 `delay>1` 拒绝相邻 embedding → 全 NaN | 相邻条件 `start_next==start+1`;delay={1,2,3} 均有有效输出 | `sequence_complexity.py` |
| P0-12 | Haar 非 2 幂长度保留最旧数据、丢弃最新 | 改 `finite[-m:]`(保留最接近 t 的数据) | `ts_model/wavelet_spectral.py` |
| P0-13 | 小波熵遇 0 能量级 `0·log0` → NaN | 过滤 `w>0` 后再求和(归一化用实际存在级数) | 同上 |
| P0-14 | systemic-tail 冷启动把首条观测机械判为 extreme | `_extreme_indicator_panel` 增加 `min_periods`(默认 10)门;两个算子暴露并校验 | `tail_systemic.py` |

## 二、缺失语义统一(Batch-2,全部完成)

| 算子 | 修复 |
|---|---|
| `volume_weighted_return` / `rolling_vwap` | numerator/denominator **cohort 一致**:任一侧缺失即两侧同缺,不再向 0 稀释 |
| `bounded_nvi` / `bounded_pvi` | 缺失 volume/收益 bar → NaN,不再被 `.where(cond,0)` 当成"无量日"注入假零收益 |
| `zero_return_ratio` | 缺失收益 → NaN(不计数),不再算作"非零收益" |
| `CMF` | 一字板/零 range bar numerator/denominator 一致排除,不再稀释 |
| `MFI` | 缺失 bar → NaN(不当作零资金流),不再把 MFI 压向 50 |
| `DMI_plus/minus` | 缺失 bar → NaN,不再被 `.where(...,0)` 变成"有效零方向运动"并进入 Wilder 平滑 |
| `cs_tail_retention` | **P1-22**:今日缺失 ≠ 退出尾部——denominator 改为"前尾 ∩ 今日可观测";**P1-23**:`side` 严格 `{top,bottom}` 枚举 |

## 三、搜索空间清理(Batch-3,全部完成)

| 编号 | 修复 |
|---|---|
| P1-40 | `fin_core_earnings_ratio.scale` 死参数**删除**(签名/param_names 同步),denominator 固定为 revenue |
| P1-21 | copula `grid` 限制为已验证集合 `{4,8,16}`;熵归一化 `H/log(grid²)`;MI/熵加 **Miller-Madow 有限样本校正**;`N >> grid²` 强制 |
| P0-06 | 整数浮点截断(见上) |
| P0-10 | candlestick 条件参数(见上) |

## 四、P1 修复(Batch-4)

| 编号 | 修复 | 文件 |
|---|---|---|
| P1-12 | `candle_inside_ratio` 多态编码:真正 containment 才取值,上移/下移/gap bar → NaN | `price_volume/candle_geometry_v2.py` |
| P1-13 | `candle_gap_atr` 的 ATR 改 **Wilder 定义**(与全库统一) | 同上 |
| P1-14 | 排列熵系对 ties 用 stable 排序制造人工序 → 窗口 tie 比例 >0.5 时 fail-closed(NaN) | `sequence_complexity.py` |
| P1-15 | wavelet/FFT 不再压缩缺失点(时间轴保持),改用**最长连续有限段** | `ts_model/wavelet_spectral.py` |
| P1-16 | 小波斜率横轴改**真实 dyadic scale**(j·log2) | 同上 |
| P1-17 | 小波斜率 cov/var 统一 ddof(总体统计) | 同上 |
| P1-18 | Corwin-Schultz `H==L`(一字板)按 one-price 状态走 alpha≤0→0 分支,仅 `H<L` 视为数据错误 | `spread_estimators.py` |
| P1-19 | spread 平滑要求 `min_periods == smooth_window`(一个样本不再提前开均值) | 同上 |
| P1-25 | `group_tail_lead_score` 文档基线改为实际计算的 `P(E_peer,d+lag)` | `tail_systemic.py` |
| P1-26 | drawdown 路径中非正值 → **break**(不删除重连) | `stateful/drawdown_path.py` |
| P1-27 | `ts_current_drawdown_area` 改**当前未收复回撤**:从最后一个新高起累计,完全收复后=0 | 同上 |
| P1-28 | `state_latch.initial_state` 严格 `{0,1}`,其余值拒绝 | `stateful/rule_language.py` |
| P1-29 | `state_hold` update 触发但 payload=NaN → **清空 hold**(不沿用旧值) | 同上 |
| P1-38 | `register_catalog_only` 禁止覆盖已有 runtime 的 canonical,除非显式 `merge_existing=True`(保留 backends/param_names) | `registry.py` |
| P1-39 | `allow_panel_broadcast` 不再完全跳过对齐校验——仍强制 **columns 一致**(禁止跨标的广播) | `base.py` |
| P1-41 | 新增显式别名 `fin_mean_abs_deviation`;`fin_mad` 描述注明是 mean absolute deviation(非中位数 MAD) | `fundamental/transforms_v2.py` |

## 五、文档化延后项(深层基础设施,已治理但需后续)

| 编号 | 项 | 说明 |
|---|---|---|
| P1-11 | 全局 OHLC invariant | 建议建立 `OHLCValidated` 类型供 candle/pattern/volatility 共用,现各算子自行 fail-closed;属跨模块统一重构 |
| P1-20 | Haar/MODWT 边界 zero-padding | 需选择 filter-state / reflection 边界并全 backend 同定义;当前实现更接近 "Haar multiscale detail correlation",命名已在 docstring 说明 |
| P1-24 | GroupIdNormalizer | 跨 `group_*`/`relation_*`/rotation/peer 统一缺失 group id 语义,属多模块重构 |
| P1-37 | CanonicalSignature | registry 对第二 backend 的 logical signature 校验(arity/类型/scalar-panel),属架构层 |
| P0-05(checkpoint) | ewm 族 checkpoint-restore | 治理已 fail-closed(full-replay);真正的状态序列化(EMA accumulator 等)是 runtime 层后续任务,与现有 `SEGMENTED_EXECUTION_CANONICALS` 机制对齐 |

## 六、验证

- 所有改动文件 `py_compile` 干净,`load_all()` 正常。
- 直接功能验证(本机复现):
  - 排列转移熵 delay={1,2,3} 全有输出;
  - 小波熵对 0 能量级有限、斜率 dyadic 横轴、常数序列 0 能量 fail-closed;
  - `ts_new_high/low` warmup/缺失 → NaN(非 0);
  - `candlestick_pattern` 3_inside/3_outside 人工构造 golden 形态正确触发、warmup NaN、`candlestick_active_params` 正确;
  - `fast_window=5.9` 被拒、`5.0` 接受;
  - 尾部冷启动 min_periods 门、rotation side 严格枚举、state_latch/hold 语义、drawdown 非正值 break、copula grid 校验均正确。
- 测试套件:`test_advanced_ops_2026.py`、`test_daily_panel_ops.py`、`test_deepening_2026_08.py`、`test_financial_next_stage.py`、`test_dynamics_pack_2026_08.py`、`test_deepening_semantics.py`、`test_ashare_typed_ops.py`、`test_final_pack_2026_08.py`(61)、`test_alpha_language_2026_08.py`(9/10)等全绿或仅余证据级联项。

## 七、当前认证状态(如实说明)

`pit_safe=False` 的审计失败是**证据过期级联**(并发会话仍在编辑同一棵树;本轮修改了 ~20 个算子源码
改变了 evidence source hashes,且并发会话新增算子改变了 operator-set),覆盖所有生产目标(含未改动的
`ts_zscore`/`ts_quantile` 等)。**非本轮代码引入**:本批改动的算子运行时零错误,且 `test_alpha_language`/
`test_final_pack` 的 pit_safe 断言对 46 个未触碰算子同样全红。恢复路径:待并发批次落定后重跑
**factor → primitive → recipe → manifests → catalog** evidence 链(按既有协调约定,不在并发编辑期间执行)。
