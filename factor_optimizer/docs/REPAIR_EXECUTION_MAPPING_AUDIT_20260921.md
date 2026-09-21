# Repair family → 执行映射审计（2026-09-21）

## 结论

注册表声明了 20 个 repair family；声明、条件搜索和 supervised 参数冻结都不等于处理已执行。逐项检查 FO 全部 adapters，并交叉检索 FE/FP 原语与测试后，FO 当前只有 `CAUSAL_SMOOTHING`、`DECAY_REFINEMENT` 两个 family 有版本化的公开编译执行桥；其余不少 family 已有底层原语，但尚未由 FO 自动接线。

| family | owner | actual public path | status |
|---|---:|---|---|
| `CAUSAL_SMOOTHING` | FP | `compile_smoothing_repair` → FP registry | executable, research-only |
| `DECAY_REFINEMENT` | FP | `compile_smoothing_repair` → FP registry | executable, research-only |
| `NO_OP_RAW`, `ABANDON` | FP / FE | control decision, no numeric transform expected | executable orchestration not assessed here |
| remaining 16 families | FE or FP | no public FO family compiler found | proposal/schema only |

The 16 numeric/structural treatment declarations without a public family compiler are `SIGN_ORIENTATION`, `WINDOW_REFINEMENT`, `ROBUST_OUTLIER`, `MISSINGNESS_FRESHNESS`, `U_SHAPE_REPAIR`, `INVERTED_U_REPAIR`, `TAIL_SATURATION`, `TAIL_HINGE`, `INDUSTRY_NEUTRALIZATION`, `SIZE_NEUTRALIZATION`, `STYLE_NEUTRALIZATION`, `REPRESENTATION_RANK`, `REPRESENTATION_ZSCORE`, `ROBUST_SCALE`, `OPERATOR_SWAP`, and `LOW_DOF_INTERACTION`.

## 20 family 逐项矩阵

| family | 已有 FE/FP 原语或控制语义 | TRAIN 状态/外部输入 | 现有数据测试证据 | FO 自动接线缺口 |
|---|---|---|---|---|
| `NO_OP_RAW` | 保留原始值的控制决策 | 无 | `test_raw_outcomes.py` | 非数值 transform；需由 runner 明确保留 treatment identity |
| `SIGN_ORIENTATION` | FE `neg`/`multiply(-1)` | 方向必须仅由 TRAIN 冻结 | `test_diagnosis_repair_policy.py` | 无 family→FE expression 编译器 |
| `WINDOW_REFINEMENT` | FE 大量 `ts_*` window 参数原语 | 原表达式、可变 window 位点、TRAIN natural scale | `test_repair_registry.py`、条件搜索测试 | 只有 schema/search，无 AST 定位与重编译 |
| `DECAY_REFINEMENT` | FP `ewma`/`one_sided_iir_lowpass` | TRAIN natural scale、training ref | 本轮审计测试、`test_preprocess_repair_adapter.py` | **已接研究桥**；部分 FO 合法候选会被 FP 参数域拒绝 |
| `CAUSAL_SMOOTHING` | FP SMA/EWMA/IIR/KAMA/Kalman | TRAIN natural scale、历史 warmup | 本轮审计测试、FP `test_executable_smoothing.py`/KAMA oracle | **已接研究桥**；`allow_research=True`，非生产 admission |
| `ROBUST_OUTLIER` | FP `cs_winsor`、`robust_ewma`；FE `winsorize` | 截面 universe；分位点若 fitted 则仅 TRAIN | FP `test_fe_operator_parity.py`、`test_dlib_standards.py` | 无 quantile schema→具体 kernel/fit policy 编译器 |
| `MISSINGNESS_FRESHNESS` | FP `forward_fill`、`missing_indicator`、`freshness_aware_fill`、freshness transforms | PIT 时间、max lag/freshness window；可能多输出 channel | FP `test_dlib_standards.py`、eligibility 测试 | `fill/drop/flag` 尚未映射为明确 recipe 与输出选择 |
| `U_SHAPE_REPAIR` | FE `abs`/`power`/`signed_power` 可组合 | center 必须 TRAIN 冻结；asymmetry 需明确定义 | `test_supervised_parameter_v3.py`、grid/preflight 测试 | 只有 center/power 搜索，无版本化表达式公式/执行器 |
| `INVERTED_U_REPAIR` | 同上，可对 U 形结果取负 | center 必须 TRAIN 冻结 | `test_repair_execution_boundaries.py` | 同上；倒 U 的符号与 asymmetry 公式未接线 |
| `TAIL_SATURATION` | FE `clip`/`saturate`；FP `cs_winsor` 可提供相近但非同义原语 | tail cutoff/quantile 应在 TRAIN 拟合；需截面 universe | supervised parameter 测试、FE/FP parity | 无 `top/bottom/both` 到阈值拟合及 apply 的 compiler |
| `TAIL_HINGE` | FE `maximum`/`minimum`/`where` 可组合 hinge | hinge value TRAIN 冻结 | supervised grid/preflight 测试 | 只有参数冻结，无 family expression compiler |
| `INDUSTRY_NEUTRALIZATION` | FP `ols_neutralize`（可路由 FE `cs_neutralize`） | PIT industry exposure；weighted 还需权重；逐日足够样本 | FP `test_fe_operator_parity.py`、`test_dlib_standards.py` | 无 exposure_set 解析/装载及 weighted_ols 映射 |
| `SIZE_NEUTRALIZATION` | 同一 OLS 原语；FE 另有 `size_neutralize`/`industry_size_neutralize` | PIT size，双重版本还需 industry | 同上及 FE backend parity 集 | 无 exposure binding 与 family compiler |
| `STYLE_NEUTRALIZATION` | FP `ols_neutralize` 可接受多暴露矩阵 | beta/liquidity/volatility/style_multi PIT exposures | FP fit/apply、parity 测试 | 无 exposure catalog、列顺序/缺失策略及 weighted 映射 |
| `REPRESENTATION_RANK` | FP `cs_rank`（FE `rank` parity）；FE `ts_rank` | CS 需当日 universe；TS 需 window（当前 schema 未给） | FP leakage/parity/representation tests | 无 axis 分流；`tie_method=min` 与已认证 average 需契约 |
| `REPRESENTATION_ZSCORE` | FP `cs_zscore`（FP-native）；FE `zscore`/`ts_zscore` | CS universe 或 TS window；cap 后处理 | FP parity、preset semantic order tests | 无 axis/window/cap 编译器；TS schema 不完整 |
| `ROBUST_SCALE` | FP `cs_scale`；FE robust stats/`cs_mad_zscore` 等 | 截面 universe；中心/尺度定义与常量策略 | FP transform/registry tests | `mad/iqr/std × median/mean` 未映射到一个认证执行契约 |
| `OPERATOR_SWAP` | FE operator registry 与 DSL 原语丰富 | 原 AST、类型/参数/时钟语义、合法替代集合 | FE operator suites；FO mutation legality tests | 只有 target family/aggressiveness，未生成可执行替换 AST |
| `LOW_DOF_INTERACTION` | FE `multiply`/低阶组合可表达 | 两个已冻结输入、DOE/选择仅 TRAIN | FO conditional/search tests | 未定义输入选择、交互公式与复杂度约束编译 |
| `ABANDON` | 控制决策，不应执行 transform | 诊断与审计理由 | repair policy/raw outcomes tests | 非原语缺失；runner 需记录停止与 lineage |

特别地，U/倒 U、tail、rank/zscore 和 neutralization 都不能写成“没有原语”；准确状态是“原语存在，FO family 到原语、拟合状态和外部输入的自动入口未接”。

## 两个已接 family 的实测映射

| method | FP transform | mapping |
|---|---|---|
| SMA | `trailing_sma` | `window=ceil(h)`, full-window warmup |
| EWMA | `ewma` | `halflife=h`, `min_periods=1` |
| IIR | `one_sided_iir_lowpass` | `alpha=1-exp(-ln(2)/h)` |
| KAMA | `kama` | ER=`ceil(h)`, fast=2, slow=`ceil(3h)`, current excluded |
| Kalman | `kalman_local_level` | measurement noise 1; process noise `alpha²/(1-alpha)` |

其中 `h = TRAIN natural_time_scale × natural_time_scale_relative`。五条路径均在交错排列的双资产小面板真实执行：使用独立手算数值、常量第二资产、单独执行对照、未来尾部突变和缺失值，分别检查递推、排除当前 bar、前缀因果、资产隔离、缺失和常量行为。EWMA 依 pandas EWM 语义跳过缺失观测；其余四条在缺失值滞后一格时输出 NaN，本轮测试明确区分。

For `DECAY_REFINEMENT`, `half_life_relative=False` treats `decay` as previous-state retention and maps to IIR `alpha=1-decay`; `True` scales the TRAIN natural time scale and maps to EWMA half-life. Absolute zero decay implies unit gain, which the FP registry rejects before evaluation as `IneligibleSmoothingRepair`; there is no clamp or raw fallback.

The FO conditional domains are broader than FP's admitted kernel domains. A
syntactically valid FO proposal can therefore be ineligible at compile time
(observed examples: SMA/EWMA horizons below 3, KAMA ER below 5 or slow period
below 20, and Kalman process noise above 0.1). The compiler fails closed; such
candidates are not executable successes and must not be scored as raw fallback.

## 边界

Execution requires `allow_research=True`; it is not production admission or publication. This audit did not batch-recompute factors, deploy, or write production data, and does not claim any candidate improves performance.

Evidence: `factor_optimizer/tests/test_repair_execution_mapping_audit.py`.
