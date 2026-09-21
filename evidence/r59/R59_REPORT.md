# R59 — 落值性能根治 + 186 算子晋升 daily surface（2026-09-21/23）

## 一、落值慢的根因与修复（polars_expr_emitter.py）

根因：18 个 `rolling_map` Python UDF 每行每窗口每标的调一次 Python，单个因子落值 95–585s。
修复：全部下沉为
1. **polars 原生**：`ts_quantile`→`rolling_quantile(linear)`、`ts_skew`→`rolling_skew(bias=False)`（加 `_sd>0` 零方差 gate 对齐 pandas）；
2. **块帧向量化 numpy**（每算子每标的一次调用）：mad/product(frexp)/argmax/argmin/time_slope/monotonicity/turning_point_ratio/endpoint_deviation/vol_shift/drawdown_duration/recovery_fraction/kurt/quantile_range/median_abs_deviation/mean_abs_deviation/under_water/running_peak；
3. **corr/cov 块帧化**：`_rolling_pair_block_frame` + `_np_pair_corr`/`_np_pair_cov`，替换 4 处 list-frame 派发。

### 落值耗时（25×500 真实面板，单因子，auto_long）

| 算子 | 旧 | 新 | 加速 |
|---|---|---|---|
| ts_quantile | 585.5s | 2.40s | 244× |
| ts_skew | 161.7s | 2.62s | 62× |
| ts_median_abs_deviation | 111.0s | 5.20s | 21× |
| ts_corr | 51s | 4.27s | 12× |
| ts_cov | 51s | 4.57s | 11× |
| ts_mean_abs_deviation | 38.7s | 2.69s | 14× |
| ts_kurt | 46.9s | 6.72s | 7× |
| ts_product | 43.6s | 3.49s | 12.5× |
| 地板（ts_mean/ts_std/ts_zscore） | — | 2.5–9.5s | 数据加载上限，未变 |

## 二、开发中抓出并修正的语义 bug（对照 pandas 权威）

1. **ts_product 符号错误**（历史 bug）：旧 UDF 用 `sum(valid)<0` 判号，重写为 frexp 尾数/指数乘法对齐 `_stable_product`，符号精确。
2. **monotonicity C/D 反号**；frexp 指数 off-by-one；skew 零方差 gate；drawdown last-break 公式；recovery 峰位行坐标；vol_shift 物理窗 `n_phys=min(i+1,w)`。

## 三、ts_vol_shift_score 最终仲裁

kernel_ab 中 old UDF 报 19/720 NaN 不一致 → 与 `regression_models._trailing_run_halves` 权威逐行对拍：**新 kernel 完全一致（worst=3.89e-16, nan_mismatch=0, 960 cases）**；不一致全部来自旧 UDF 在预热边缘行（n<w）midpoint 语义错误。

## 四、daily surface 晋升（operator_surface.py）

- parity 波 v1=123 / v2=44 / v4=30 / v5=16 → **166 个**真 parity PASS；
- pandas-only 执行证据 **20 个**（polars 后端不支持、auto 路由回 pandas）；
- 合计 **186 个**晋升 daily（`_R59_PROMOTIONS`，注册于 R58 block 之后）。
- **排除 30 个 RESEARCH_ONLY 算子**：静态 research 面与 daily 分区重叠会触发 `finalize_layer_governance` 重叠检查（首次插入时被抓出并回滚修正）；12 个 unsafe/internal/legacy 维持排除。
- triage 最终：A=186 / B=381 / C=12（`evidence/r59/surface_triage_r59.json`）。

## 五、dsl_parser.py 解析期 gate

新增 `_missing_required_params` + parse 期缺参拒绝（镜像运行时 binder），`ts_abdi_ranaldo_spread(close,20)` 现在解析期即拒绝。

## 六、回归结论

- kernel A/B：18 kernel 全部通过（skew-const/vol_shift FAIL 均为 harness/旧 UDF 历史行为，权威对拍通过）；
- 端到端 smoke：12 算子真实落值全部干净完成（ts_product 无 NaN→u64 crash，3.49s）；
- 治理检查：`load_all()` + `finalize_layer_governance` 通过，migrated surface = 1429；
- 真实数据 corr/cov 端到端 = 引擎落值与 pandas rolling 对照一致（本地对拍脚本 `/tmp/r59_pair_e2e.py`）。

## 证据文件

- `evidence/r59/parity/<canonical>.json`（186 份逐算子证据）
- `evidence/r59/R59_PROMOTION_MERGE.json`
- `evidence/r59/surface_triage_r59.json`
- `evidence/r59/R59_CONSOLIDATED.patch`（emitter ×6 + parser ×2 + surface）
- 基准：`/home/sunhaiwei/_r84b_new_kernel.json`、重测日志 `/tmp/r84c_retime.log`

按规不 commit/push；R58 改动保持原样。
