# riskfolio_qs — 组合优化与风控工具包（cvxpy）

基于 cvxpy 的配置驱动组合优化工具包：Barra 预计算 / 历史协方差两条风险路径、
主动风险硬约束、行业/风格中性、逐资产成本模型、PIT 契约与完整审计落盘。

**定位:** 企业级 A 股横截面日频多因子量化项目的**组合构建层** —— 上游模型/因子出 alpha，
本库把它变成满足风控约束的目标仓位 `target_positions.parquet`，交给 vectorbt_qs 回测执行。

**版本:** 0.3.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/riskfolio_qs (私有)
**文档:** [`docs/README.md`](docs/README.md) ｜ 数学规范: `docs/riskfolio_qs_v0.3_功能与数学规范.md` ｜ CLI: `docs/CLI使用指南.md`

---

## 它是什么 / 不是什么

**做什么：** 读 alpha + 风险模型 + 基准 + 上期持仓 → cvxpy 求解（因子形式 Σ=BFB'+D 或历史协方差）→
施加 TE/换手/行业/风格/成本约束 → 输出目标仓位、逐笔交易、运行审计四件套。
`alpha` 输入接受 parquet/CSV（long/wide），其余生产数据一律走 data_access。

**不做什么：** 不产 alpha（上游因子/模型做）、不做成交模拟（vectorbt_qs 做）、
不改参数黑箱寻优（factor_optimizer 的组合链专属）。

## 优化器全景（8 个路由名）

| 优化器 | 场景 | 后端实现 |
|---|---|---|
| `meanvar_enhance_barra_precomputed` | 预计算 Barra 指增（**主力**） | cvxpy 因子形式，Σ = B F B' + D（因子协方差 + 特异方差） |
| `minvar_enhance_barra_precomputed` | 指增（保守版，最小方差目标） | 同上 |
| `meanvar_enhance_hist` | 历史协方差指增 | cvxpy 历史样本协方差 |
| `minvar_enhance_hist` | 历史协方差最小方差 | 同上 |
| `meanvar_absolute_return` | 绝对收益（无基准） | 经典 mean-variance |
| `absolute_return` | 绝对收益别名 | 路由到 meanvar_absolute_return |
| `topn_long_only_equal_weight` | **兜底**（求解失败/无风险模型时） | topN 等权规则，不依赖 cvxpy |
| 旧名 `*_enhance_index` | 兼容旧配置 | CLI 自动路由到 precomputed |

另有 draft 阶段的 long-short 组合（未启用生产）。

## 安装与运行

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/riskfolio_qs.git
cd riskfolio_qs
pip install -r requirements.txt
pip install -e ".[dev]"
```

```bash
# CLI（单一回测链路入口）
riskfolio-qs optimize --config examples/cli/barra_precomputed.yaml \
    --alpha /data/alpha.parquet --output-dir out
riskfolio-qs analyze-positions --positions out/target_positions.parquet
```

```python
# Python API
from riskfolio_qs.adapters import BarraPrecomputedAdapter
from riskfolio_qs.runners.pipeline import OptimizationPipeline

bundle = BarraPrecomputedAdapter(
    risk_root="../v1_sbi_fullA",   # exposure/factor_cov/specific_var 三件套
    alpha_df=alpha, benchmark_df=benchmark,
    prev_positions_df=prev, tradable_df=tradable,
).build_bundle()
output = OptimizationPipeline().run(bundle, scenario="index_enhancement")
```

## 输出（每次运行 8 件套）

| 文件 | 内容 |
|---|---|
| `target_positions.parquet` | 目标权重矩阵（下游 vectorbt_qs 的唯一输入） |
| `trades.parquet` | 目标持仓 − 上期持仓的逐笔调仓 |
| `summary.parquet` | 组合汇总统计 |
| `metadata.json` | PIT 信息（决策/执行时点、as-of 风险快照、暴露滞后） |
| `run_manifest.yaml` | 运行清单 |
| `resolved_mapping.yaml` | 解析后的最终字段映射（审计） |
| `resolved_params.yaml` | 解析后的最终参数（审计） |
| 验证报告 | 求解结果 vs 约束的最大误差（实测 2.77e-6） |

## 关键特性全清单

### 四层配置优先级 + 白名单
`global → risk → solver → optimizer` 逐层覆盖，**白名单 26 个参数**之外的名字直接拒绝
（配置写错名不会静默忽略），值域校验在加载时完成。优化器映射与参数定义
（`src/riskfolio_qs/configs/optimizer/mapping.v0.2.0.yaml`、`parameters.v0.2.0.yaml`）
**冻结治理、随 wheel 分发**。

### 风控约束（指增场景硬约束）
- **主动风险**：年化 tracking-error 上限默认 **3%**（cvxpy 二阶锥）
- **换手**：单次调仓换手率 ≤ 20%（普通）/ 25%（豁免场景）
- **行业中性**：strict（主动暴露=0）或 band（±带宽）两种模式
- **风格中性**：风格因子主动暴露 ±0.10
- **集中度**：单资产权重上限

### 成本模型
- **线性成本**：逐资产 bps，CLI 固定 5 bps 单边
- **二次冲击**：与下单金额平方成比例的市场冲击（仅 Python API 可配）

### 信号平滑（SignalSmoother）
检测 topN 成员换手率，超阈值自动触发 EMA 平滑 alpha 再优化 —— 降低无效换手。

### PIT（point-in-time）契约
交易所日历对齐、决策时点/执行时点分离、风险模型 as-of 快照、暴露滞后全部写入 metadata。

## 目录

```
src/riskfolio_qs/
├── configs/optimizer/   # mapping.v0.2.0.yaml + parameters.v0.2.0.yaml（冻结，随 wheel）
├── optimizers/          # router（8 优化器路由）、portfolio_optimizer（cvxpy 核心）、risk_model_builder
├── adapters/            # BarraPrecomputedAdapter、mock/real、data_access_inputs、output
├── runners/pipeline.py  # OptimizationPipeline（统一入口）
├── constraints/         # TE、行业/风格中性、集中度约束模板
├── smoothers/           # SignalSmoother（topN 换手检测 + EMA）
├── analysis/            # 仓位分析（P0 产物 / P1 PIT enrichment）
└── core/                # contracts、schema 校验
```

## 数据需求（按优化器路径）

| 路径 | alpha | market 数据 | Barra 风险模型 | 就绪 |
|---|---|---|---|---|
| topN 兜底 | ✅ | — | — | ✅ |
| 绝对收益 | ✅ | `data_access.Return` | — | ✅ |
| 历史指增 | ✅ | `data_access.Return` | 可选 | ✅ |
| 预计算 Barra | ✅ | — | exposure + factor_cov + specific_var | ✅ |

CLI 单一回测链路的固定取数：benchmark 从 `data_access/ashare_index_constituent`，
tradable 从 `ashare_stock_daily`，后续持仓 = 上一期优化结果。

## 测试

```bash
cd riskfolio_qs && pytest tests/ -q    # 43 个测试
```
覆盖：配置白名单/优先级、8 优化器路由、TE/中性约束生效性、成本计算、
求解验证报告（最大误差 2.77e-6）、topN 兜底触发。

## 硬性规则

- 组合寻优（Optuna/LGBM 调参）是组合链专属，本库只做**给定参数下**的优化求解
- alpha 一律 vwap-to-vwap 口径（全平台统一）
- 配置白名单外的参数名直接报错，不静默

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（benchmark/tradable/Return）、cvxpy + 求解器（ECOS/SCS/Clarabel）。
- **被谁调用**：`lightgbm_qs`（全链路组合优化环节）。

## 相关仓库

- **data_access** — 数据层（唯一读通道）
- **vectorbt_qs** — 下游回测执行（消费 target_positions.parquet）
- **factor_optimizer / factor_assets** — 上游因子寻优与入库
