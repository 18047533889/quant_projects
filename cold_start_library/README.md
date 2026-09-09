# cold_start_library（V9 + expand）

FactorEngine **backend-audited V9** 冷启动因子库，供 AlphaPROBE / Factorminer 直接抽样使用。
已基于本地 COS 镜像（`~/quant_projects/data/a_share/lqtp_data`）扩充算子覆盖，并接入估值字段。

旧版四库（PV curated + Alpha101/158/191）已归档到同级 `cold_start_library_legacy_*`。

## AlphaPROBE 用法（A 股）

主产物：

| 文件 | 说明 |
|---|---|
| `library/production_default_core_v9.json` | 随 Python 包交付的默认 A 股、日频、core、V9 validated 冷启动池 |
| `data/ashare/backend_v9_core.yaml` | 可选的 AlphaPROBE 预计算 yaml；只有显式生成/配置后才使用 |
| `data/ashare/backend_v9_metrics.jsonl` | 逐条执行 / IC 明细 |
| `data/ashare/expand_generated.json` | expand_v1 新增因子底稿 |
| `library/production_default_core_v9.json` | V9 原始审计库 |

配置示例（已写入 `experiment_ashare_pv_7x24.yaml`）：

```yaml
data:
  fe_profile: ashare_pv_valuation   # 价量 + pe/pb/turnover/market_cap
mining:
  cold_start_sample_size: 100
    cold_start_library: ~/quant_projects/cold_start_library/data/ashare/backend_v9_core.yaml
```

不传路径时，运行时只加载随包交付的
`library/production_default_core_v9.json`，不会在缺文件时自动改用 extended、US
或其他审计层级。显式传入 `.yaml`/`.yml` 仍保留原 AlphaPROBE 格式。

## 扩充算子覆盖 + 增量预计算

```bash
export ASHARE_PARQUET_ROOT=~/quant_projects/data/a_share/lqtp_data
cd ~/quant_projects/cold_start_library
PYTHONPATH=src:../factor_engine:../data_access \
  python -u scripts/expand_and_precompute_ashare.py \
    --start 2021-07-01 --end 2021-09-30 --label-days 20
```

说明：

- 补齐稀有/缺失算子（技术指标、回归、topk、条件计数、逻辑门控、估值等）
- 多窗口 / 短长差结构，扩大 AlphaPROBE 搜索空间
- 估值数据来自 `StockValuationDaily`（与 COS 镜像一致）
- 增量写入 yaml，并为新因子落 `metrics.ic/icir/finite_ratio`

## 重新生成 V9 基线 yaml

```bash
PYTHONPATH=src:../factor_engine:../data_access \
  python scripts/build_ashare_alphaprobe_yaml.py \
    --start 2021-07-01 --end 2021-09-30 --label-days 20
```

## 包布局

```text
cold_start_library/
├── data/ashare/           # AlphaPROBE 消费
├── library/               # V9 JSON 原件
├── scripts/               # 构建 / 扩充 / 预计算
├── src/cold_start_library/
└── vendor_v9/             # zip 完整交付副本
```
