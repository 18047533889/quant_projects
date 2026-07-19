# FactorEngine 原生冷启动因子库

本目录是面向因子挖掘、遗传搜索、LLM/Agent 迭代和 AutoFactorEvaluation 的**冷启动母库**。它与 `gtja191/`、`week2_pv_factors/`并列，不修改历史公式包，也不把候选因子塞进评估框架内部。

## 当前规模

| 市场 | 层级 | 解析 surface | 数量 | 说明 |
|---|---|---|---:|---|
| A股 | daily | `daily` | 819 | 仅使用production-safe daily算子 |
| 美股 | daily | `daily` | 1007 | 增加隔夜/日内、短售、盘口和美股派生字段 |
| A股 | extended | `compat` | 1359 | 技术指标、尾部统计、条件滚动、回归和高级截面 |
| 美股 | extended | `compat` | 1435 | A股extended能力 + 美股隔夜/日内扩展 |

合计 **4620** 条。公式与GTJA191/GTJA185及Week2包做AST结构哈希去重，而不是只按字符串空格去重。

## 设计原则

- **市场隔离**：A股使用`factor`，美股使用`adj_factor`和`ret__intra`、`ret__overnight`等双下划线canonical字段。
- **surface隔离**：daily目录只能调用`DAILY_CANONICALS`；extended目录使用`compat`承载daily+extended组合，禁止research-only算子混入。
- **字段分层**：`core`、`derived`、`enriched`、`microstructure`，下游可按实际数据列过滤。
- **因果约束**：禁止负lag、lead、backfill等未来信息路径。
- **可复现抽样**：按family、期限桶和复杂度分层轮询，不是简单随机抽取。
- **不强行滥用算子**：报告期基本面算子、复数相位和奇异/爆炸三角函数有明确排除理由。

## 目录

```text
factor_cold_start/
├── catalogs/                      # build_catalog.py可生成的JSON交付目录
├── reports/coverage.json          # 机器可读覆盖报告
├── reports/coverage.md            # 人类可读覆盖报告
├── autofactor/provider.py         # AutoFactorEvaluation FactorPack provider
├── generator.py                   # 确定性公式生成器
├── catalog.py                     # 目录加载与字段过滤
├── sampler.py                     # 多样性分层抽样
├── scripts/build_catalog.py       # 重建目录和报告
├── scripts/validate_catalog.py    # 全量fail-closed校验
├── scripts/sample_batch.py        # CLI抽样
└── tests/
```

## 使用

### 加载目录（默认由生成器确定性构建并缓存）

```python
from factor_cold_start import load_catalog

ashare_daily = load_catalog("ashare", "daily")
us_extended = load_catalog("us", "extended")
```

### 按真实字段抽样

```python
from factor_cold_start import sample_factors

batch = sample_factors(
    market="us",
    surface="daily",
    size=64,
    seed="experiment-001",
    available_fields={
        "open", "high", "low", "close", "pre_close", "volume", "amount",
        "ret", "vwap", "adj_factor", "ret__intra", "ret__overnight",
        "high__low__ratio", "upper__shadow__ratio", "vwap__close__dist",
    },
    availability_tiers=("core", "derived"),
    max_per_family=8,
)
```

### CLI

```bash
python factor_cold_start/scripts/sample_batch.py \
  --market ashare --surface daily --size 64 --seed round-1

python factor_cold_start/scripts/validate_catalog.py --all
python factor_cold_start/scripts/build_catalog.py
```

### AutoFactorEvaluation

```bash
PYTHONPATH=.:factor_engine:AutoFactorEvaluation-RECONSTRUCT \
python -m evaluation.batch \
  --provider factor_cold_start.autofactor.provider:load_ashare_daily_pack \
  --output-dir output/factor_evaluation \
  --synthetic
```

可用provider：

- `load_ashare_daily_pack`
- `load_us_daily_pack`
- `load_ashare_extended_pack`
- `load_us_extended_pack`
- `load_pack`：读取`FACTOR_COLD_START_MARKET`和`FACTOR_COLD_START_SURFACE`

## 因子族

覆盖收益动量、反转、趋势位置与质量、波动与区间、成交量/成交额、流动性冲击、价量相关与beta、K线/VWAP、条件regime、稳健非线性、市场宽度、调整事件、换手与规模、分组相对值、Wilder技术指标、MACD、尾部/Top-K/Bottom-K、条件均值与streak、衰减、滚动回归、偏相关、高级加权截面，以及美股隔夜—日内、短售和盘口微观结构。

完整统计见[`reports/coverage.md`](reports/coverage.md)。
