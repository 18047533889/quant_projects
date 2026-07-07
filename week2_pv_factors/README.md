# Week2 A股价量因子（factor_engine DSL）

来源：**Week2 汇报.pdf**（28 条 Qualified）+ **A.pdf**（9 条 Elite），合计 **37 条** factor_engine DSL。

本目录为**独立回测包**，与 `gtja191/` 并列，使用 **factor_engine 自有算子**（`ts_mean` / `ema` / `ts_rank` / `clip` / `sigmoid` 等）。

## 目录结构

```text
week2_pv_factors/
├── README.md
├── source/
│   ├── week2_factors_catalog.json
│   └── audit_report.json
├── formulas/
│   └── w2_001_*.dsl … w2_028_*.dsl
├── candidate_pool/                # ★ disk.v1 投递包
│   └── manual_ashare_week2_pv_202606301800/
│       ├── config.json
│       └── manual_*/manifest.json   # 37 条（28 Week2 + 9 A.pdf Elite）
├── lib/
│   ├── dsl_validate.py
│   └── paths.py
└── scripts/
    ├── build_catalog.py
    ├── build_delivery.py          # 生成 candidate_pool
    ├── validate_all.py
    ├── validate_manifests.py
    └── audit_formulas.py
```

## 快速开始

```bash
cd week2_pv_factors/scripts
python3 build_catalog.py      # 生成 formulas/ 与 catalog
python3 build_delivery.py     # 生成 candidate_pool（config + manifest）
python3 validate_manifests.py # 校验 disk.v1 投递包
python3 validate_all.py       # factor_engine parse_expr
python3 audit_formulas.py     # 算子白名单 + 汇报对照
```

需本机有 `../factor_engine`，或：

```bash
export FACTOR_ENGINE_ROOT=/path/to/factor_engine
```

## 打包投递

```bash
cd quantsociety
zip -r week2_pv_delivery.zip week2_pv_factors/candidate_pool week2_pv_factors/formulas week2_pv_factors/source week2_pv_factors/README.md
```

将 `candidate_pool/manual_ashare_week2_pv_202606301800/` 拷入 AFV 的 `candidate_pool/` 即可跑 Gateway / 回测。

## 在 factor_engine 里回测单条

```bash
cd ../factor_engine
PYTHONPATH=. python3 -c "
from api.dsl_parser import parse_expr
from runtime.real_data_factor_smoke import main as smoke  # 若已配置数据路径
expr = open('../week2_pv_factors/formulas/w2_001_vol_quantile_stress.dsl').read().strip()
parse_expr(expr)
print('OK', expr[:80])
"
```

或在引擎内按 `Factor(name=..., expr=...)` 批量注册 `source/week2_factors_catalog.json` 中的 `formula` 字段。

## 因子来源

| 来源 | 数量 | 说明 |
|------|------|------|
| Week2 汇报.pdf | 28 | 第二周 Qualified 因子池 |
| A.pdf | 9 | LQTP Elite 因子（含 IC/ICIR 指标） |

### A.pdf Elite 九条（`a_001`–`a_009`）

| ID | 名称 | Rank IC |
|----|------|---------|
| a_001 | 日内收益×量能 Surge | 4.52% |
| a_002 | 羊群动量×应激区间 | 4.51% |
| a_003 | 方向性波动×量能(25日) | 4.32% |
| a_004 | 量确认动量(EWM平滑) | 4.32% |
| a_005 | 日内反转 | 3.94% |
| a_006 | 量调整动量 | 3.85% |
| a_007 | 自适应方向波动交叉 | 3.75% |
| a_008 | 波动门控动量×量比 | 3.65% |
| a_009 | 量确认动量(5日) | 3.62% |

> a_004 / a_006 / a_007 在 A.pdf 中无完整公式展开，已按经济逻辑簇与同类 Elite 结构还原；a_001–a_003 / a_005 / a_008 / a_009 与 PDF 公式页一致。

## 五大逻辑簇（Week2 部分）

| 簇 | 数量 | 代表因子 |
|----|------|----------|
| 羊群 × 应激 | 8 | w2_001（Rank IC 4.99%） |
| 日内压力 / 反转 | 6 | w2_007（4.56%） |
| 方向波动 × 量能 | 3 | w2_013（4.32%） |
| 波动率门控 / Regime | 4 | w2_016（3.91% Elite） |
| 订单失衡 / 不对称 | 7 | w2_022（3.28%） |

## 自检

```bash
cd week2_pv_factors/scripts
python3 build_catalog.py
python3 validate_all.py       # factor_engine parse_expr
python3 audit_formulas.py     # 算子白名单 + 汇报对照
```

审计报告见 `source/audit_report.json`。

## 符号对照（汇报 → DSL）

| 汇报符号 | DSL |
|----------|-----|
| C/O/H/L/V | `close` / `open` / `high` / `low` / `volume` |
| ΔC（绝对涨跌） | `ts_delta(close, 1)` |
| ΔC/C（日收益） | `ts_delta(close,1)/delay(close,1)` |
| ts_pct(C,n) | `close/delay(close,n)-1` |
| MA(V,n) | `ts_mean(volume,n)` |
| rank_pct(V,n) | `ts_rank(volume,n)` |
| Q(x,n,50) | `median(x,n)` |
| z(range,n) | `(range-ts_mean(range,n))/ts_std(range,n)` |

## 审计结论（2026-07-05）

- **37/37** 通过 `factor_engine.parse_expr`
- **18 种算子**，全部在 FE 白名单内
- **已修正 5 处**与汇报原文不一致的翻译（见 `audit_report.json`）
- **3 条近似**：w2_022/027/028 的条件 median 用 `ts_sum` 条件均值近似（FE 无原生条件 median）

## 近似说明

- **条件 median**（w2_022/027/028）：`divide(ts_sum(if_else(cond,range,0),n), ts_sum(if_else(cond,1,0),n))` 近似 `median(range|cond,n)`
- **ts_pct / ts_quantile**：已改写为等价式（见 `notation_map`）

## Elite 因子（3 条）

- `w2_014` 自适应 sigmoid 量能混合（3.85%）
- `w2_016` 量加权价格变动（3.91%）
- `w2_019` 应激累积 × 量权 × 方向（3.45%）
