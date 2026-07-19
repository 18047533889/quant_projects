# GTJA-191 / GTJA185 因子库（factor_engine DSL）

国泰君安《基于短周期价量特征的多因子选股体系》**185 条** A 股价量因子（源式 191，排除 6 条 index/stub），已翻译为 **factor_engine canonical DSL**。

> **投递规范**：[`../factor_engine/docs/miner_delivery_spec.md`](../factor_engine/docs/miner_delivery_spec.md)  
> **本包说明**：[`docs/miner_delivery_spec.md`](docs/miner_delivery_spec.md)  
> **平台总览**：[`../docs/量化平台使用总览.md`](../docs/量化平台使用总览.md)

**重要**：本包是 **`dsl_surface=compat`** published 包（含 `ts_ema` / `flex_max` / `ts_time_slope` 等）。  
校验 / 落盘必须 `parse_expr(..., surface="compat")`，不要用默认 `daily`。

## 目录结构

```text
gtja191/
├── README.md
├── docs/miner_delivery_spec.md    # 指向 FE 规范 + 本包约定
├── source/gtja191_formulas.json   # 191 条源公式
├── dsl/
│   ├── gtja191_dsl_catalog.json   # 转换后 catalog
│   ├── manual_dsl_overrides.json  # 手工 DSL
│   └── fe_dsl_allowlist.json      # compat 白名单快照
├── formulas/gtja191_alpha_*.dsl
├── candidate_pool/                # ★ disk.v1（185 manifests）
├── examples/materialize/          # 落值 YAML
├── lib/                           # catalog / data_source / engine
├── autofactor/provider.py         # FactorPack name=gtja185
└── scripts/
```

## 快速自检

```bash
cd gtja191/scripts
python3 export_allowlist.py --surface compat
python3 convert_and_build_delivery.py   # 重建 catalog + formulas + candidate_pool
python3 generate_materialize_configs.py
python3 run_tests.py
python3 validate_manifests.py
python3 run_smoke.py --factor gtja191_alpha_001
```

需 sibling `../factor_engine`，或 `export FACTOR_ENGINE_ROOT=...`。读数需 `dataaccess`（`import data_access`）。

## 落值

```bash
# 包内
python3 run_materialize.py --smoke --factor gtja191_alpha_001

# monorepo 脚本（surface=compat，走 deliverable_catalog）
cd ..
python3 scripts/materialize_gtja191_factors.py --skip-existing
```

## 执行栈（与当前 factor_engine 对齐）

| 层 | 配置 |
|----|------|
| Campaign `data_source` | `{local, cos}`（AFV 路径提示） |
| 执行 `data_source` | `type: data_access` + `ashare_stock_daily` + `read_auto` |
| DSL 解析 | `surface=compat` |
| Backend | `pandas`（嵌套 `ts_corr(rank,rank)` 数值 parity） |

## 打包投递

```bash
cd /home/shw/quant_projects
zip -r gtja185_delivery.zip \
  gtja191/candidate_pool \
  gtja191/formulas \
  gtja191/dsl \
  gtja191/docs \
  gtja191/README.md
```

将 `candidate_pool/manual_ashare_pv_202607041600/` 拷入 AFV `candidate_pool/`；校验时设 `FACTOR_ENGINE_DSL_SURFACE=compat`。
