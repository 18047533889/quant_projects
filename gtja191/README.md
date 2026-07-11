# GTJA-191 因子库（factor_engine DSL）— 独立投递包

国泰君安《基于短周期价量特征的多因子选股体系》**185 条** A 股价量因子，已翻译为 **factor_engine canonical DSL**，支持 **data_access 读数 + 因子湖落值**。

**本目录为独立包**，可直接 zip 发给他人；不依赖放在 `factor_engine/` 内部。

## 目录结构

```text
gtja191/
├── README.md
├── docs/miner_delivery_spec.md
├── examples/
│   ├── gtja191_smoke.yaml
│   └── materialize/               # 落值 YAML（generate 脚本生成）
├── lib/
│   ├── catalog.py                 # 185 条可投递因子
│   ├── data_source.py
│   ├── materialize_config.py
│   ├── engine_config.py
│   └── ...
├── dsl/gtja191_dsl_catalog.json
├── formulas/gtja191_alpha_*.dsl
├── scripts/
│   ├── run_materialize.py         # ★ 全量落值
│   ├── generate_materialize_configs.py
│   └── ...
└── candidate_pool/                # 185 manifest
```

## 打包发给别人

```bash
cd /home/shw/quant_projects
zip -r gtja191_delivery.zip \
  gtja191/candidate_pool \
  gtja191/formulas \
  gtja191/dsl \
  gtja191/docs \
  gtja191/README.md
```

## 落值（185 条 → 因子湖 Parquet）

```bash
cd gtja191/scripts

# 生成 185 个落值 YAML
python3 generate_materialize_configs.py

# 全量落值（默认 2014-01-01 ~ 2026-06-25）
python3 run_materialize.py

# 快速 smoke（短窗口，单因子）
python3 run_materialize.py --smoke --factor gtja191_alpha_001
```

默认因子湖路径：`../data/factors/lake/gtja191/factors/<factor_id>/year=YYYY/data.parquet`

环境变量：`GTJA191_LAKE_ROOT`、`FACTOR_ENGINE_ROOT`

## 自检

```bash
cd gtja191/scripts
python3 export_allowlist.py
python3 convert_and_build_delivery.py
python3 run_tests.py
python3 validate_manifests.py
python3 run_smoke.py --factor gtja191_alpha_001
python3 run_materialize.py --smoke --factor gtja191_alpha_001
```

## 执行栈

- **读数**：`data_access` + `read_auto` + `ashare_stock_daily`
- **算子**：`PandasBackend`（185 条数值 parity）

见 `lib/materialize_config.py`、`lib/engine_config.py`。
