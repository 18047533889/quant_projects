# GTJA-191 因子库（factor_engine DSL）— 独立投递包

国泰君安《基于短周期价量特征的多因子选股体系》191 个因子，已翻译为 **factor_engine DSL**，并按 `docs/miner_delivery_spec.md` 生成 `disk.v1` 投递包。

**本目录为独立包**，可直接 zip 发给他人；不依赖放在 `factor_engine/` 内部。

## 目录结构

```text
gtja191/
├── README.md
├── docs/
│   └── miner_delivery_spec.md     # 投递规范（随包附带）
├── examples/
│   └── gtja191_smoke.yaml         # factor_engine + data_access smoke 配置
├── lib/
│   ├── data_source.py             # data_access 数据源配置（与 factor_engine 对齐）
│   ├── dsl_normalize.py           # canonical 算子名归一（post-process）
│   ├── dsl_legacy_ops.py          # 废弃别名检测（测试/审计）
│   ├── engine_config.py           # smoke 最快路径（auto + read_auto）
│   ├── dsl_validate.py            # 独立校验（可选接 factor_engine 完整 parse）
│   └── paths.py
├── source/
│   └── gtja191_formulas.json      # 原始 GTJA 公式
├── dsl/
│   ├── gtja191_dsl_catalog.json   # 191 条 DSL + 校验状态
│   ├── manual_dsl_overrides.json
│   ├── operator_mapping.md
│   ├── fe_dsl_allowlist.json      # factor_engine 算子快照
│   ├── dsl_allowlist.json         # 策略白名单原文
│   └── audit_report.json
├── formulas/
│   └── gtja191_alpha_001.dsl … gtja191_alpha_191.dsl
├── scripts/
│   ├── parse_gtja_source.py
│   ├── convert_and_build_delivery.py
│   ├── export_allowlist.py        # 从 factor_engine 刷新算子白名单
│   ├── run_smoke.py               # data_access 读数 + factor_engine 执行 smoke
│   ├── audit_gtja_dsl.py
│   └── validate_manifests.py
└── candidate_pool/                # ★ 直接打包发人的核心内容
    └── manual_ashare_pv_202607041600/
        ├── config.json
        └── manual_*/manifest.json   # 185 条可投递 manifest
```

## 数据源（data_access）

读数已统一走 **`data_access`** 登记数据集，不再使用 legacy `local`/`cos` parquet 路径。

Campaign `config.json` 中的 `data_source` 示例：

```json
{
  "type": "data_access",
  "dataset": "ashare_stock_daily",
  "fields": {
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
    "vwap": "Vwap",
    "amount": "Amount",
    "ret": "Return",
    "preclose": "PreClose"
  },
  "read_auto": true
}
```

与 `factor_engine/docs/mining_data_source_presets.json` 中 `ashare_pv` preset 一致。配置定义见 `lib/data_source.py`。

## 打包发给别人

```bash
cd quantsociety
zip -r gtja191_delivery.zip gtja191/candidate_pool gtja191/formulas gtja191/dsl gtja191/docs gtja191/README.md
```

对方将 `candidate_pool/manual_ashare_pv_202607041600/` 拷入其 factor_engine 的 `candidate_pool/` 即可按 `miner_delivery_spec.md` 跑 AFV。

## 本包内自检

```bash
cd gtja191/scripts
python3 run_tests.py          # 全量单元 + 回归 + 投递一致性测试
python3 audit_gtja_dsl.py
python3 validate_manifests.py
python3 export_allowlist.py    # 刷新 fe_dsl_allowlist.json（需旁边有 factor_engine）
```

## 执行 smoke（read_auto + pandas，191 条数值 parity）

```bash
cd gtja191/scripts
python3 run_smoke.py --factor gtja191_alpha_001

# 或在 factor_engine 根目录
cd ../factor_engine
PYTHONPATH=. python3 run_pipeline.py config ../gtja191/examples/gtja191_smoke.yaml --run-only
```

执行栈：
- **读数据**：`data_access` + `read_auto`（Arrow 零拷贝 collect）
- **算子**：`backend: pandas` + `operator_backend: pandas_numpy`（与 factor_engine 参考结果一致）
- **说明**：`backend: auto` 对部分 `ts_corr(rank(...), rank(...))` 嵌套式在 Polars 层暂无数值 parity；smoke 与示例 YAML 固定走 pandas

见 `lib/engine_config.py`。

## 重新生成（可选）

```bash
cd gtja191/scripts
python3 convert_and_build_delivery.py
python3 export_allowlist.py
```

若本机旁边有 `../factor_engine`，会自动用完整 `parse_expr`；否则用包内 AST + 白名单校验。也可设置：

```bash
export FACTOR_ENGINE_ROOT=/path/to/factor_engine
export DATA_ACCESS_ROOT=/path/to/data_access
```

## 暂不投递（6 条）

| 因子 | 原因 |
|------|------|
| 030、183 | 零因子 stub |
| 075、149、181、182 | 需 `index_close` / `index_open` benchmark 列（`ashare_index_daily` composite 待接） |

详见 `dsl/operator_mapping.md`。
