# GTJA-191 因子库（factor_engine DSL）— 独立投递包

国泰君安《基于短周期价量特征的多因子选股体系》191 个因子，已翻译为 **factor_engine DSL**，并按 `docs/miner_delivery_spec.md` 生成 `disk.v1` 投递包。

**本目录为独立包**，可直接 zip 发给他人；不依赖放在 `factor_engine/` 内部。

## 目录结构

```text
gtja191/
├── README.md
├── docs/
│   └── miner_delivery_spec.md     # 投递规范（随包附带）
├── lib/
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
│   ├── audit_gtja_dsl.py
│   └── validate_manifests.py
└── candidate_pool/                # ★ 直接打包发人的核心内容
    └── manual_ashare_pv_202607041600/
        ├── config.json
        └── manual_*/manifest.json   # 185 条可投递 manifest
```

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
```

## 重新生成（可选）

```bash
cd gtja191/scripts
python3 convert_and_build_delivery.py
```

若本机旁边有 `../factor_engine`，会自动用完整 `parse_expr`；否则用包内 AST + 白名单校验。也可设置：

```bash
export FACTOR_ENGINE_ROOT=/path/to/factor_engine
```

## 暂不投递（6 条）

| 因子 | 原因 |
|------|------|
| 030、183 | 零因子 stub |
| 075、149、181、182 | 需 `index_close` / `index_open` benchmark 列 |

详见 `dsl/operator_mapping.md`。
