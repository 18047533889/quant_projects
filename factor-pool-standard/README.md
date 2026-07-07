# factor-pool-standard

因子投递契约的**单一事实来源**（canonical 字段、domain 规则、manifest 校验）。

## 目录

```text
factor-pool-standard/
├── enums/
│   ├── canonical_data_fields.json   # 全市场字段注册表（由 build_canonical_fields.py 生成）
│   └── domain_roots.yaml            # domain_root 枚举
└── scripts/
    └── check_manifest_fields.py     # manifest 字段 / domain 校验 CLI
```

## 生成 canonical JSON

```bash
cd ~/quant_projects
python3 factor_engine/scripts/build_canonical_fields.py
```

## 校验 manifest

```bash
python3 factor-pool-standard/scripts/check_manifest_fields.py path/to/manifest.json
python3 factor-pool-standard/scripts/check_manifest_fields.py campaign_dir/ --recursive
```

与 `factor_engine/scripts/validate_delivery_formula.py` 分工：本脚本查**列名与 domain**；后者查 **DSL 语法**。
