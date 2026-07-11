# Label 产物结构

## 输出目录

```
{state_dir}/
├── tag_package.json
├── admission_decision.json
├── route_record.json
├── lifecycle_event.json
└── logs/
    └── label_pipeline.log
```

## TagPackage 结构

```json
{
  "factor_id": "equity_1d_cross_sectional_a3f2b1c0",
  "rule_tags": {
    "performance_tags": ["high_ic", "low_turnover"],
    "style_tags": ["momentum"],
    "lifecycle_tags": ["new"],
    "action_tags": ["route_to_production"]
  },
  "deepseek_tags": {
    "primary_label": "momentum_factor",
    "confidence": 0.85,
    "explanation": "..."
  },
  "label_governance": {
    "registry_hit": true,
    "matched_label_key": "momentum_factor",
    "registry_action": "noop"
  }
}
```

## AdmissionDecision 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `tier` | string | 目标 tier：Tier3A / Tier3B / Tier3C / Tier3D / Tier2 / Tier2X / Tier4 |
| `reason` | string | 决策原因 |
| `incubation_goal` | string | 孵化目标（仅 Tier2/Tier2X） |
