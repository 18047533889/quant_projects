# Purification 模块功能概述

## 一、模块目的

**因子纯化（Purification）** 是因子评估流水线的第三道环节，负责对
Assetization 产出的原始因子值进行清洗和去噪，产出高质量的纯化因子。

三大处理步骤：
1. **动态缺失值填补** — 处理因子值中的缺失数据
2. **稳健去极值** — 消除极端值对因子分布的影响
3. **风险正交化**（可选）— 剥离已知风险因子暴露

## 二、核心函数

### 2.1 run_purification_pipeline

**签名**: `run_purification_pipeline(raw_factor_dir, pure_factor_dir, formal_base_dir=None) → bool`

**功能**: 完整纯化流程编排。

**入参**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `raw_factor_dir` | str | 是 | RawFactor 输入目录路径 |
| `pure_factor_dir` | str | 是 | PureFactor 输出目录路径 |
| `formal_base_dir` | str | 否 | 正交化依赖数据路径（行业/权重/风险暴露） |

**出参**: `bool` — 执行是否成功

### 2.2 dynamic_imputation

**签名**: `dynamic_imputation(factor_matrix, method, industry_labels, weights, max_delay) → pd.DataFrame`

**功能**: 填补因子宽表中的缺失值。

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| `industry_weighted` | 行业内市值加权均值填补 | 有行业分类和市值权重数据 |
| `forward_fill_decay` | 时序前向填充 + 衰减 | 无辅助数据时的回退方案 |

### 2.3 robust_winsorization

**签名**: `robust_winsorization(factor_matrix) → pd.DataFrame`

**功能**: 使用 MAD 方法进行稳健去极值。
- MAD 乘数固定为 3.148（对应正态分布的 3-Sigma）
- 基于中位数而非均值，对异常值更鲁棒

### 2.4 risk_orthogonalization

**签名**: `risk_orthogonalization(factor_matrix, risk_exposures, weights) → pd.DataFrame`

**功能**: WLS 回归正交化，剥离已知风格因子暴露。
- 需要风险因子矩阵（如 SIZE, BETA, MOMENTUM）和权重
- 输出纯净 Alpha 残差
