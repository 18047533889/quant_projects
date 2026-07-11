# Gateway 模块功能概述

## 一、模块目的

**Gateway（极速网关）** 是因子评估流水线的第一道关口，对整个系统中所有候选因子进行快速合法性校验、风险拦截与路由决策。

核心职责：
1. **Schema 校验** — 检查候选因子 JSON 格式和必需字段
2. **算子白名单验证** — 确保表达式中仅使用允许的算子
3. **未来函数拦截** — 检测并拒绝引用未来数据的因子
4. **复杂度评估** — 评估计算开销，超出预算则拒绝
5. **语义去重** — 识别重复因子，避免冗余计算
6. **数据质量检测**（可选）— 检测市场数据的分布漂移与流动性异常

---

## 二、检查流程

```
候选因子 afv.json
    │
    ▼
┌─────────────────────┐
│ ① Schema 校验        │  必需字段、类型检查
└────┬────────────────┘
    │
    ▼
┌─────────────────────┐
│ ② 算子白名单验证      │  检查是否全是允许的算子
└────┬────────────────┘
    │
    ▼
┌─────────────────────┐
│ ③ 未来函数拦截        │  检测是否引用未来数据
└────┬────────────────┘
    │
    ▼
┌─────────────────────┐
│ ④ 复杂度评估          │  计算表达式复杂度得分
└────┬────────────────┘
    │
    ▼
┌─────────────────────┐
│ ⑤ 语义去重            │  哈希匹配 + 可选向量相似度
└────┬────────────────┘
    │
    ▼
┌─────────────────────┐
│ ⑥ 数据质量检测(可选)   │  分布漂移、流动性异常
└────┬────────────────┘
    │
┌────┴──────┬──────────┬──────────────┐
▼           ▼          ▼              ▼
Pass     Duplicated   Rejected     (Pass+警告)
```

---

## 三、路由决策

| 结果 | 条件 | 输出目录 |
|------|------|----------|
| **Pass** | 所有检查通过 | `tier1/gateway_pass_base/` |
| **Duplicated** | 语义重复（引用历史报告） | `tier1/gateway_duplicated_base/` |
| **Rejected** | Schema 失败 / 非法算子 / 未来函数 / 超预算 | `tier4/anti_sample_base/gateway/` |

---

## 四、子组件

| 组件 | 类/函数 | 文件 | 职责 |
|------|---------|------|------|
| Schema 校验 | `StaticValidator` | `validator.py` | 检查必需字段、格式、schema 版本 |
| 算子白名单 | `OperatorValidator` | `validator.py` | 验证算子在白名单中 |
| 未来函数扫描 | `FutureFunctionScanner` | `future_scanner.py` | 检测未来函数和时间偏移 |
| 复杂度评估 | `ComplexityEvaluator` | `complexity.py` | 计算表达式复杂度得分 |
| 语义去重 | `SemanticDeduplicator` | `deduplicator.py` | 哈希匹配 + 可选的向量相似度 |
| 数据质量 | `DataQualityRunner` | `data_quality.py` | 分布漂移、流动性异常检测 |
| 磁盘 I/O | `IOManager` | `io_utils.py` | 读写 `afv.json` / `manifest.json`，管理路由与缓存 |
| Kafka 发送 | `KafkaProducer` | `kafka_producer.py` | 将 Pass 的候选因子发送到 Kafka |
| 配置管理 | `GatewayConfig` | `config.py` | 加载 YAML+JSON 配置文件 |

---

## 五、输入输出

### 输入
- 候选因子目录（来自 `candidate_pool`）：内含 `candidate.json`（上游投喂的原始候选 JSON）
- 也支持直接传入已解析的候选因子字典

### 输出
- `write_afv()`：在临时库写入 `afv.json`（统一元数据文件，包含 Gateway 审查结果，替代旧版 `candidate.json`）
- `write_output()`：（旧接口）写入 `candidate.json` + `manifest.json`
- `move_to_temp()`：将目录转移到 `tier0/gateway_temp/` 等待路由
- `route_factor()`：路由到最终目标库

### afv.json 的 Gateway 段结构

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "cand_20260528_a1b2c3d4",
  "campaign_id": "campaign_alpha_001",
  "formula": "ts_mean(close, 10) / ts_std(close, 10)",
  "gateway_version": "1.0.0",
  "Gateway": {
    "Label": "Pass",
    "Reason": "",
    "run_id": "gateway_20260528_083000_a1b2c3d4",
    "checked_at": "2026-05-28T08:30:00.123456+00:00",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 3.5,
      "is_within_budget": true,
      "is_duplicate": false
    }
  }
}
```
