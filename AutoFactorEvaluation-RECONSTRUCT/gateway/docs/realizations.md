# Gateway 模块工程实现

## 一、GatewayCore 流程详解

`GatewayCore.process(candidate_dict)` 的执行顺序：

```
1. Candidate.from_dict(candidate_dict)
   → 解析 JSON 为 Candidate 对象
   → 失败 → Rejected

2. StaticValidator.validate_all(candidate_dict)
   → Schema 校验 + 算子白名单
   → 失败 → Rejected (is_legal=False)

3. FutureFunctionScanner.scan(candidate.expr)
   → 正则匹配黑名单函数和偏移模式
   → 检测到 → Rejected (has_future=True)

4. ComplexityEvaluator.evaluate(candidate.expr)
   → 算子权重累加 + 嵌套深度惩罚
   → 超预算 → Rejected

5. SemanticDeduplicator.is_duplicate(expr, config, candidate_id)
   → 哈希匹配已有记录
   → 重复 → Duplicated

6. DataQualityRunner.check(market_data_path)  [可选]
   → 分布漂移检测 + 流动性异常 + 时钟抖动
   → 结果附加到 GatewayResult，不影响 Pass/Rejected 决策

7. 注册到去重缓存 → 返回 Pass
```

## 二、AST 解析模式

### 快速模式（默认）
- 使用正则表达式 `\b([A-Z][A-Z0-9_]*)\s*\(` 匹配算子
- 括号深度分析通过 `[()]` 正则计数
- 优点：零依赖、速度快
- 缺点：无法处理嵌套复杂表达式

### 精确模式（可选，use_ast_parser=True）
- 调用 `factor_engine` 的 `api.dsl_parser.parse_expr`
- 生成完整的 AST 树进行分析
- 优点：精确处理所有边缘情况
- 缺点：依赖 `factor_engine` 包

## 三、语义去重实现

### 哈希去重（当前默认）
```python
expr_hash = sha256(normalized_expression)
candidate_hash = sha256(normalized_expr + canonical_config)

# 先查 candidate_hash（完全匹配）
# 再查 expr_hash + config 等价判断（表达式相同、配置等价）
```

### 向量去重（use_vector_db=True 时启用，当前未激活）
- 使用向量数据库（Milvus 等）进行语义相似度搜索
- 需要外部向量数据库服务

## 四、数据质量检测

### 分布漂移检测
- **KS 检验**：比较当前数据与历史数据的分布差异
- **Wasserstein 距离**：衡量分布偏移的幅度
- 配置阈值：`ks_p_value_threshold: 0.05`, `wasserstein_threshold: 1.0`

### 流动性异常检测
- **Spread 突变**：检测买卖价差的 Z-score 异常
- **LOB 缺失**：检测订单簿深度为零的比例
- 配置阈值：`spread_spike_threshold: 2.0`, `lob_gap_threshold: 0.05`

### 时钟抖动验证
- 检测时间戳逆序数量
- 计算时序有效性得分

## 五、Kafka 集成

- 可选启用，通过 `features.enable_kafka` 控制
- 发送通过网关的候选因子到 `factor_candidates_compute` topic
- 支持异步发送 + 重试机制
- 无 Kafka 依赖时自动降级（跳过发送）

## 六、配置管理

配置来源优先级：
1. `gateway_config.yaml` 主配置
2. `operator_whitelist.json` / `future_blacklist.json` / `complexity_weights.json`
3. 环境变量覆盖（`GATEWAY_*`）

## 七、CLI 入口

```bash
# 处理单个候选
python -m gateway.main --candidate-id cand_20260528_a1b2c3d4

# 批量处理 campaign
python -m gateway.main --campaign-id campaign_alpha_001

# 自定义配置
python -m gateway.main --candidate-id xxx --config /path/to/config.yaml
```
