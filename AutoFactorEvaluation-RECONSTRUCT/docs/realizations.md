# AutoFactorEvaluation 工程实现

## 整体架构

### 四阶段流水线

```
┌──────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│ Gateway  │───→│Assetization│───→│Purification │───→│ Evaluation │
│ (校验/路由)│   │ (ID生成/计算)│   │ (清洗/去噪)  │   │ (评估/贴标) │
└──────────┘    └────────────┘    └─────────────┘    └────────────┘
     │                │                 │                  │
     ▼                ▼                 ▼                  ▼
 tier1/           tier1/             tier1/          统计指标 +
 gateway_pass    assetization_      purification_   标签 + 路由
 _base           raw_factor_base    pure_factor_base  信息
```

### 设计原则

1. **磁盘静态库接力** — 各模块通过约定目录结构和标准文件格式松耦合衔接
2. **纯函数核心** — `resolve_factor_id` 和 `compute_factor_values` 是无副作用的纯函数
3. **Fail-fast 校验** — 每个模块入口/出口均有严格校验，及早暴露问题
4. **可选降级** — Purification 在缺少辅助数据时跳过正交化，不影响主干流程

---

## 各模块关键实现

### Gateway

位于 `gateway/gateway/` 子包中。

- **GatewayCore.process()** 按序执行 6 步检查：Schema 校验 → 算子白名单 → 未来函数检测 → 复杂度评估 → 语义去重
- **AST 解析**：默认正则模式（零依赖），可选 factor_engine AST 解析器（精确模式）
- **去重策略**：基于 SHA256 哈希的语义去重
- **Kafka**：可选发送 Pass 因子到消息队列（默认关闭）
- **配置**：通过 `GatewayConfig` 加载 `gateway_config.yaml`，支持独立配置目录

### Assetization

位于 `assetization/scripts/` 中。

- **factor_id 格式**：`{asset_class}_{frequency}_{signal}_{uuid8/sha8}`
- **DSL 编译流水线**：parse → IR → LogicalPlan → Optimize → PhysicalPlan → execute
- **双后端**：pandas（默认）和 polars，通过 `factor_engine` 统一接口
- **市场数据校验**：必需列 + 公式引用列检查 + 空值率阈值
- **输入兼容**：同时支持新版 `Gateway.Label` 和旧版 `Gateway.status`
- **输出**：`data/` 目录下按日分片 parquet，**非单一 `data.parquet`**
- **afv.json**：文件名固定为 `afv.json`，**非 `candidate.json`**
- **无 fallback 默认值**：缺失路径直接抛出 `ValueError`

### Purification

位于 `purification/scripts/` 中。

- **缺失值填补**：行业加权均值填补（行业面板数据）/ 时序前向填充 + 衰减
- **去极值**：MAD 方法，numpy 向量化实现，乘数 3.148（对应 3-Sigma）
- **正交化**：WLS 回归，逐时间截面计算残差
- **校验**：5 项校验（列完整性、主键唯一、factor_id 一致性、时间序、manifest row_count）
- **降级策略**：无辅助数据时跳过正交化，使用 forward_fill_decay 回退

### Evaluation

位于 `evaluation/scripts/` 中。

- **多 horizon 支持**：同时计算多个持有期的绩效（默认 [1, 5, 20] 天，**非 21 天**）
- **PIT 对齐**：因子数据与行情数据点对点对齐，无前视偏差
- **单 horizon 契约**：indicator 每次只处理一个 horizon，pipeline 按 horizon 切片
- **DeepSeek V4 Flash 真实 API**：通过 `deepseek_client.py` 使用真实 API，支持重试和退避
- **前向收益率缓存**：使用 `database/cache/timeseries_forward_return/` 共享缓存目录，避免重复计算
- **vectorbt 优化**：
  - 分位数计算：使用 vectorbt，获得约 2.1x 加速
  - IC 计算：使用 vectorbt，获得约 1.4x 加速
- **`EvaluationConfig`**：从 `all_configs/auto_factor_evaluation/evaluation_config.yaml` 加载

---

## 配置管理

### ConfigManager（配置总线）

```python
from config_manager import ConfigManager

config = ConfigManager()           # 使用环境变量 AFVCONFIG 或默认路径
config = ConfigManager(config_dir="/path/to/config")  # 显式指定

# 路径访问
config.path("candidate_pool")      # → Path 对象
config.output_tier("tier3b_satellite")  # → Path 对象
config.market_data_path            # → Path
config.database_root               # → Path

# 参数访问
config.gateway_param("max_complexity")       # → 20.0
config.evaluation_param("horizons")          # → [1, 5, 20]
config.purification_param("imputation_method")  # → "industry_weighted"
```

### pipeline.py 中的 `_get_config()` 单例

```python
_CONFIG: Optional[Any] = None

def _get_config() -> Any:
    global _CONFIG
    if _CONFIG is None:
        from config_manager import ConfigManager
        _CONFIG = ConfigManager()
    return _CONFIG
```

所有路径通过此单例访问，无任何代码级默认值。

### 环境变量 `AFVCONFIG`

```bash
export AFVCONFIG=/path/to/custom_profile
python pipeline.py --all
```

---

## pipeline.py 全流程调用链

```
main()
├── [--gateway-only]
│   ├── run_gateway_pipeline()        ← 多进程 Gateway 审查
│   └── run_router()                  ← 路由至目标目录
│
├── [--assetization-only]
│   ├── run_assetization_pipeline()   ← 多进程因子计算
│   └── run_assetization_router()     ← 路由至 raw_factor_base
│
├── [--purification-only]
│   ├── run_purification_pipeline()   ← 多进程因子纯化
│   └── run_purification_router()     ← 路由至 pure_factor_base
│
├── [--evaluation-only]
│   ├── run_evaluation_pipeline()     ← 多进程评估
│   └── run_evaluation_router()       ← 路由至目标 tier
│
└── [--all]  全流程（带基准计时）
    ├── 预加载行情数据 → fork → 子进程继承
    ├── Gateway → Router
    ├── Assetization → Router
    ├── Purification → Router
    └── Evaluation → Router
        └── 输出性能基准报告
```

---

## 日志体系

| 模块 | 日志文件 | 位置 |
|------|---------|------|
| Pipeline 编排 | pipeline 日志（stdout） | 终端输出 |
| Gateway | `gateway.log` | `database/log/gateway/` |
| Assetization | `assetization.log` | `database/log/assetization/` |
| Purification | `purification.log` | `database/log/purification/` |
| Evaluation | `evaluation.log` | `database/log/evaluation/` |
| Timeseries (4.1) | `timeseries/{eval_run_id}.log` | `database/log/evaluation/timeseries/` |
| Indicator (4.2) | `stage4_2_metrics.log` | `database/log/evaluation/indicator/` |
| Label (4.3) | `label_pipeline.log` | `database/log/evaluation/label/` |

---

## 市场数据缓存机制

采用三级缓存，逐步优化 I/O 性能：

```
层级1: 原始 parquet 分片         ← 原始行情数据（按日存储）
            ↓ [merge]
层级2: 合并 parquet（~388MB）    ← 约 6.8x 压缩比
            ↓ [preload → fork]
层级3: 进程内存（COW 共享）       ← fork 子进程继承，零 I/O
```

- 合并缓存位于 `database/cache/market_data/`
- 通过 `cache_utils.py` 中的 `preload_market_data()` 函数管理
- `pipeline.py --preload`（默认开启）在 fork 前执行预加载
