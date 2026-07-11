# Purification 模块工程实现

## 一、数据加载

### 因子数据加载

从因子目录的 `data/` 子目录下读取所有 `*.parquet` 文件并合并为长表：

```python
data_dir = factor_dir / "data"
parquet_files = sorted(data_dir.glob("*.parquet"))
dfs = [pd.read_parquet(f) for f in parquet_files]
result = pd.concat(dfs, ignore_index=True)
```

列名映射（Purification 内部统一使用 `datetime`/`asset`）：
- `TradeDate` → `datetime`
- `Symbol` → `asset`

### 行业分类数据加载

从 `StockIndustry` 目录按日期加载行业分类，筛选指定口径（`IndustrySource` 字段）：
```
{industry_data_path}/StockIndustry/{YYYY-MM-DD}.parquet
```

## 二、处理流程详解

```
RawFactor (data/{YYYY-MM-DD}.parquet files)
    │
    ├── 读取合并为 long-format DataFrame
    ├── 列名映射 TradeDate→datetime, Symbol→asset
    ├── 校验：必需列(datetime, asset, factor_id, factor_value)、主键唯一
    │
    ├── pivot to wide-format (index=datetime, columns=asset)
    │
    ├── ① 动态缺失值填补
    │   ├── industry_weighted: 行业内市值加权均值填补（需行业分类数据）
    │   └── forward_fill_decay: 时序前向填充 + 0.9 衰减因子（回退方案）
    │
    ├── ② 稳健去极值 (MAD=3.148)
    │   └── numpy vectorized: median → abs_dev → mad → clip
    │
    ├── ③ [可选] WLS 风险正交化
    │   └── X = [行业 dummy | 因子池暴露], W = weights
    │   └── 逐时间截面: lstsq(X_w, y_w) → resid = y - X @ beta
    │
    ├── melt back to long-format
    ├── 列名恢复 datetime→TradeDate, asset→Symbol
    ├── 校验（同输入校验标准）
    │
    └── 写入 PureFactorBase
        ├── data/{YYYY-MM-DD}.parquet（按日分片）
        ├── afv.json（追加 Purification 段）
        └── manifest.json
```

## 三、缺失值填补实现

### industry_weighted 策略
- 将资产按行业标签分组
- 每时间截面，计算行业内市值加权均值
- 用该均值填补同一行业内缺失的因子值
- 若行业均值为 NaN，回退到简单均值

### forward_fill_decay 策略
- 使用 pandas ffill(limit=max_delay) 前向填充
- 衰减因子 0.9，距离当前越远的填充值权重越低
- 最大容忍延迟期数 max_delay=5

## 四、因子数据校验

```python
_REQUIRED_FACTOR_COLUMNS = {"datetime", "asset", "factor_id", "factor_value"}
```

校验项:
1. 列完整性：必需列是否存在
2. 主键唯一：(datetime, asset, factor_id) 无重复，发现重复则去重
3. factor_id 一致性：parquet / afv.json / 目录名一致

## 五、行业数据辅助

- 正交化时，将行业分类标签转为 one-hot 虚拟变量矩阵作为基线风险暴露
- 若已有因子池（`existing_factor_library` 配置），将因子暴露合并到同一矩阵
- 最终 `risk_exposures = [行业 dummy | 因子池暴露]`，仅执行一次正交化

## 六、配置管理

Purification 配置由 `PurificationConfig` 类加载，配置来源为 `all_configs/auto_factor_evaluation/purification_config.yaml`，字段包括：
- `paths.raw_factor_base`：上游 RawFactor 基座
- `paths.temp_base`：纯化临时路径（tier0）
- `paths.pure_factor_base`：纯化因子输出基座（tier1）
- `paths.existing_factor_library`：已有因子库路径（用于正交化）
- `industry_data.path` / `industry_data.standard`：行业数据
- `purification.*`：纯化参数（填补方法、MAD 乘数等）
