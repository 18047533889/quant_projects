# AutoFactorEvaluation-RECONSTRUCT

这是自动化因子评估系统的**独立可部署版本**。该目录内直接包含经过同步和校验的 `factor_engine` 与 `data_access`，因此最终交付时只上传本目录即可，不再依赖 `quant_projects` monorepo 的其他目录。

## 目录边界

```text
AutoFactorEvaluation-RECONSTRUCT/
├── factor_engine/                 # 内嵌的 FactorEngine 运行时
├── data_access/                   # 内嵌的统一数据访问层与 datasets.yaml
├── integrations/quant_platform.py # 业务侧唯一平台桥接层
├── platform_bootstrap.py          # 确定性运行时选择与路径引导
├── embedded_platform_manifest.json# 源提交、文件数、大小与哈希
├── standalone.py                  # doctor / DSL 校验 / 本地数据执行 / pipeline 入口
├── pipeline.py                    # Gateway→资产化→纯化→评估全流程
└── scripts/sync_embedded_platform.py
```

默认运行模式为 `bundled`：只能加载本目录中的 `factor_engine` 和 `data_access`。这避免了服务器上恰好存在其他版本时，系统静默加载错误代码。

## 安装

Python 要求：3.10 及以上。

最小计算与评估环境：

```bash
cd AutoFactorEvaluation-RECONSTRUCT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

需要 LLM 标注、VectorBT 或完整扩展功能时：

```bash
pip install -r requirements.txt
```

## 部署前检查

```bash
python standalone.py doctor
```

检查内容包括：

- 内嵌 FactorEngine 与 DataAccess 文件是否齐全；
- `embedded_platform_manifest.json` 是否存在；
- DataAccess 是否使用本目录的 `config/datasets.yaml`；
- Python 依赖是否完整；
- `api.dsl_parser`、`runtime.engine`、`backend.factory`、`data_access` 是否确实从本目录加载；
- 使用合成行情执行一条真实 DSL，验证 parser、analyzer、planner、backend 与算子链路。

返回 `status=PASS` 后再启动正式任务。

## 常用命令

### 1. 校验 DSL

```bash
python standalone.py validate "rank(ts_mean(close, 20))"
```

生产门禁校验：

```bash
python standalone.py validate \
  "rank(ts_mean(close, 20))" \
  --run-mode production
```

输出包含递归 lookback、时序/截面算子识别、引用字段及物理计划摘要。

### 2. 在本地长表上计算因子

输入应包含时间列、标的列和公式引用字段。默认支持常见字段名 `datetime/TradeDate` 与 `asset/Symbol/Ticker`。

```bash
python standalone.py evaluate-frame \
  "ts_mean(close, 5) / close - 1" \
  market.parquet \
  factor_result.parquet \
  --name mom_5d
```

输出统一为：

```text
datetime | asset | value | factor_name | data_snapshot_id
```

### 3. 启动现有全流程

```bash
python standalone.py pipeline --all --ev-workers 8
```

也可继续使用：

```bash
python pipeline.py --all
```

`pipeline.py` 启动时会做平台预检，除非显式传入 `--skip-platform-check`。

## 数据配置

DataAccess 默认固定读取：

```text
AutoFactorEvaluation-RECONSTRUCT/data_access/config/datasets.yaml
```

可通过环境变量覆盖：

```bash
export DATA_ACCESS_CONFIG=/absolute/path/to/datasets.yaml
```

行情、因子湖、COS、本地镜像等实际根路径仍通过 `datasets.yaml` 中的环境变量和运行参数控制。代码目录与数据目录必须分离，不要把真实行情数据打包进本项目。

AutoFactorEvaluation 自身配置仍由：

```bash
export AFVCONFIG=/absolute/path/to/config_dir
```

控制，目录中需要存在 `config.yaml`。可参考 `config.yaml.example`。

## 平台版本选择

默认：

```bash
export AUTOFAC_PLATFORM_MODE=bundled
```

开发环境需要临时使用外部平台时：

```bash
export AUTOFAC_PLATFORM_MODE=external
export AUTOFAC_PLATFORM_ROOT=/path/to/quant_projects
```

仅开发同步场景可使用：

```bash
export AUTOFAC_PLATFORM_MODE=auto
```

生产环境不要使用 `auto`，否则可能因机器目录结构不同产生版本漂移。

## 从 monorepo 同步最新平台代码

在 `quant_projects` 仓库根目录执行：

```bash
python AutoFactorEvaluation-RECONSTRUCT/scripts/sync_embedded_platform.py
```

脚本会：

1. 原子替换目录内的 `factor_engine` 与 `data_access`；
2. 排除测试、notebook、benchmark、缓存、压缩包和 `workspace_data` 运行日志等非发布文件；
3. 校验关键文件，包括最新的 `semantic_hardening.py`；
4. 生成 `embedded_platform_manifest.json`；
5. 记录源提交、文件数量、字节数和整棵目录哈希。

仅验证当前内嵌副本是否被人工修改：

```bash
python scripts/sync_embedded_platform.py --check
```

## 生产约束

- 因子候选只执行 FactorEngine DSL，不执行任意 Python 因子代码；
- 正式行情读取、快照 lineage、staging upsert 和 publish 全部经过 DataAccess；
- daily DSL surface 与 production-denied 策略保持一致；
- FactorEngine/DataAccess 更新必须重新同步、跑测试、重新生成 manifest 后发布；
- 不允许业务模块自行扫描 Parquet、拼接平台目录或绕过统一桥接层。

## 建议发布流程

```bash
python scripts/sync_embedded_platform.py --check
python standalone.py doctor
pytest -q tests
```

随后仅打包或上传整个 `AutoFactorEvaluation-RECONSTRUCT` 目录。不要只复制部分 Python 文件，否则 manifest、算子证据文件、DataAccess registry 与运行时代码可能失配。
