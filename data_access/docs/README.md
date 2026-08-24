# data_access 文档

> 仓库：https://github.com/HKUST-QUANT-SOCIETY/data_access  
> Python：`import data_access`｜包版本 **0.3.1**

| 文档 | 读者 | 说明 |
|------|------|------|
| **[用户使用手册.md](用户使用手册.md)** | 因子组、风控、合作方 | **对外首选**：安装、读/写、COS、**HTTP 服务（§10）**、compute_and_write |
| **[../README.md](../README.md)** | 平台组 / 内部开发 | 模块定位、API 摘要、组织仓安装路径、部署入口 |
| **[COS语义与PIT契约.md](COS语义与PIT契约.md)** | 平台 / 数据工程 | COS 面板、PIT、事件时间轴契约 |
| **[SOURCE_SYNC.md](SOURCE_SYNC.md)** | 平台 | 与源仓同步说明 |
| **[../CHANGELOG.md](../CHANGELOG.md)** | 全员 | 版本变更 |
| **[../config/README.md](../config/README.md)** | 登记数据集的人 | `datasets.yaml` 字段说明 |
| **[../ops/README.md](../ops/README.md)** | 运维 | stats 刷新等脚本 |
| **[../deploy/README.md](../deploy/README.md)** | 运维 | Docker / K8s / systemd |
| **[../config/datasets.yaml](../config/datasets.yaml)** | 全员 | 数据集登记表（机器可读） |

**转发给外部同事**：发 [用户使用手册.md](用户使用手册.md)。HTTP 只用服务时直接看手册 **§10**。
