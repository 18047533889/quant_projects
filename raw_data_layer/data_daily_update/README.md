# 每日增量数据更新系统

本目录包含一个为每日增量更新市场数据而设计的全自动化系统。该系统旨在实现健壮、高效且易于维护的目标。

## 核心理念

- **单一入口**: 整个流程由一个独立的shell脚本启动，使其极易与`cron`等调度器集成。
- **配置即代码**: 所有数据源和系统配置都在Python执行脚本中明确定义，确保了版本控制和透明度。
- **生产就绪**: 该系统为无人值守的日常执行而构建，具备全面的日志记录、错误处理和自包含的依赖管理。

---

## 📁 文件结构与功能

### 核心Python模块

| 文件 | 功能 |
| :--- | :--- |
| `production_update_runner.py` | **主入口**. 定义所有配置，并通过调用主控制器来编排更新流程。这是唯一应该被直接执行的Python脚本。 |
| `async_incremental_update.py` | **主控制器**. 系统的“大脑”。它协调所有其他组件，管理批处理，并确保端到端的工作流程。 |
| `async_universal_fetcher.py` | **通用数据获取器**. 系统的“手臂”。负责从各种数据源（REST API, S3等）实际下载数据，并内置了重试、分页和增量获取的逻辑。 |
| `cursor_manager.py` | **游标管理器**. 系统的“记忆”。它读取和写入每个数据源的进度标记（游标），以确保只从上一个检查点开始获取数据。 |
| `async_reporter.py` | **报告生成器**. 生成一个最终的JSON格式总结报告，详细说明每个数据源更新的结果。 |
| `quality_controller.py` | **数据质量控制器**. 在数据写入磁盘之前，对下载的数据执行基本检查。 |
| `config_validator_v3_compatible.py` | **配置验证器**. 确保传递给主控制器的配置结构是有效的。 |

### 自动化脚本

| 文件 | 功能 |
| :--- | :--- |
| `scripts/daily_update_scheduler.sh` | **核心调度器**. 这是**唯一需要运行的脚本**。它负责设置环境（如`PYTHONPATH`）、执行Python运行器、捕获日志并处理成功或失败的通知。 |
| `scripts/install_cron_jobs.sh` | **Cron安装器**. 一个用于将`daily_update_scheduler.sh`自动添加到系统`crontab`中以进行计划执行的实用工具脚本。 |

---

## 🔄 工作流程

系统遵循一个清晰、线性和自动化的流程：

1.  **触发**: `cron`任务在预定时间执行`scripts/daily_update_scheduler.sh`。
2.  **环境准备**: shell脚本将`PYTHONPATH`设置为项目根目录，并为当前运行准备一个日志文件。
3.  **执行**: 脚本使用正确的Anaconda Python解释器运行`production_update_runner.py`。
4.  **初始化**:
    - Python运行器定义所有系统和数据源配置。
    - 它实例化`AsyncIncrementalUpdateMaster`。
5.  **核心处理**:
    - `Master`使用`CursorManager`读取最后已知的游标位置。
    - 它将所有已启用数据源的更新任务以并行批处理的方式分派给`AsyncUniversalFetcher`。
    - 每个`Fetcher`根据游标下载新数据，处理分页，并将数据写入`lfl_workspace/data/raw/`目录。
6.  **报告与收尾**:
    - `Master`收集结果，并使用`AsyncReporter`在`lfl_workspace/data/reports/`中生成摘要报告。
    - 然后，它指示`CursorManager`为下一次运行保存新的游标位置。
7.  **完成**:
    - Python脚本以状态码退出（0表示成功，1表示失败）。
    - `daily_update_scheduler.sh`脚本记录最终结果并清理旧的日志文件。

---

## 🚀 如何运行

### 手动执行

要手动运行一个完整的更新周期，只需从终端执行调度程序脚本。这对于测试或即时数据检索非常有用。

```bash
bash /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/daily_update_scheduler.sh
```

所有的输出和日志都将保存到 `/home/yluel/share/projects/quantsociety_backend_project/user_workspace/lfl_workspace/logs/` 目录中。

### 自动调度

要每天自动运行更新，请使用`install_cron_jobs.sh`脚本。这会将必要的条目添加到系统的`crontab`中。

1.  **检查计划（可选）**: 检查`install_cron_jobs.sh`脚本以查看默认执行时间（例如，每天凌晨4点）。
2.  **安装cron任务**:
    ```bash
    bash /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/install_cron_jobs.sh
    ```
3.  **验证安装**:
    ```bash
    crontab -l
    ```
    您应该能看到一个指向`daily_update_scheduler.sh`的新行。

---

## 🔧 维护与修改

### 添加/修改数据源

1.  打开 `production_update_runner.py`。
2.  找到 `get_data_sources_config()` 函数。
3.  在 `data_sources` 列表中添加或编辑条目。请遵循现有格式。

### 调整系统参数

1.  打开 `production_update_runner.py`。
2.  找到 `get_system_config()` 函数。
3.  修改 `max_retry_attempts`、`parallel_workers` 等参数。