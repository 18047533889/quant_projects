# 多服务器部署配置指南

## 概述

FactorEngine 从 2026-08-13 起支持**自适应资源配置**：引擎启动时自动探测服务器的内存和 CPU 资源，根据实际硬件调整所有资源相关的配置参数。

设计目标：
- **零配置启动**：小内存服务器（8GB）和大内存服务器（512GB）使用同一份代码，无需手动调整参数
- **自适应缩放**：DuckDB/Polars 线程数、内存限制、批处理大小等参数根据硬件自动缩放
- **环境变量覆盖**：对于特殊场景，支持通过环境变量手动覆盖任何参数

---

## 自动探测机制

### 探测内容

引擎启动时探测以下资源：

1. **内存**：通过 `psutil.virtual_memory().total` 获取系统总内存（GB）
2. **CPU 核心数**：优先使用物理核心数 `psutil.cpu_count(logical=False)`，回退到逻辑核心数的一半
3. **容器限制**（如果在容器/cgroup 中运行）：`psutil` 会自动识别 cgroup 内存限制

### 优先级顺序

配置参数的优先级（从高到低）：

1. **环境变量**：`DUCKDB_THREADS`、`POLARS_MAX_THREADS` 等显式覆盖
2. **函数参数**：`get_adaptive_config(force_memory_gb=X)` 显式指定
3. **自动探测**：从系统读取实际资源

---

## 各档位服务器配置

以下表格为**真实生成值**（通过执行 `runtime/adaptive_config.py` 生成）：

### 小型服务器（8GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 8 GB | |
| CPU 核心数 | 4 | |
| **DuckDB** | | |
| `duckdb_threads` | 4 | 线程数（不超过 CPU 核心数） |
| `duckdb_memory_limit` | 4096MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 4 | 等于 CPU 核心数 |
| `polars_streaming_chunk_size` | 50,000 | 小内存环境用小块 |
| **批处理** | | |
| `compile_chunk_size` | 258 | 编译批量大小 |
| `dag_chunk_size` | 516 | DAG 处理批量 |
| `batch_size` | 51,639 | 通用批量大小 |
| **内存块** | | |
| `block_abs_max_bytes` | 1.03 GB | 单个内存块最大值 |
| `cache_size_bytes` | 1.03 GB | 缓存大小 |
| `streaming_threshold_bytes` | 1.03 GB | 流式处理阈值 |
| `hard_memory_limit_bytes` | 4.13 GB | ResourceBroker 硬限制 |
| `safe_envelope_bytes` | 1.03 GB | OOM 安全阈值 |
| `compile_budget_bytes` | 1.03 GB | 编译阶段内存预算 |
| **并发** | | |
| `max_workers_io` | 6 | IO 密集型线程池（1.5× CPU） |
| `max_workers_compute` | 4 | CPU 密集型线程池 |

### 中小型服务器（16GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 16 GB | |
| CPU 核心数 | 8 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | |
| `duckdb_memory_limit` | 8192MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 8 | |
| `polars_streaming_chunk_size` | 100,000 | 大内存环境用大块 |
| **批处理** | | |
| `compile_chunk_size` | 365 | |
| `dag_chunk_size` | 730 | |
| `batch_size` | 73,029 | |
| **内存块** | | |
| `block_abs_max_bytes` | 1.46 GB | |
| `cache_size_bytes` | 1.46 GB | |
| `streaming_threshold_bytes` | 1.46 GB | |
| `hard_memory_limit_bytes` | 5.84 GB | |
| `safe_envelope_bytes` | 1.46 GB | |
| `compile_budget_bytes` | 1.46 GB | |
| **并发** | | |
| `max_workers_io` | 12 | |
| `max_workers_compute` | 8 | |

### 中型服务器（30GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 30.5 GB | 缩放基准 |
| CPU 核心数 | 8 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | 根据 benchmark 固定为 8 |
| `duckdb_memory_limit` | 15616MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 8 | |
| `polars_streaming_chunk_size` | 100,000 | |
| **批处理** | | |
| `compile_chunk_size` | 504 | 基准值 500 |
| `dag_chunk_size` | 1,008 | |
| `batch_size` | 100,829 | |
| **内存块** | | |
| `block_abs_max_bytes` | 2.02 GB | 基准值 2GB |
| `cache_size_bytes` | 2.02 GB | |
| `streaming_threshold_bytes` | 2.02 GB | |
| `hard_memory_limit_bytes` | 8.06 GB | 约 27% 系统内存 |
| `safe_envelope_bytes` | 2.02 GB | |
| `compile_budget_bytes` | 2.02 GB | |
| **并发** | | |
| `max_workers_io` | 12 | 不超过 16 |
| `max_workers_compute` | 8 | |

### 中大型服务器（64GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 64 GB | |
| CPU 核心数 | 16 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | 固定为 8 |
| `duckdb_memory_limit` | 32768MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 16 | |
| `polars_streaming_chunk_size` | 100,000 | |
| **批处理** | | |
| `compile_chunk_size` | 730 | |
| `dag_chunk_size` | 1,460 | |
| `batch_size` | 146,059 | |
| **内存块** | | |
| `block_abs_max_bytes` | 2.92 GB | |
| `cache_size_bytes` | 2.92 GB | |
| `streaming_threshold_bytes` | 2.92 GB | |
| `hard_memory_limit_bytes` | 11.68 GB | |
| `safe_envelope_bytes` | 2.92 GB | |
| `compile_budget_bytes` | 2.92 GB | |
| **并发** | | |
| `max_workers_io` | 16 | 上限封顶 |
| `max_workers_compute` | 16 | |

### 大型服务器（128GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 128 GB | |
| CPU 核心数 | 32 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | 固定为 8 |
| `duckdb_memory_limit` | 65536MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 32 | |
| `polars_streaming_chunk_size` | 100,000 | |
| **批处理** | | |
| `compile_chunk_size` | 1,032 | |
| `dag_chunk_size` | 2,065 | |
| `batch_size` | 206,559 | |
| **内存块** | | |
| `block_abs_max_bytes` | 4.13 GB | |
| `cache_size_bytes` | 4.13 GB | |
| `streaming_threshold_bytes` | 4.13 GB | |
| `hard_memory_limit_bytes` | 16.52 GB | |
| `safe_envelope_bytes` | 4.13 GB | |
| `compile_budget_bytes` | 4.13 GB | |
| **并发** | | |
| `max_workers_io` | 16 | 上限封顶 |
| `max_workers_compute` | 32 | |

### 超大型服务器（256GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 256 GB | |
| CPU 核心数 | 64 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | 固定为 8 |
| `duckdb_memory_limit` | 131072MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 64 | |
| `polars_streaming_chunk_size` | 100,000 | |
| **批处理** | | |
| `compile_chunk_size` | 1,460 | |
| `dag_chunk_size` | 2,921 | |
| `batch_size` | 292,118 | |
| **内存块** | | |
| `block_abs_max_bytes` | 5.84 GB | |
| `cache_size_bytes` | 5.84 GB | |
| `streaming_threshold_bytes` | 5.84 GB | |
| `hard_memory_limit_bytes` | 23.36 GB | |
| `safe_envelope_bytes` | 5.84 GB | |
| `compile_budget_bytes` | 5.84 GB | |
| **并发** | | |
| `max_workers_io` | 16 | 上限封顶 |
| `max_workers_compute` | 64 | |

### 特大型服务器（512GB 内存）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **系统资源** | | |
| 内存 | 512 GB | |
| CPU 核心数 | 96 | |
| **DuckDB** | | |
| `duckdb_threads` | 8 | 固定为 8 |
| `duckdb_memory_limit` | 262144MB | 50% 系统内存 |
| **Polars** | | |
| `polars_threads` | 96 | |
| `polars_streaming_chunk_size` | 100,000 | |
| **批处理** | | |
| `compile_chunk_size` | 2,000 | 上限封顶 |
| `dag_chunk_size` | 4,131 | |
| `batch_size` | 413,118 | |
| **内存块** | | |
| `block_abs_max_bytes` | 8.26 GB | |
| `cache_size_bytes` | 8.26 GB | |
| `streaming_threshold_bytes` | 8.26 GB | |
| `hard_memory_limit_bytes` | 33.04 GB | |
| `safe_envelope_bytes` | 8.26 GB | |
| `compile_budget_bytes` | 8.26 GB | |
| **并发** | | |
| `max_workers_io` | 16 | 上限封顶 |
| `max_workers_compute` | 96 | |

---

## 容器/Cgroup 部署

### 容器内存限制识别

当引擎在 Docker/Kubernetes/LXC 容器中运行时，`psutil` 会自动识别 cgroup 内存限制，而非宿主机总内存。

**示例**：宿主机 128GB 内存，容器限制 16GB，引擎会探测到 16GB 并按小型服务器配置。

### 验证探测结果

在容器中运行以下命令验证引擎探测到的资源：

```bash
cd /home/shw/quant_projects/factor_engine
python3 -c "
from runtime.adaptive_config import get_adaptive_config
config = get_adaptive_config()
print(f'探测到: {config.system_memory_gb:.1f}GB 内存, {config.system_cpu_cores} CPU 核心')
print(f'DuckDB: {config.duckdb_threads} 线程, {config.duckdb_memory_limit}')
print(f'Polars: {config.polars_threads} 线程')
print(f'批处理: compile_chunk={config.compile_chunk_size}, batch={config.batch_size}')
print(f'硬内存限制: {config.hard_memory_limit_bytes / (1024**3):.2f}GB')
"
```

**健康的输出示例**（16GB 容器）：

```
探测到: 16.0GB 内存, 8 CPU 核心
DuckDB: 8 线程, 8192MB
Polars: 8 线程
批处理: compile_chunk=365, batch=73029
硬内存限制: 5.84GB
```

如果探测到宿主机的全部内存（而非容器限制），说明容器未正确设置内存限制，需要在容器启动时显式指定：

```bash
# Docker
docker run --memory=16g ...

# Kubernetes
resources:
  limits:
    memory: 16Gi
```

---

## 环境变量覆盖

对于需要手动调优的场景，支持以下环境变量覆盖自动配置：

### DuckDB 配置

| 环境变量 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `DUCKDB_THREADS` | int | DuckDB 线程数 | `export DUCKDB_THREADS=4` |
| `DUCKDB_MEMORY_LIMIT_MB` | int | DuckDB 内存限制（MB） | `export DUCKDB_MEMORY_LIMIT_MB=8192` |

**使用场景**：
- 共享服务器上限制 DuckDB 资源占用
- DuckDB 版本升级后线程数最佳实践变化

### Polars 配置

| 环境变量 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `POLARS_MAX_THREADS` | int | Polars 线程数 | `export POLARS_MAX_THREADS=16` |

**使用场景**：
- 与其他 Polars 进程共享 CPU 资源
- 避免超线程干扰（例如物理核心 32 个，显式设为 32 而非 64）

### 批处理配置

| 环境变量 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `COMPILE_CHUNK_SIZE` | int | 编译批量大小 | `export COMPILE_CHUNK_SIZE=200` |

**使用场景**：
- 编译阶段 OOM：减小批量（例如从 500 降到 200）
- 小任务频繁提交：增大批量减少编译开销

### 并发配置

| 环境变量 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `MAX_WORKERS_IO` | int | IO 密集型线程池大小 | `export MAX_WORKERS_IO=8` |
| `MAX_WORKERS_COMPUTE` | int | CPU 密集型线程池大小 | `export MAX_WORKERS_COMPUTE=16` |

**使用场景**：
- 存储 IOPS 受限：减小 IO 线程池避免过度并发
- CPU 密集型任务占用过多核心：减小计算线程池为其他进程留出资源

### 完整覆盖示例

针对 64GB 服务器手动调优（共享环境，限制资源占用）：

```bash
#!/bin/bash
# 限制 DuckDB 资源
export DUCKDB_THREADS=4
export DUCKDB_MEMORY_LIMIT_MB=16384  # 16GB

# 限制 Polars 并发
export POLARS_MAX_THREADS=8

# 减小批量避免内存峰值
export COMPILE_CHUNK_SIZE=300

# 限制并发工作线程
export MAX_WORKERS_IO=8
export MAX_WORKERS_COMPUTE=8

# 启动引擎
cd /home/shw/quant_projects/factor_engine
python3 -m your_entry_point
```

---

## 新服务器验证步骤

### 1. 打印探测到的配置

```bash
cd /home/shw/quant_projects/factor_engine
python3 -c "
from runtime.adaptive_config import get_adaptive_config
import json
config = get_adaptive_config()
print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
"
```

### 2. 检查关键指标

健康配置应满足：

- `system.memory_gb`：与 `free -h` 或容器限制一致
- `system.cpu_cores`：与 `lscpu | grep "Core(s) per socket"` 一致（物理核心）
- `duckdb.memory_limit_mb`：约为系统内存的 50%
- `duckdb.threads`：不超过 8（benchmark 最佳值）
- `polars.threads`：等于 CPU 核心数
- `memory.hard_memory_limit_bytes`：约为系统内存的 25-30%

### 3. 冷启动测试

运行一个轻量级因子计算任务，观察：

```bash
# 启动引擎并观察日志
python3 your_script.py 2>&1 | grep -E "(Adaptive config|DuckDB|Polars|auto_configure)"
```

预期日志输出：

```
INFO: Adaptive config: 64.0GB RAM, 16 cores -> DuckDB 8t/32768MB, Polars 16t, compile_chunk=730, hard_limit=11.7GB
INFO: auto_configure: 64.0GB memory, 16 CPU cores detected
INFO:   DuckDB: 8 threads, 32768MB
INFO:   Batch size: 146059, DAG chunk: 1460
INFO:   Workers: 16 compute, 16 I/O
```

### 4. 压力测试

运行大批量因子计算（1000+ 因子），监控：

- **内存使用**：`top` 或 `htop` 观察 RSS，不应超过系统内存的 80%
- **CPU 利用率**：应看到 DuckDB 和 Polars 按配置的线程数并发
- **OOM killer**：`dmesg | grep -i oom` 应无输出

---

## 故障排查

### 问题 1：OOM（Out of Memory）despite auto-config

**症状**：
- 进程被 OOM killer 杀死
- `dmesg` 显示 `Out of memory: Killed process`

**诊断**：

```bash
# 检查实际内存使用峰值
python3 -c "
from runtime.adaptive_config import get_adaptive_config
config = get_adaptive_config()
total_gb = config.system_memory_gb
hard_limit_gb = config.hard_memory_limit_bytes / (1024**3)
duckdb_gb = config.duckdb_memory_limit_mb / 1024
print(f'系统总内存: {total_gb:.1f}GB')
print(f'DuckDB 限制: {duckdb_gb:.1f}GB')
print(f'ResourceBroker 硬限制: {hard_limit_gb:.1f}GB')
print(f'理论峰值（DuckDB + ResourceBroker）: {duckdb_gb + hard_limit_gb:.1f}GB')
print(f'安全余量: {total_gb - (duckdb_gb + hard_limit_gb):.1f}GB')
"
```

**解决方案**：

1. **容器内存限制过小**：增加容器内存限制，或显式设置环境变量
   ```bash
   export DUCKDB_MEMORY_LIMIT_MB=4096  # 降低 DuckDB 内存
   ```

2. **多进程并发**：多个引擎实例同时运行，总内存超限
   - 为每个实例显式分配内存预算
   - 使用进程池而非多进程并发

3. **数据倾斜**：某些因子计算需要远超平均的内存
   - 识别高内存因子，单独处理
   - 减小 `COMPILE_CHUNK_SIZE` 避免同时编译多个高内存因子

### 问题 2：引擎使用核心数过少

**症状**：
- `top` 显示 CPU 利用率远低于预期（例如 16 核服务器只用到 200%）
- DuckDB 查询很慢

**诊断**：

```bash
python3 -c "
from runtime.adaptive_config import get_adaptive_config
config = get_adaptive_config()
print(f'DuckDB 线程数: {config.duckdb_threads}')
print(f'Polars 线程数: {config.polars_threads}')
print(f'计算线程池: {config.max_workers_compute}')
"
```

**解决方案**：

1. **DuckDB 线程数固定为 8**：这是 benchmark 最佳值，无需增加
   - 如果确实需要更多并发，显式覆盖：
     ```bash
     export DUCKDB_THREADS=16
     ```

2. **Polars 未充分利用**：检查是否有全局线程池限制
   ```bash
   export POLARS_MAX_THREADS=32  # 显式指定
   ```

3. **任务本身串行**：某些算子不支持并行，属于正常现象

### 问题 3：DuckDB 频繁 spilling to disk

**症状**：
- 日志中出现 `DuckDB spilling to disk`
- 磁盘 I/O 飙升，查询变慢

**诊断**：

```bash
python3 -c "
from runtime.adaptive_config import get_adaptive_config
config = get_adaptive_config()
print(f'DuckDB 内存限制: {config.duckdb_memory_limit}')
print(f'系统总内存: {config.system_memory_gb:.1f}GB')
print(f'DuckDB 占比: {config.duckdb_memory_limit_mb / (config.system_memory_gb * 1024) * 100:.1f}%')
"
```

**解决方案**：

1. **增加 DuckDB 内存限制**（牺牲其他组件的内存）：
   ```bash
   export DUCKDB_MEMORY_LIMIT_MB=49152  # 从 32GB 增加到 48GB（64GB 服务器）
   ```

2. **查询本身内存需求过大**：
   - 检查 SQL 中是否有大表 JOIN 或 GROUP BY
   - 考虑分批处理或增加过滤条件

3. **升级服务器内存**：如果频繁 spilling 是常态，说明当前服务器配置不足

### 问题 4：配置未生效

**症状**：
- 设置了环境变量，但日志显示仍使用默认值

**诊断**：

```bash
# 检查环境变量是否生效
python3 -c "
import os
from runtime.adaptive_config import get_adaptive_config
config = get_adaptive_config()
print('环境变量:')
for key in ['DUCKDB_THREADS', 'DUCKDB_MEMORY_LIMIT_MB', 'POLARS_MAX_THREADS']:
    print(f'  {key} = {os.environ.get(key, \"(未设置)\")}')
print('\\n实际配置:')
print(f'  duckdb_threads = {config.duckdb_threads} (来源: {config.config_source[\"duckdb_threads\"]})')
print(f'  duckdb_memory_limit_mb = {config.duckdb_memory_limit_mb} (来源: {config.config_source[\"duckdb_memory_limit_mb\"]})')
print(f'  polars_threads = {config.polars_threads} (来源: {config.config_source[\"polars_threads\"]})')
"
```

**解决方案**：

1. **环境变量未导出**：确保使用 `export` 而非直接赋值
   ```bash
   export DUCKDB_THREADS=4  # 正确
   DUCKDB_THREADS=4         # 错误（仅当前 shell）
   ```

2. **配置被缓存**：重启 Python 进程或调用 `reset_global_adaptive_config()`
   ```python
   from runtime.adaptive_config import reset_global_adaptive_config
   reset_global_adaptive_config()
   ```

3. **环境变量名拼写错误**：检查变量名大小写和拼写

---

## 实际使用的模块

以下模块已接入自适应配置（从代码自动发现）：

- `runtime/resource_autopilot.py`：资源自动驾驶
- `runtime/memory_budget_allocator.py`：内存预算分配
- `runtime/resource_broker.py`：资源代理（使用 `block_abs_max_bytes`）
- `runtime/multibackend/polars_lazy_fusion.py`：Polars 流式融合（使用 `streaming_threshold_bytes`）
- `runtime/multibackend/concurrent_token_manager.py`：并发令牌管理（使用 `hard_memory_limit_bytes`）
- `runtime/multibackend/cse_native_representation.py`：CSE 原生表示（使用 `cache_size_bytes`）
- `planner/physical_lowerer.py`：物理计划下降器
- `planner/native_fusion.py`：原生融合（使用 `compile_budget_bytes`）
- `backend/polars_thread_config.py`：Polars 线程配置（使用 `polars_streaming_chunk_size`）

这些模块在运行时自动调用 `get_global_adaptive_config()` 获取配置，无需手动传参。

---

## 附录：完整字段列表

### 系统探测结果

- `system_memory_gb`: float — 系统总内存（GB）
- `system_cpu_cores`: int — CPU 物理核心数

### DuckDB 配置

- `duckdb_threads`: int — 线程数（环境变量：`DUCKDB_THREADS`）
- `duckdb_memory_limit`: str — 内存限制字符串（如 "15GB"）
- `duckdb_memory_limit_mb`: int — 内存限制（MB，环境变量：`DUCKDB_MEMORY_LIMIT_MB`）

### Polars 配置

- `polars_threads`: int — 线程数（环境变量：`POLARS_MAX_THREADS`）
- `polars_streaming_chunk_size`: int — 流式块大小

### 批处理配置

- `compile_chunk_size`: int — 编译批量（环境变量：`COMPILE_CHUNK_SIZE`）
- `dag_chunk_size`: int — DAG 处理批量
- `batch_size`: int — 通用批量大小

### 内存块配置

- `block_abs_max_bytes`: int — 单个块最大内存
- `cache_size_bytes`: int — 缓存大小
- `streaming_threshold_bytes`: int — 流式处理阈值
- `hard_memory_limit_bytes`: int — ResourceBroker 硬限制
- `safe_envelope_bytes`: int — OOM 安全阈值
- `compile_budget_bytes`: int — 编译阶段内存预算

### 并发配置

- `max_workers_io`: int — IO 线程池（环境变量：`MAX_WORKERS_IO`）
- `max_workers_compute`: int — CPU 线程池（环境变量：`MAX_WORKERS_COMPUTE`）

### 配置来源

- `config_source`: dict[str, str] — 每个字段的来源（"adaptive" 或 "env"）
