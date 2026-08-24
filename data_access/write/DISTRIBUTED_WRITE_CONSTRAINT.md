# R32-P0-104: 分布式写入约束与未来路线图

## 当前状态（2026-08-12）

DataAccess write 模块当前使用 **本地 POSIX 文件锁**（`write/mutation_lock.py`）
协调同一数据集的并发写入。

### 适用场景

✅ **单 server 部署**（所有写操作在同一台机器）
✅ **单 worker 部署**（同一文件系统可见的多进程）
✅ **本地文件系统**（NFS/GPFS 等支持 POSIX 锁的共享文件系统）

### 不支持场景

❌ **多 server 写入同一 COS prefix**
   - 不同机器无法通过本地文件锁协调
   - 可能导致 lost update、generation 覆盖、partial write 等数据竞态

❌ **对象存储（COS/S3）无 POSIX 锁**
   - 对象存储不提供文件锁语义
   - 需要应用层分布式协调

## 生产部署要求

### 当前必须满足

1. **单写者约束（Single-Writer Constraint）**
   ```yaml
   # 部署配置示例
   write_mode: single_writer
   coordination: local_filesystem
   ```
   - 每个 dataset 同一时刻只有一个 server/worker 可执行写操作
   - 通过部署拓扑、任务调度或应用层编排保证单写者
   - 读操作可多 server 并发（无竞态）

2. **文件系统要求**
   - 本地文件系统 或
   - 支持 POSIX 锁的共享文件系统（NFS with lockd、GPFS 等）

3. **监控告警**
   - 监控并发写入检测（mutation_lock 冲突告警）
   - 写操作审计日志包含 server_id、worker_id

## 未来增强路线（R32-P0-104 完整实现）

### Phase 1: COS 条件写入（ETag-based Optimistic Locking）

```python
# write/distributed_coordination.py (future)
class COSConditionalWriter:
    """基于 ETag/If-Match 的乐观锁写入。"""
    
    def atomic_put_if_match(
        self,
        key: str,
        data: bytes,
        expected_etag: str | None,
    ) -> str:
        """条件 PUT：只有 current ETag == expected_etag 时写入成功。
        
        Returns:
            新对象的 ETag
        
        Raises:
            ConflictError: ETag 不匹配（其他 writer 已更新）
        """
```

### Phase 2: Generation Epoch + Fencing Token

```python
class GenerationCoordinator:
    """基于 generation epoch 的分布式 fencing。"""
    
    def acquire_generation_lock(
        self,
        dataset: str,
        generation_id: str,
        fencing_token: int,
    ) -> GenerationLease:
        """获取 generation 写入租约（带 fencing token）。"""
```

### Phase 3: 外部分布式锁（Redis/etcd）

```python
class DistributedMutationLock:
    """基于 Redis/etcd 的分布式互斥锁。"""
    
    def __enter__(self):
        self._lease = self._backend.acquire_lock(
            key=f"dataaccess:write:{self.dataset}",
            timeout=self.timeout,
        )
```

## 开发者指南

### 如何判断是否满足单写者约束

```python
# 部署时验证
from data_access.write.deployment import validate_write_topology

validate_write_topology(
    deployment_config,
    require_single_writer=True,  # production 必须
)
```

### 违反约束的症状

- **写操作偶发失败**（mutation_lock timeout）
- **数据丢失**（后写覆盖先写，未合并）
- **Generation 混乱**（publish 后发现旧 generation）
- **Manifest 损坏**（并发写入同一 manifest 文件）

### 应急响应

1. 立即停止所有写操作
2. 检查 audit log 确认冲突 server/worker
3. 修复部署拓扑确保单写者
4. 验证数据一致性后恢复写入

## 测试

### 当前测试覆盖

- `tests/unit/test_mutation_lock.py`: 本地锁正确性
- `tests/contract/test_upsert.py`: 单进程写入原子性
- `tests/contract/test_publish.py`: 发布流程完整性

### 缺失测试（需要分布式环境）

- ❌ 多 server 并发写同一 dataset
- ❌ COS 条件写入竞态
- ❌ Network partition 期间写入行为

## 参考

- AWS S3 条件写入: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-requests.html
- etcd distributed lock: https://etcd.io/docs/v3.5/learning/api/#lock-api
- Fencing tokens: Martin Kleppmann, "Designing Data-Intensive Applications", Ch. 8

---

**维护**: DataAccess Core Team  
**最后更新**: 2026-08-12  
**关联**: R32-P0-104, R32-P0-102, R32-P0-103
