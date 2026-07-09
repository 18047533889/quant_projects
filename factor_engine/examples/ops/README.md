# Factor Engine 运维模板

## 队列 Worker（systemd）

1. 复制 [`fe-queue-worker.service`](fe-queue-worker.service) 到 `/etc/systemd/system/`
2. 修改 `WorkingDirectory`、`PYTHONPATH`、`--queue-root`
3. 启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fe-queue-worker
sudo journalctl -u fe-queue-worker -f
```

## K8s Job

见 [`fe-queue-worker-job.yaml`](fe-queue-worker-job.yaml)。部署前替换镜像、PVC 与代码挂载路径。

## Bucket 切读 YAML 补丁

```bash
python3 data_access/scripts/emit_bucket_cutover_patch.py \
  --dataset ashare_stock_minute \
  --target-root /data/bucket/minute
```

## 生产严格 Polars

```bash
export FACTOR_ENGINE_PRODUCTION_STRICT_POLARS=1  # pandas fallback → fail
```

## Bucket 迁移 Playbook

```bash
python3 data_access/scripts/run_bucket_migration.py \
  --dataset ashare_stock_minute \
  --target-root /path/to/bucket_layout \
  --json

# 实际迁移 + 校验
python3 data_access/scripts/run_bucket_migration.py \
  --dataset ashare_stock_minute \
  --target-root /path/to/bucket_layout \
  --execute

# 灰度切读（不改 yaml）
export DATA_ACCESS_READ_ROOT_ASHARE_STOCK_MINUTE=/path/to/bucket_layout
```

## 事件驱动增量

```bash
python3 run_pipeline.py queue enqueue-event \
  --dataset us_stocks --column close --updated-date 2026-07-09 \
  --queue-root /var/lib/factor_engine/queue

python3 run_pipeline.py deps --lake-root /path/to/lake --column close
```
