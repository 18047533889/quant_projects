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

## 数据集 stats 刷新（data_access）

```bash
cd quant_projects
PYTHONPATH=. python3 dataaccess/ops/refresh_dataset_stats.py --dataset ashare_stock_daily
```

## 生产严格 Polars

```bash
export FACTOR_ENGINE_PRODUCTION_STRICT_POLARS=1  # pandas fallback → fail
```

## 读路径灰度（不改 datasets.yaml）

```bash
export DATA_ACCESS_READ_ROOT_ASHARE_STOCK_MINUTE=/path/to/alternate_layout
```

## 事件驱动增量

```bash
python3 run_pipeline.py queue enqueue-event \
  --dataset us_stocks --column close --updated-date 2026-07-09 \
  --queue-root /var/lib/factor_engine/queue

python3 run_pipeline.py deps --lake-root /path/to/lake --column close
```
