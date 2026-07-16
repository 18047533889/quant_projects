# `storage/materialize` — 落盘辅助模块

[`../materializer.py`](../materializer.py) 的企业级落盘拆分子模块：分区策略、写入目标、DQ 元数据等。

主入口仍是 **`FactorMaterializer`**（`storage/materializer.py`），本目录放可复用的落盘策略与格式定义。

## 典型流程

```
FactorEngine.run() → Series
  → materializer.write() → 本地 lake 目录 / factor_lake_staging upsert
  → lake_publish.publish_factor_lake() → factor_lake (published)
```

## 相关

- [`../write_targets.py`](../write_targets.py) — local / staging / clickhouse 写目标
- [`../lake_publish.py`](../lake_publish.py) — staging → published
- [`../../runtime/materialize_service.py`](../../runtime/materialize_service.py) — 编排层
