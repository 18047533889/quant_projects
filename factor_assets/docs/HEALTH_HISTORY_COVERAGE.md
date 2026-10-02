# 按覆盖区间重放因子健康状态

调用 `SnapshotManager.create_snapshot` 时，你可以传入 `health_history_coverage`，声明健康事件的覆盖区间和各因子的起始状态。库据此检查状态链，再计算查询时刻的健康状态。生命周期筛选和里程碑计算沿用原接口。

## 参数与边界

`HealthHistoryCoverage(covered_from, covered_through, baseline_health_by_factor)` 接受两个 ISO-8601 时间字符串和一组因子基线。基线值必须是 `HealthState`；你可以传映射或 `(factor_id, HealthState)` 对序列。库复制这些值并保存只读映射，拒绝重复因子 ID、无效状态和倒置区间。

基线表示 `covered_from` **之前**的状态。库包含起点、终点处的事件，并要求查询时刻位于这个闭区间。带时区的时间按 UTC 比较；兼容接口中的无时区时间按 UTC 解释。

库先应用注册时间、因子 ID、生命周期、campaign 和 tags 筛选，然后检查匹配资产的基线。你无须为被筛除的资产提供基线。库按时间排序匹配因子的健康事件；同一时刻的事件保留输入顺序。每条事件的 `health_from` 必须等于前一状态。

库检查整个声明区间的状态链，包括查询时刻之后的边，但仅将不晚于查询时刻的 `health_to` 写入结果。区间之外的事件不参与这条状态链。

## 调用示例

```python
from factor_assets.contracts.lifecycle import HealthState
from factor_assets.registry.health_history import HealthHistoryCoverage
from factor_assets.registry.snapshots import SnapshotManager, SnapshotQuery

coverage = HealthHistoryCoverage(
    covered_from="2026-01-01T00:00:00Z",
    covered_through="2026-01-05T00:00:00Z",
    baseline_health_by_factor={"factor_a": HealthState.ACTIVE},
)
result = SnapshotManager().create_snapshot(
    assets, events,
    SnapshotQuery(as_of_timestamp="2026-01-03T00:00:00Z", factor_id="factor_a"),
    health_history_coverage=coverage,
)
```

若 `events` 包含 1 月 2 日的 `ACTIVE → DEPRECATED` 和 1 月 4 日的 `DEPRECATED → ACTIVE`，1 月 3 日结果为 `DEPRECATED`。若你误删 1 月 2 日事件，只保留后一个转换，库会拒绝从 `ACTIVE` 基线接上 `DEPRECATED → ACTIVE`。起点处发生的转换也需要从声明基线接续。

## 调用方仍需保证事件来源完整

调用方声明覆盖范围。库能够发现缺少基线、越界查询和不连续的状态链；若调用方遗漏一整个回到原状态的转换环，剩余事件仍可能与基线一致。区间和基线参数无法证明外部事件库没有遗漏。你应从可信事件存储取得覆盖声明和事件，不要根据当前 ACTIVE 状态猜测基线。

`health_history_coverage=None` 保留原调用行为。旧模式从首个健康转换的 `health_from` 推断基线；当前非 ACTIVE 且缺少健康事件的匹配资产会报错。当前 ACTIVE 或仅有历史后缀的资产仍存在历史完整性歧义。需要检查覆盖声明的调用方应传入上述参数。

这个接口提供健康状态重放校验。你仍需按数据可用时间管理回测输入，覆盖声明也不替代事件存储的版本、哈希或授权校验。
