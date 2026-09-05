# -*- coding: utf-8 -*-
"""
data_access.write.publish_errors —— 对象存储发布的统一错误层级。

对应 UPSTREAM_FIX_PLAN 问题二「推荐错误类型」：

    class RemotePublishError(DataError): ...
    class RemoteWriteVerificationError(RemotePublishError): ...
    class StaleWriterError(RemotePublishError): ...
    class RemotePointerUpdateError(RemotePublishError): ...

语义（data_access 错误类型三分类之一，见 ``data_access/core/exceptions.py``）：
- ``RemotePublishError`` 是对象存储发布路径所有错误的基类，挂在 ``DataError``
  之下 —— 上层「发布失败 = 数据操作失败」一把抓时能命中，同时保留精确子类供
  写告警/重试策略区分：

    - ``RemoteWriteVerificationError``：artifact 上传后 HEAD/回读校验失败
      （size 不匹配 / sha256 不匹配 / head 缺失）。CURRENT 未受影响，可安全重试；
    - ``StaleWriterError``：并发 CURRENT 下 stale writer 被拒绝（fencing-epoch
      单调性 / ETag 条件写不匹配）。**不重试**（fail-closed，禁止覆盖已晋升
      CURRENT）—— 与 t8_publish 重试器（``publish_t8_artifacts_retry``）的
      catch-and-raise 语义一致；
    - ``RemotePointerUpdateError``：CURRENT 指针条件更新失败且**不属于 stale
      writer 场景**（如对象存储后端不支持条件写 / 元数据读失败后无法证明
      指针状态）。由调用方决定重试或人工介入。

错误被移动的历史：``StaleWriterError`` 最初定义在
``data_access/write/object_store_generation_publisher.py``；为让所有发布路径
共享同一层级，它被迁移到这里并继承 ``RemotePublishError``。module 级
re-export 保留 ``object_store_generation_publisher.StaleWriterError`` 名字，
既有 ``from data_access.write.object_store_generation_publisher import
StaleWriterError`` 的调用方（tests / t8_publish）无需改动。
"""
from __future__ import annotations

from data_access.core.exceptions import DataError

__all__ = [
    "RemotePublishError",
    "RemoteWriteVerificationError",
    "StaleWriterError",
    "RemotePointerUpdateError",
]


class RemotePublishError(DataError):
    """对象存储发布失败基类（CURRENT 未变或未知；见各子类语义）。

    挂在 ``DataError`` 下：发布失败本质是「数据没写对」，上层按数据错误处理。
    需要区分具体场景（校验失败 / stale / 指针更新失败）时 catch 子类。
    """


class RemoteWriteVerificationError(RemotePublishError):
    """artifact 上传后校验失败（size / sha256 / head 缺失）。

    CURRENT 未受影响（本代次尚未晋升）——调用方可以安全地清理本代次并重试。
    """


class StaleWriterError(RemotePublishError):
    """并发 CURRENT 下 stale writer 被拒绝的显式信号。

    判定依据有两层，任一命中即拒绝：

    - **应用层 fencing（monotonic fencing-epoch）**：begin 时快照的 CURRENT
      epoch 落后于 finish 时的 CURRENT epoch（期间已有别的 writer 晋升）→ 本
      writer stale，拒绝覆盖（``resolve_stale`` / ``expect_sole_writer`` 语义）；
    - **对象存储层条件写（compare-and-swap）**：CURRENT.json 更新必须携带旧
      ETag（``If-Match``）或 ``If-None-Match``；底层 ETag 不匹配 → 存储服务
      拒绝本次覆盖，同样抛本错误。

    旧 writer 不能覆盖新 writer 的 CURRENT。**不重试**：重试只会重放同一 stale
    覆盖，被拒是终态（fail-closed）。
    """


class RemotePointerUpdateError(RemotePublishError):
    """CURRENT 指针条件更新失败，且不属于 stale-writer 场景。

    典型场景：对象存储后端不支持条件写、执行条件写前 HEAD 读不到对象当前
    元数据、服务端返回非 412 的更新错误。指针状态无法证明 → 由调用方决定
    重试或人工介入（绝不在状态未知时继续覆盖）。
    """
