"""R40 #55 —— ManifestFetchResult：manifest/token 检索的 typed 降级层级。

problem
-------
manifest / token 检索失败过去只有「返回 None / 抛异常」两种裸形态：None 同时
表示「确实没有 manifest」（合法）与「检索失败」（权限 / 超时 / 目录异常）——
调用方无法区分，容易把「找不到」当「没变化」放行（fail-open）。

design
------
:class:`ManifestFetchResult` 是 typed 枚举：

    OK / ABSENT / LOOKUP_FAILED / PERMISSION_DENIED / DEADLINE_EXCEEDED

- ``OK``                检索成功，拿到 manifest/token；
- ``ABSENT``            manifest 确实不存在（合法空状态，非错误）；
- ``LOOKUP_FAILED``     存在但读取/解析失败（损坏 / IO 异常）；
- ``PERMISSION_DENIED`` 权限不足被拒；
- ``DEADLINE_EXCEEDED`` 超过 deadline 放弃。

``classify_manifest_fetch`` 把裸异常/None 映射到 typed 结果；``fetch_manifest_typed``
包装任意检索调用返回 typed 结果。所有 manifest/token 检索路径都应返回该 typed
结果，杜绝「无法证明有没有变化」被当成「没变化」。
"""
from __future__ import annotations

import enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ManifestFetchResult(str, enum.Enum):
    """manifest/token 检索的 typed 状态。"""

    OK = "ok"
    ABSENT = "absent"
    LOOKUP_FAILED = "lookup_failed"
    PERMISSION_DENIED = "permission_denied"
    DEADLINE_EXCEEDED = "deadline_exceeded"

    @property
    def ok(self) -> bool:
        return self is ManifestFetchResult.OK

    @property
    def degraded(self) -> bool:
        return self is not ManifestFetchResult.OK


#: 常见「没有 manifest」的异常类型（mapping 到 ABSENT）。
_ABSENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    NotADirectoryError,
)


def classify_manifest_fetch(
    exc: BaseException | None,
    *,
    absent_if_missing: bool = True,
) -> ManifestFetchResult:
    """把检索失败异常/None 映射到 typed 状态。

    - ``None`` + ``absent_if_missing`` → ABSENT（检索函数以 None 表示「无」）；
    - 权限类（PermissionError / 403）→ PERMISSION_DENIED；
    - deadline 类（``DeadlineExceeded`` / ``TimeoutError``）→ DEADLINE_EXCEEDED；
    - 其余 → LOOKUP_FAILED。
    """
    if exc is None:
        return ManifestFetchResult.ABSENT if absent_if_missing else ManifestFetchResult.LOOKUP_FAILED
    # deadline 探测：异常名含 Deadline/Timeout，或类型是 DeadlineExceeded。
    exc_name = type(exc).__name__.lower()
    if "timeout" in exc_name or "deadline" in exc_name or "exceeded" in exc_name:
        return ManifestFetchResult.DEADLINE_EXCEEDED
    if isinstance(exc, PermissionError) or getattr(exc, "status_code", None) in (401, 403):
        return ManifestFetchResult.PERMISSION_DENIED
    if absent_if_missing and isinstance(exc, _ABSENT_EXCEPTIONS):
        return ManifestFetchResult.ABSENT
    return ManifestFetchResult.LOOKUP_FAILED


def fetch_manifest_typed(
    fetch_fn: Callable[[], T],
    *,
    absent_if_missing: bool = True,
) -> tuple[ManifestFetchResult, T | None]:
    """包装任意 manifest/token 检索调用，返回 ``(typed_result, value)``。

    用法::

        result, token = fetch_manifest_typed(lambda: store.manifest_version(ds))
        if result is ManifestFetchResult.OK:
            ...  # token 可用
        elif result is ManifestFetchResult.ABSENT:
            ...  # 确实没有（合法空状态）
        else:
            ...  # 检索降级（LOOKUP_FAILED / PERMISSION_DENIED / DEADLINE_EXCEEDED）
    """
    try:
        value = fetch_fn()
    except Exception as exc:  # noqa: BLE001
        return classify_manifest_fetch(exc, absent_if_missing=absent_if_missing), None
    if value is None:
        return (ManifestFetchResult.ABSENT if absent_if_missing
                else ManifestFetchResult.OK), None
    return ManifestFetchResult.OK, value


__all__ = [
    "ManifestFetchResult",
    "classify_manifest_fetch",
    "fetch_manifest_typed",
]
