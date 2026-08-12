"""R32-P0-027 —— CloudErrorClassifier：云服务商 SDK 错误分类器。

problem
-------
manifest fetch / COS LIST / HEAD 失败过去只返回 None 或抛泛型异常——调用方无法区分
「404 not-found」（合法空状态）与「403 permission-denied」（权限问题）与「timeout」
（可重试），容易把「找不到」当「没变化」fail-open。

design
------
:class:`CloudErrorClassifier` 把 botocore / tencent-cloud / httpfs SDK 的原始异常
映射到 typed 错误码：

    - 401/403 → AUTH_FAILED（权限不足）
    - 404 → NOT_FOUND（对象/manifest 不存在）
    - 409/412 → PRECONDITION_FAILED（版本条件失败）
    - 429 → THROTTLED（限流）
    - 5xx → SERVER_ERROR（服务端错误）
    - timeout / ReadTimeout / ConnectTimeout → DEADLINE_EXCEEDED
    - ExpiredToken / InvalidToken → CREDENTIALS_INVALID
    - DNS / TLS / SSL → NETWORK_ERROR
    - 其他 → UNKNOWN

输出 :class:`CloudErrorCode`：
    - ``code``：typed 错误码
    - ``retryable``：是否可重试（5xx/timeout/throttle=True，4xx 多数=False）
    - ``retry_after``：重试延迟（429 Retry-After header）
    - ``security_sensitive``：是否安全敏感（credential 相关不记日志明文）
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class CloudErrorKind(str, enum.Enum):
    """云服务商错误分类（R32-P0-027）。"""

    AUTH_FAILED = "auth_failed"
    NOT_FOUND = "not_found"
    PRECONDITION_FAILED = "precondition_failed"
    THROTTLED = "throttled"
    SERVER_ERROR = "server_error"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CREDENTIALS_INVALID = "credentials_invalid"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CloudErrorCode:
    """typed 云错误码 + 重试策略（R32-P0-027）。"""

    kind: CloudErrorKind
    retryable: bool
    retry_after: float | None = None
    security_sensitive: bool = False
    original_code: str | None = None


class CloudErrorClassifier:
    """云服务商 SDK 错误分类器（R32-P0-027）。"""

    @staticmethod
    def classify(exc: BaseException | None) -> CloudErrorCode:
        """分类云 SDK 异常 → typed 错误码 + 重试策略。

        支持：
            - botocore (boto3)：ClientError / NoCredentialsError / ...
            - tencent-cloud-sdk：TencentCloudSDKException
            - httpfs / requests：HTTPError / Timeout / ...
        """
        if exc is None:
            return CloudErrorCode(kind=CloudErrorKind.UNKNOWN, retryable=False)

        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        # ---- botocore (boto3) ----
        if exc_type == "ClientError":
            response = getattr(exc, "response", {})
            error = response.get("Error", {})
            code = error.get("Code", "")
            http_status = response.get("ResponseMetadata", ).get("HTTPStatusCode", 0)
            original_code = code or str(http_status)

            if code in ("NoSuchKey", "NoSuchBucket", "NotFound") or http_status == 404:
                return CloudErrorCode(
                    kind=CloudErrorKind.NOT_FOUND,
                    retryable=False,
                    original_code=original_code,
                )
            if code in ("AccessDenied", "Forbidden") or http_status in (401, 403):
                return CloudErrorCode(
                    kind=CloudErrorKind.AUTH_FAILED,
                    retryable=False,
                    security_sensitive=True,
                    original_code=original_code,
                )
            if code == "PreconditionFailed" or http_status in (409, 412):
                return CloudErrorCode(
                    kind=CloudErrorKind.PRECONDITION_FAILED,
                    retryable=False,
                    original_code=original_code,
                )
            if code in ("RequestLimitExceeded", "Throttling", "TooManyRequests") or http_status == 429:
                retry_after = _extract_retry_after(response)
                return CloudErrorCode(
                    kind=CloudErrorKind.THROTTLED,
                    retryable=True,
                    retry_after=retry_after,
                    original_code=original_code,
                )
            if http_status >= 500:
                return CloudErrorCode(
                    kind=CloudErrorKind.SERVER_ERROR,
                    retryable=True,
                    original_code=original_code,
                )

        # ---- botocore credential errors ----
        if exc_type in ("NoCredentialsError", "PartialCredentialsError", "InvalidToken", "ExpiredToken"):
            return CloudErrorCode(
                kind=CloudErrorKind.CREDENTIALS_INVALID,
                retryable=False,
                security_sensitive=True,
                original_code=exc_type,
            )

        # ---- tencent cloud SDK ----
        if exc_type == "TencentCloudSDKException":
            code = getattr(exc, "code", "")
            if "notfound" in code.lower() or "nosuch" in code.lower():
                return CloudErrorCode(
                    kind=CloudErrorKind.NOT_FOUND,
                    retryable=False,
                    original_code=code,
                )
            if "accessdenied" in code.lower() or "unauthorized" in code.lower():
                return CloudErrorCode(
                    kind=CloudErrorKind.AUTH_FAILED,
                    retryable=False,
                    security_sensitive=True,
                    original_code=code,
                )
            if "throttl" in code.lower() or "limitexceed" in code.lower():
                return CloudErrorCode(
                    kind=CloudErrorKind.THROTTLED,
                    retryable=True,
                    original_code=code,
                )

        # ---- timeout ----
        if "timeout" in exc_type.lower() or "timeout" in exc_str:
            return CloudErrorCode(
                kind=CloudErrorKind.DEADLINE_EXCEEDED,
                retryable=True,
                original_code=exc_type,
            )

        # ---- network / DNS / TLS ----
        if any(
            kw in exc_type.lower()
            for kw in ("connection", "network", "dns", "ssl", "tls", "certificate")
        ):
            return CloudErrorCode(
                kind=CloudErrorKind.NETWORK_ERROR,
                retryable=True,
                original_code=exc_type,
            )

        # ---- requests / httpx ----
        if hasattr(exc, "response"):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return CloudErrorCode(kind=CloudErrorKind.NOT_FOUND, retryable=False, original_code=str(status))
            if status in (401, 403):
                return CloudErrorCode(
                    kind=CloudErrorKind.AUTH_FAILED,
                    retryable=False,
                    security_sensitive=True,
                    original_code=str(status),
                )
            if status == 429:
                return CloudErrorCode(kind=CloudErrorKind.THROTTLED, retryable=True, original_code=str(status))
            if status and status >= 500:
                return CloudErrorCode(kind=CloudErrorKind.SERVER_ERROR, retryable=True, original_code=str(status))

        # ---- fallback ----
        return CloudErrorCode(kind=CloudErrorKind.UNKNOWN, retryable=False, original_code=exc_type)


def _extract_retry_after(response: dict[str, Any]) -> float | None:
    """从 HTTP 响应提取 Retry-After header（秒）。"""
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass
    return None


__all__ = [
    "CloudErrorKind",
    "CloudErrorCode",
    "CloudErrorClassifier",
]
