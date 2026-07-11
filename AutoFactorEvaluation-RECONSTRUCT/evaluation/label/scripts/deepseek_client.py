"""DeepSeek 语义因子贴标的正式客户端边界。

模块: evaluation.label
职责:
1. 构造正式语义贴标 prompt。
2. 调用真实 DeepSeek API。
3. 强制校验响应结构，并显式处理重试和退避行为。

本文件是 label 模块唯一的网络边界。
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
from urllib.parse import urlparse

from .logging_utils import get_label_logger
from .schemas import DeepSeekConfig, JsonDict


LOGGER = get_label_logger()


class _TransientAPIError(RuntimeError):
    """DeepSeek 临时传输异常或服务端异常。"""


class DeepSeekClient:
    """用于标准化语义贴标的正式 DeepSeek 客户端。

    已实现的评审要求:
    1. v1_review.md 4.3-(4): 使用真实 DeepSeek API，不走 mock 路径。
    2. v1_review.md 4.3-(15): 仅对临时 API 异常执行重试。
    3. v1_review.md 4.3-(16): 解析失败直接暴露。
    4. v1_review.md 4.3-(17): prompt 使用严格的白名单 JSON 结构。
    5. v1_review.md 4.3-(18): 连接超时和读取超时分别处理。
    """

    def __init__(self, settings: DeepSeekConfig) -> None:
        self.settings = settings

    def resolve_api_key(self) -> str:
        """从显式配置或环境变量解析 API key。"""

        if self.settings.api_key:
            return self.settings.api_key
        return os.getenv(self.settings.api_key_env, "")

    def build_messages(
        self,
        expression: str,
        charts_summary: JsonDict,
        label_registry_excerpt: JsonDict,
    ) -> list[dict[str, str]]:
        """构造语义贴标 prompt。

        当前逻辑:
        1. 模型接收当前标准标签集合。
        2. 如果已有标签中存在匹配项，应选择最合适的已有标签作为 primary_label，
           并返回空 suggested_new_label 和 similar_candidates。
        3. 如果没有合适的已有标签，应构造一个新的标准 snake_case 标签，
           令 primary_label == suggested_new_label，并在相关时将相似已有标签返回到
           similar_candidates。
        """

        if not isinstance(expression, str):
            raise TypeError("expression must be a string for DeepSeek prompting")
        if not isinstance(charts_summary, dict):
            raise TypeError("charts_summary must be a JSON object")
        if not isinstance(label_registry_excerpt, dict):
            raise TypeError("label_registry_excerpt must be a JSON object")

        user_payload = {
            "expression": expression,
            "charts_summary": charts_summary,
            "label_registry": label_registry_excerpt,
            "output_contract": {
                "primary_label": "string",
                "confidence": "float in [0, 1]",
                "explanation": "string",
                "suggested_new_label": "string",
                "similar_candidates": ["string"],
            },
        }

        system_prompt = (
            "You are a quantitative factor semantic labeling service. "
            "Use the provided current label set as the canonical reference. "
            "When one existing label already fits the factor semantics, choose the "
            "single best existing label as primary_label and leave "
            "suggested_new_label empty. In that case similar_candidates should "
            "also be empty. "
            "When none of the existing labels fit well enough, create one new "
            "canonical English snake_case label that is not already in the current "
            "label set. In that case set primary_label equal to suggested_new_label. "
            "Return similar_candidates only as existing labels that are close to "
            "the new label; otherwise return an empty list. "
            "Keep the explanation short and factual. "
            "Output exactly one JSON object with only the fields "
            "primary_label, confidence, explanation, suggested_new_label, "
            "similar_candidates."
        )

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
            },
        ]

    def request_semantic_tags(
        self,
        expression: str,
        charts_summary: JsonDict,
        label_registry_excerpt: JsonDict,
    ) -> JsonDict:
        """调用 DeepSeek API，并返回一个通过校验的语义标签结果。"""

        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError("DeepSeek API key is missing")
        LOGGER.info(
            "deepseek_request_start model=%s expression_length=%s registry_labels=%s",
            self.settings.model,
            len(expression),
            len(label_registry_excerpt.get("labels", [])),
        )

        messages = self.build_messages(
            expression=expression,
            charts_summary=charts_summary,
            label_registry_excerpt=label_registry_excerpt,
        )
        request_payload = json.dumps(
            {
                "model": self.settings.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")

        response_body = self._http_post_with_retry(request_payload, api_key)
        response_payload = json.loads(response_body)
        if not isinstance(response_payload, dict):
            raise ValueError("DeepSeek API response root must be a JSON object")

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("DeepSeek API response choices are missing or empty")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("DeepSeek API response choice must be a JSON object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("DeepSeek API response message is missing")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek API response content must be a non-empty string")

        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("DeepSeek semantic result must be a JSON object")
        validated_result = self._validate_semantic_result(result)
        LOGGER.info(
            "deepseek_request_done primary_label=%s suggested_new_label=%s "
            "similar_candidates=%s",
            validated_result["primary_label"],
            validated_result["suggested_new_label"],
            validated_result["similar_candidates"],
        )
        return validated_result

    def _http_post_with_retry(self, payload: bytes, api_key: str) -> str:
        # http请求异常处理
        parsed_base_url = urlparse(self.settings.base_url)
        if parsed_base_url.scheme != "https":
            raise ValueError("DeepSeek base_url must use https")
        if parsed_base_url.hostname is None:
            raise ValueError("DeepSeek base_url must contain a hostname")

        request_path = parsed_base_url.path.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt_index in range(self.settings.max_retries + 1):
            if attempt_index > 0:
                delay_seconds = self._retry_delay(attempt_index - 1)
                LOGGER.info(
                    "deepseek_retry_wait attempt=%s delay_seconds=%s",
                    attempt_index + 1,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

            LOGGER.info(
                "deepseek_http_attempt attempt=%s max_attempts=%s host=%s path=%s",
                attempt_index + 1,
                self.settings.max_retries + 1,
                parsed_base_url.hostname,
                request_path,
            )
            connection = http.client.HTTPSConnection(
                parsed_base_url.hostname,
                port=parsed_base_url.port,
                timeout=self.settings.connect_timeout_seconds,
            )
            try:
                connection.request(
                    "POST",
                    request_path,
                    body=payload,
                    headers=headers,
                )
                response = connection.getresponse()
                self._set_read_timeout(response)

                response_body = response.read().decode("utf-8")
                status_code = response.status
                LOGGER.info(
                    "deepseek_http_response attempt=%s status_code=%s body_size=%s",
                    attempt_index + 1,
                    status_code,
                    len(response_body),
                )

                if status_code == 429 or 500 <= status_code < 600:
                    raise _TransientAPIError(
                        f"DeepSeek API returned transient status {status_code}: "
                        f"{response_body[:300]}"
                    )
                if not 200 <= status_code < 300:
                    raise RuntimeError(
                        f"DeepSeek API returned non-success status {status_code}: "
                        f"{response_body[:300]}"
                    )
                return response_body
            except (_TransientAPIError, socket.timeout, TimeoutError, OSError) as exc:
                last_error = exc
                LOGGER.warning(
                    "deepseek_http_transient_failure attempt=%s max_attempts=%s "
                    "error=%s",
                    attempt_index + 1,
                    self.settings.max_retries + 1,
                    exc,
                )
                continue
            finally:
                connection.close()

        if last_error is None:
            raise RuntimeError("DeepSeek API call failed without an explicit error")
        raise RuntimeError(
            f"DeepSeek API call failed after {self.settings.max_retries + 1} attempts"
        ) from last_error

    def _set_read_timeout(self, response: http.client.HTTPResponse) -> None:
        raw_stream = response.fp
        if raw_stream is None:
            raise RuntimeError("DeepSeek HTTP response stream is missing")
        if hasattr(raw_stream, "raw") and hasattr(raw_stream.raw, "_sock"):
            raw_stream.raw._sock.settimeout(self.settings.read_timeout_seconds)
            return
        if hasattr(raw_stream, "_sock"):
            raw_stream._sock.settimeout(self.settings.read_timeout_seconds)
            return
        raise RuntimeError("DeepSeek HTTP response socket is not accessible")

    def _retry_delay(self, retry_index: int) -> int:
        if retry_index < len(self.settings.backoff_seconds):
            return self.settings.backoff_seconds[retry_index]
        return self.settings.backoff_seconds[-1]

    def _validate_semantic_result(
        self,
        result: JsonDict,
    ) -> JsonDict:
        """
        验证返回的json字段
        """
        primary_label = result.get("primary_label")
        if not isinstance(primary_label, str) or not primary_label:
            raise ValueError("DeepSeek result.primary_label must be a non-empty string")

        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("DeepSeek result.confidence must be numeric")

        explanation = result.get("explanation", "")
        if not isinstance(explanation, str):
            raise ValueError("DeepSeek result.explanation must be a string")

        suggested_new_label = result.get("suggested_new_label", "")
        if not isinstance(suggested_new_label, str):
            raise ValueError("DeepSeek result.suggested_new_label must be a string")

        similar_candidates = result.get("similar_candidates", [])
        if not isinstance(similar_candidates, list):
            raise ValueError("DeepSeek result.similar_candidates must be a list")
        for value in similar_candidates:
            if not isinstance(value, str):
                raise ValueError(
                    "DeepSeek result.similar_candidates items must be strings"
                )

        return {
            "primary_label": primary_label,
            "confidence": float(confidence),
            "explanation": explanation,
            "suggested_new_label": suggested_new_label,
            "similar_candidates": list(similar_candidates),
        }
