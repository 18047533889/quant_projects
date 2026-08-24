import openai
import asyncio
import os
from typing import Any


class LLMQuotaExhaustedError(RuntimeError):
    """API 额度/余额耗尽，应停止连续挖掘，勿无限重试。"""


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    keywords = (
        "insufficient",
        "quota",
        "balance",
        "billing",
        "payment required",
        "exceeded your current quota",
        "欠费",
        "余额不足",
        "额度",
    )
    if any(k in msg for k in keywords):
        return True
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    if status in (401, 403) and any(k in msg for k in ("balance", "quota", "insufficient", "billing")):
        return True
    return False


def _quota_max_retries() -> int:
    try:
        return max(1, int(os.getenv("LLM_QUOTA_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def completions_with_backoff(client, **kwargs):
    """带限次重试的 chat.completions；额度耗尽立即抛 LLMQuotaExhaustedError。"""
    max_retries = _quota_max_retries()
    transient_retries = max(1, int(os.getenv("LLM_TRANSIENT_MAX_RETRIES", "5")))
    delay = 2.0
    last_exc: BaseException | None = None
    quota_hits = 0
    transient_hits = 0

    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc):
                quota_hits += 1
                if quota_hits >= max_retries:
                    raise LLMQuotaExhaustedError(
                        f"LLM quota/balance exhausted after {quota_hits} retries: {exc}"
                    ) from exc
                print(f"[llm] quota-like error ({quota_hits}/{max_retries}): {exc}")
            elif isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
                transient_hits += 1
                if transient_hits >= transient_retries:
                    # 持续 429 也可能是日额度；按额度耗尽处理，避免 24h 空转
                    if isinstance(exc, openai.RateLimitError) or _is_quota_error(exc):
                        raise LLMQuotaExhaustedError(
                            f"LLM rate-limit/quota persisted after {transient_hits} retries: {exc}"
                        ) from exc
                    raise
                print(f"[llm] transient error ({transient_hits}/{transient_retries}): {type(exc).__name__}: {exc}")
            else:
                raise
            import time

            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)

    assert last_exc is not None
    raise last_exc


async def dispatch_openai_chat_requests(
    client,
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
) -> list[str]:
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


class OpenAIModel:
    def __init__(self, model_name, stop_words, max_new_tokens) -> None:
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens

    def chat_generate(self, client, system_prompt: str, user_prompt: str, temperature=0.0):
        response = completions_with_backoff(
            client,
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=1,
            stop=self.stop_words or None,
        )
        generated_text = (response.choices[0].message.content or "").strip()
        finish_reason = response.choices[0].finish_reason
        return generated_text, finish_reason

    def generate(self, client, input_string, temperature=0.0):
        return self.chat_generate(client, input_string, temperature)

    def batch_chat_generate(self, client, messages_list, temperature=0.0):
        open_ai_messages_list = []
        system_prompt = (
            "You are a helpful assistant. Make sure you carefully and fully understand "
            "the details of user's requirements before you start solving the problem."
        )
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                client,
                open_ai_messages_list,
                self.model_name,
                temperature,
                self.max_new_tokens,
                1.0,
                self.stop_words,
            )
        )
        finish_reason = [x.choices[0].finish_reason.strip() for x in predictions]
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, client, messages_list, temperature=0.0):
        return self.batch_chat_generate(client, messages_list, temperature)
