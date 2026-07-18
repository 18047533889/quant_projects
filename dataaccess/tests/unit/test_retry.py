"""
retry 单元测试：瞬时 IO 错重试、业务错不重试。
"""
from __future__ import annotations

import pytest

from data_access.core.exceptions import DataError, ValidationError
from data_access.core.retry import retry_io


def test_no_retry_on_success():
    calls = [0]

    @retry_io(max_attempts=3, backoff_sec=0.0)
    def fn():
        calls[0] += 1
        return "ok"

    assert fn() == "ok"
    assert calls[0] == 1


def test_retries_on_oserror_then_succeeds():
    calls = [0]

    @retry_io(max_attempts=3, backoff_sec=0.0, gc_before_retry=False)
    def flaky():
        calls[0] += 1
        if calls[0] < 3:
            raise OSError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls[0] == 3


def test_gives_up_after_max_attempts():
    calls = [0]

    @retry_io(max_attempts=2, backoff_sec=0.0, gc_before_retry=False)
    def always_fail():
        calls[0] += 1
        raise OSError("dead")

    with pytest.raises(OSError, match="dead"):
        always_fail()
    assert calls[0] == 2


def test_validation_error_not_retried():
    """业务错误立即抛，不重试（重试没意义还拖慢失败反馈）。"""
    calls = [0]

    @retry_io(max_attempts=5, backoff_sec=0.0)
    def bad_input():
        calls[0] += 1
        raise ValidationError("bad param")

    with pytest.raises(ValidationError):
        bad_input()
    assert calls[0] == 1  # 只跑一次


def test_data_error_not_retried():
    calls = [0]

    @retry_io(max_attempts=5, backoff_sec=0.0)
    def no_data():
        calls[0] += 1
        raise DataError("empty")

    with pytest.raises(DataError):
        no_data()
    assert calls[0] == 1
