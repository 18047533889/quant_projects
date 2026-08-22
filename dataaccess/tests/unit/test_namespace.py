"""
namespace 单元测试：共用账号下身份识别要靠环境变量，不能靠 $HOME 文件。
"""
from __future__ import annotations

import os

import pytest

from data_access.core import namespace as ns
from data_access.core.exceptions import ValidationError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个测试前清一下可能干扰的 env，避免本地开发者环境漏进测试。"""
    monkeypatch.delenv("QUANT_RUN_NAMESPACE", raising=False)
    monkeypatch.delenv("QUANT_OPERATOR", raising=False)
    # git_branch 有 lru_cache，要清，不然跨测试串扰
    ns._git_branch.cache_clear()


def test_namespace_explicit_env_wins(monkeypatch):
    """显式设置 QUANT_RUN_NAMESPACE 必须是最高优先。"""
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "zhangsan_2026")
    assert ns.resolve_namespace() == "zhangsan_2026"
    assert ns.is_namespace_explicit() is True


def test_namespace_env_gets_sanitized(monkeypatch):
    """#P1-63 显式坏 namespace（斜杠/空格）→ 拒绝（不静默清洗成 anon 仍当显式）。"""
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "zhang/san 01")
    with pytest.raises(ValidationError, match="非法字符"):
        ns.resolve_namespace()


def test_namespace_env_valid_passes_through(monkeypatch):
    """合法显式 namespace 原样返回，不 sanitize。"""
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "zhang.san_01-run")
    result = ns.resolve_namespace()
    assert result == "zhang.san_01-run"
    assert ns.is_namespace_explicit() is True


def test_namespace_fallback_when_not_set(monkeypatch):
    """没设 env：兜底成 anon__<branch>__<pid>，保证不 hang。"""
    result = ns.resolve_namespace()
    assert result.startswith("anon__")
    # 一定含 pid（数字），避免多窗口撞路径
    pid_str = str(os.getpid())
    assert pid_str in result
    assert ns.is_namespace_explicit() is False


def test_operator_returns_none_when_not_set(monkeypatch):
    """没设就返回 None；上游决定要不要告警。"""
    assert ns.resolve_operator() is None


def test_operator_explicit(monkeypatch):
    monkeypatch.setenv("QUANT_OPERATOR", "zhangsan")
    assert ns.resolve_operator() == "zhangsan"


def test_operator_sanitized(monkeypatch):
    """operator 的规则比 namespace 宽松：允许 @ + - . _（邮箱风格身份），
    但仍剥离空格/中文/其他可能造成解析问题的字符。"""
    monkeypatch.setenv("QUANT_OPERATOR", "zhang san@company")
    result = ns.resolve_operator()
    # 空格被替换掉
    assert " " not in result
    # @ 保留
    assert "@" in result
    assert result == "zhang_san@company"


def test_operator_strips_chinese_and_odd_chars(monkeypatch):
    """严格非 ASCII 字符（中文/表情）被替换为 _；审计日志里只留 ASCII 身份串。"""
    monkeypatch.setenv("QUANT_OPERATOR", "张三@team")
    result = ns.resolve_operator()
    assert "张" not in result
    assert "@team" in result
