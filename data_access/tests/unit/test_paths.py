"""
paths 单元测试：env 展开、默认值语法、白名单越界拒绝。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data_access.exceptions import ValidationError
from data_access.paths import (
    PathAuthorizer,
    canonicalize,
    expand_env,
    path_is_under,
)


def test_expand_env_simple_var(monkeypatch):
    monkeypatch.setenv("DA_TEST_ROOT", "/tmp/foo")
    assert expand_env("${DA_TEST_ROOT}/bar") == "/tmp/foo/bar"


def test_expand_env_default_when_unset(monkeypatch):
    """${VAR:-default} 在 VAR 没设时用 default，这是本模块相对 os.path.expandvars 的加强。"""
    monkeypatch.delenv("DA_MISSING_VAR", raising=False)
    assert expand_env("${DA_MISSING_VAR:-/opt/fallback}/data") == "/opt/fallback/data"


def test_expand_env_default_ignored_when_set(monkeypatch):
    monkeypatch.setenv("DA_SET_VAR", "/real/path")
    assert expand_env("${DA_SET_VAR:-/opt/fallback}/data") == "/real/path/data"


def test_expand_env_tilde(tmp_path, monkeypatch):
    """~ 要展开成 $HOME。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert expand_env("~/data") == str(tmp_path / "data")


def test_path_is_under_trivial():
    assert path_is_under(Path("/a/b/c"), Path("/a/b"))
    assert path_is_under(Path("/a/b"), Path("/a/b"))
    assert not path_is_under(Path("/a/b"), Path("/a/c"))


def test_authorizer_accepts_under_root(tmp_path):
    auth = PathAuthorizer([tmp_path])
    child = tmp_path / "sub" / "file.parquet"
    # 文件不存在也能 resolve；只校验前缀
    assert auth.resolve_and_authorize(child) == canonicalize(child)


def test_authorizer_rejects_escape(tmp_path):
    """越界路径：拒绝并抛 ValidationError，错误信息含白名单提示。"""
    auth = PathAuthorizer([tmp_path])
    with pytest.raises(ValidationError) as exc_info:
        auth.resolve_and_authorize("/etc/passwd")
    assert "路径越界" in str(exc_info.value)
    assert str(tmp_path) in str(exc_info.value)


def test_authorizer_rejects_dotdot_escape(tmp_path):
    """用 .. 试图逃出 —— canonicalize 先消解，再比较，逃不掉。"""
    auth = PathAuthorizer([tmp_path / "inner"])
    escape_attempt = tmp_path / "inner" / ".." / ".." / "other"
    with pytest.raises(ValidationError):
        auth.resolve_and_authorize(escape_attempt)


def test_authorizer_deduplicates_roots(tmp_path):
    """同一个根被多次登记应该只存一份。"""
    auth = PathAuthorizer([tmp_path, tmp_path, tmp_path / "."])
    assert len(auth.allowed_roots) == 1
