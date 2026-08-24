"""R40 #82: service version resolution (metadata first, pyproject fallback)."""

from __future__ import annotations

import factor_engine.service.app as app


def test_resolve_version_uses_package_metadata(monkeypatch):
    calls: list[str] = []

    def fake_version(name):
        calls.append(name)
        return "9.9.9"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert app._resolve_version() == "9.9.9"
    assert calls == ["factor-engine"]


def test_resolve_version_falls_back_on_metadata_error(monkeypatch):
    def boom(name):
        raise RuntimeError("package not installed")

    monkeypatch.setattr("importlib.metadata.version", boom)
    v = app._resolve_version()
    # fallback reads pyproject.toml (version 0.3.1) — never the empty sentinel
    assert v and v != "0.0.0"
