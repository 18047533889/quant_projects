"""R40 #76/#77/#78/#79/#83/#135: runtime.config remediation tests."""

from __future__ import annotations

import pytest

from runtime.config import (
    CONFIG_SCHEMA_VERSION,
    DataSourceConfig,
    FactorDefinitionConfig,
    FactorEngineConfig,
    MaterializationConfig,
    TypedDataSourceOptions,
    _validate_config_schema_version,
    canonical_config_hash,
    load_config,
)


def _base_config(*, password: str | None = None, dsn: str | None = None) -> FactorEngineConfig:
    options: dict = {"dataset": "d"}
    if dsn is not None:
        options["url"] = dsn
    return FactorEngineConfig(
        factor=FactorDefinitionConfig(name="f", expr="close"),
        data_source=DataSourceConfig(type="data_access", options=options),
        materialization=(
            MaterializationConfig(clickhouse_password=password)
            if password is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# #76: canonical_config_hash uses json.dumps but json wasn't imported
# ---------------------------------------------------------------------------


def test_canonical_config_hash_deterministic():
    c1 = _base_config()
    h1 = canonical_config_hash(c1)
    h2 = canonical_config_hash(c1)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hexdigest
    assert all(ch in "0123456789abcdef" for ch in h1)


# ---------------------------------------------------------------------------
# #83: secrets/DSN credentials never enter the canonical hash
# ---------------------------------------------------------------------------


def test_canonical_hash_same_for_different_passwords():
    a = _base_config(password="pass-aaa")
    b = _base_config(password="pass-bbb")
    assert canonical_config_hash(a) == canonical_config_hash(b)


def test_canonical_hash_same_for_different_dsn_credentials():
    a = _base_config(dsn="postgres://user:oldpass@dbhost/db")
    b = _base_config(dsn="postgres://user:newpass@dbhost/db")
    assert canonical_config_hash(a) == canonical_config_hash(b)


def test_canonical_hash_differs_when_nonsecret_changes():
    a = _base_config(password="same")
    b = _base_config(password="same")
    # change the dataset (non-secret) -> hash must change
    b = FactorEngineConfig(
        factor=FactorDefinitionConfig(name="f", expr="close"),
        data_source=DataSourceConfig(type="data_access", options={"dataset": "OTHER"}),
        materialization=MaterializationConfig(clickhouse_password="same"),
    )
    assert canonical_config_hash(a) != canonical_config_hash(b)


# ---------------------------------------------------------------------------
# #78: schema_version strict validation
# ---------------------------------------------------------------------------


def test_schema_version_rejects_bool():
    with pytest.raises(ValueError, match="bool"):
        _validate_config_schema_version({"schema_version": True})
    with pytest.raises(ValueError, match="bool"):
        _validate_config_schema_version({"schema_version": False})


def test_schema_version_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        _validate_config_schema_version({"schema_version": -1})


def test_schema_version_rejects_zero():
    with pytest.raises(ValueError, match="0"):
        _validate_config_schema_version({"schema_version": 0})


def test_schema_version_missing_requires_explicit():
    # 显式 legacy 策略关闭时，缺失 schema_version 必须报错（不再静默当 current）。
    with pytest.raises(ValueError, match="missing"):
        _validate_config_schema_version({}, allow_missing=False)
    # 显式 legacy 策略开启时放行（向后兼容历史配置）。
    _validate_config_schema_version({}, allow_missing=True)


def test_schema_version_older_requires_migration(monkeypatch):
    monkeypatch.setattr("runtime.config.CONFIG_SCHEMA_VERSION", 2)
    with pytest.raises(ValueError, match="older"):
        _validate_config_schema_version({"schema_version": 1})


def test_schema_version_current_accepted():
    _validate_config_schema_version({"schema_version": CONFIG_SCHEMA_VERSION})


# ---------------------------------------------------------------------------
# #77: schema_version must be validated AFTER profile merge
# ---------------------------------------------------------------------------


def test_schema_version_validated_after_profile_merge(tmp_path, monkeypatch):
    monkeypatch.setattr("runtime.config._PROFILES_DIR", tmp_path)
    (tmp_path / "inject.yaml").write_text(
        "schema_version: 99\n", encoding="utf-8"
    )
    cfg = tmp_path / "factor.yaml"
    cfg.write_text(
        "profile: inject\n"
        "factor:\n  name: f\n  expr: close\n"
        "data_source:\n  type: data_access\n  dataset: d\n",
        encoding="utf-8",
    )
    # raw payload 无 schema_version（merge 前校验会放行）；profile 注入 99 ——
    # merge 后校验必须拒绝。
    with pytest.raises(ValueError, match="newer"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# #79: data_access / label top-level keys are rejected (fail-fast)
# ---------------------------------------------------------------------------


def test_data_access_label_rejected(tmp_path):
    for key in ("data_access", "label"):
        cfg = tmp_path / f"factor_{key}.yaml"
        cfg.write_text(
            "factor:\n  name: f\n  expr: close\n"
            "data_source:\n  type: data_access\n  dataset: d\n"
            f"{key}:\n  read_auto: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown key"):
            load_config(cfg)


def test_profile_injected_data_access_still_allowed(tmp_path):
    """profile 文件里 documented 的 data_access 段在合并时被剥掉，不触发拒绝。"""
    from runtime.config import load_profile, _deep_merge, _forbid_unknown

    profile = load_profile("prod")  # prod.yaml has data_access section
    assert "data_access" in profile
    # load_config 内部剥掉后 merged payload 应通过严格 allowed set。
    cfg = tmp_path / "factor.yaml"
    cfg.write_text(
        "profile: prod\n"
        "factor:\n  name: f\n  expr: close\n"
        "data_source:\n  type: data_access\n  dataset: d\n",
        encoding="utf-8",
    )
    config = load_config(cfg)
    assert config.factor.name == "f"


# ---------------------------------------------------------------------------
# #135: nested source options typed coercion (composite/production)
# ---------------------------------------------------------------------------


def test_composite_source_options_typed_coercion():
    opts = TypedDataSourceOptions.from_dict(
        {
            "sources": {
                "pv": {"type": "data_access", "dataset": "d", "read_auto": "true"},
                "val": {"type": "data_access", "dataset": "v", "max_files": "5"},
            },
            "joins": {"val": "exact"},
            "snapshot_only": "true",
        }
    )
    d = opts.to_dict()
    assert d["sources"]["pv"]["read_auto"] is True
    assert d["sources"]["val"]["max_files"] == 5
    assert d["snapshot_only"] is True
    assert d["joins"] == {"val": "exact"}


def test_typed_source_options_rejects_bad_int():
    with pytest.raises(ValueError):
        TypedDataSourceOptions.from_dict({"max_files": "not-an-int"})


def test_typed_source_options_rejects_bad_bool():
    with pytest.raises(ValueError):
        TypedDataSourceOptions.from_dict({"read_auto": "maybe"})


def test_load_config_composite_source_typed(tmp_path):
    cfg = tmp_path / "factor.yaml"
    cfg.write_text(
        "factor:\n  name: f\n  expr: close\n"
        "data_source:\n"
        "  type: composite\n"
        "  anchor: pv\n"
        "  sources:\n"
        "    pv:\n      type: data_access\n      dataset: d\n      read_auto: \"true\"\n"
        "  joins:\n    pv: exact\n",
        encoding="utf-8",
    )
    config = load_config(cfg)
    assert config.data_source.options["sources"]["pv"]["read_auto"] is True
    assert config.data_source.options["joins"] == {"pv": "exact"}
