# -*- coding: utf-8
"""R39 PERF-069 / PERF-070 —— 批量增量物化按 ExecutionIdentity 分组 + FactorCampaignSession。

PERF-069
    把 ``materialize_incremental_many_from_config`` 从逐配置串行改为：
    先加载全部 config → 按严格 ``ExecutionIdentity`` 分组 → 同组共享一个
    engine/source session，逐因子仍 ``materialize_incremental``。
    * 同 identity 的两份配置：engine 创建数 < 配置数；
    * 不同 market/frequency 的配置：各自独立 engine；
    * 输出 dict 与参考串行实现完全一致。

PERF-070
    ``FactorCampaignSession(identity, engine, source_session, compiled_factors)``
    一个 identity 一个 session；``compile()`` 用 engine 的 CSE 批编译
    ``compile_many``，2 个因子一个 session 的编译次数 < 2 个 session 各自编译。

测试全部使用 synthetic config / fake engine —— 不跑真实引擎。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.execution_identity import ExecutionIdentity, execution_identity_from_config
from factor_engine.runtime.factor_campaign_session import FactorCampaignSession


def _write_factor_yaml(
    tmp_path: Path,
    name: str,
    expr: str,
    root: Path,
    *,
    market: str | None = None,
    freq: str | None = None,
    write_target: str | None = None,
) -> Path:
    lines = [
        f"factor:",
        f"  name: {name}",
        f"  expr: {expr}",
        (f"  freq: {freq}" if freq else ""),
        f"data_source:",
        f"  type: parquet_kline",
        f"  root: {root}",
        f"  timestamp_column: datetime",
        f"  instrument_column: asset",
        f"  fields:",
        f"    close: close",
    ]
    if market is not None:
        lines.append(f"run:")
        lines.append(f"  market: {market}")
    if write_target is not None:
        lines.append(f"materialization:")
        lines.append(f"  target: {write_target}")
    path = tmp_path / f"{name}.yaml"
    path.write_text("\n".join(l for l in lines if l) + "\n", encoding="utf-8")
    return path


class FakeEngine:
    """Synthetic engine: counts compile_many, records materialize calls."""

    def __init__(self) -> None:
        self.data_source = object()
        self.materialize_calls: list[str] = []
        self.compile_calls: list[list[str]] = []

    def materialize_incremental(self, factor, **kwargs):
        self.materialize_calls.append(factor.name)
        return {
            "materialization": {"factor_id": factor.name, "rows_written": 1},
            "incremental": {"factor_id": factor.name, "since": kwargs.get("since")},
        }

    def compile_many(self, factors, *, enable_cse=True):
        self.compile_calls.append([f.name for f in factors])
        return type("FakeDAGPlan", (), {"roots": [], "shared_nodes": {}})()


def _patch_from_loaded_config(monkeypatch, counter: dict[str, int]) -> None:
    """Replace ``FactorEngine.from_loaded_config`` with a fake that counts."""

    def _fake_from_loaded_config(cls, config, *args, **kwargs):
        counter["engine_created"] += 1
        return FakeEngine(), None

    monkeypatch.setattr(
        FactorEngine, "from_loaded_config", classmethod(_fake_from_loaded_config)
    )


def _reference_serial(engine_cls, config_paths):
    """旧行为的逐配置串行参考实现（不跑真实引擎）。"""
    from factor_engine.runtime.config import load_config
    from factor_engine.runtime.config_runtime import resolve_materialize_kwargs_for_pipeline
    from factor_engine.runtime.incremental_event_service import _factor_from_config

    outputs: dict[str, object] = {}
    for path in config_paths:
        config = load_config(path, profile=None)
        factor = _factor_from_config(config)
        opts = resolve_materialize_kwargs_for_pipeline(config, None)
        eng = FakeEngine()
        out = eng.materialize_incremental(
            factor, **opts.to_incremental_materialize_kwargs()
        )
        out["config"] = config
        out["config_path"] = str(path)
        outputs[factor.name] = out
    return {"materializations": outputs}


def _normalized(outputs: dict[str, object]) -> dict[str, object]:
    """去掉不可稳定比较的 ``config`` 对象，仅比较物化语义字段。"""
    return {
        name: {k: v for k, v in out.items() if k != "config"}
        for name, out in outputs.items()
    }


# ---------------------------------------------------------------------------
# PERF-069
# ---------------------------------------------------------------------------


def test_materialize_incremental_many_groups_identical_identity(tmp_path, monkeypatch):
    """(a) 相同 ExecutionIdentity 的两份配置共享一个 engine（创建数 < 配置数）。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg_a = _write_factor_yaml(tmp_path, "ga", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "gb", "close * 2", root)

    counter = {"engine_created": 0}
    _patch_from_loaded_config(monkeypatch, counter)

    out = FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])

    # 同一 identity → 一个 engine（少于 2 份配置）。
    assert counter["engine_created"] == 1
    assert counter["engine_created"] < 2
    assert set(out["materializations"]) == {"ga", "gb"}


def test_materialize_incremental_many_separates_different_market(tmp_path, monkeypatch):
    """(b) 不同 market 的配置各自独立 engine。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg_a = _write_factor_yaml(tmp_path, "ma", "close", root, market="ashare")
    cfg_b = _write_factor_yaml(tmp_path, "mb", "close", root, market="us")

    counter = {"engine_created": 0}
    _patch_from_loaded_config(monkeypatch, counter)

    out = FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])

    # 不同 market → 两个 identity → 两个 engine（各自独立，绝不共享）。
    assert counter["engine_created"] == 2
    assert set(out["materializations"]) == {"ma", "mb"}


def test_materialize_incremental_many_separates_different_frequency(
    tmp_path, monkeypatch
):
    """(b) 不同 frequency 的配置各自独立 engine。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg_a = _write_factor_yaml(tmp_path, "fa", "close", root, freq="1d")
    cfg_b = _write_factor_yaml(tmp_path, "fb", "close", root, freq="2h")

    counter = {"engine_created": 0}
    _patch_from_loaded_config(monkeypatch, counter)

    FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])
    assert counter["engine_created"] == 2


def test_materialize_incremental_many_outputs_match_serial(tmp_path, monkeypatch):
    """(c) 新分组实现的输出 dict 与参考逐配置串行完全一致。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg_a = _write_factor_yaml(tmp_path, "oa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "ob", "close * 2", root)

    counter = {"engine_created": 0}
    _patch_from_loaded_config(monkeypatch, counter)

    new = FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])
    ref = _reference_serial(FactorEngine, [cfg_a, cfg_b])

    assert _normalized(new["materializations"]) == _normalized(
        ref["materializations"]
    )
    assert set(new["materializations"]) == set(ref["materializations"])


def test_materialize_incremental_many_shared_engine_reuses_materialize(
    tmp_path, monkeypatch
):
    """同组两个因子在共享 engine 上逐因子 materialize（调用两次、引擎一次）。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg_a = _write_factor_yaml(tmp_path, "sa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "sb", "close * 2", root)

    engines: list[FakeEngine] = []

    def _fake_from_loaded_config(cls, config, *args, **kwargs):
        eng = FakeEngine()
        engines.append(eng)
        return eng, None

    monkeypatch.setattr(
        FactorEngine, "from_loaded_config", classmethod(_fake_from_loaded_config)
    )

    out = FactorEngine.materialize_incremental_many_from_config([cfg_a, cfg_b])

    assert len(engines) == 1
    engine = engines[0]
    assert sorted(engine.materialize_calls) == ["sa", "sb"]
    # 同一 engine 上两个因子都有输出。
    assert set(out["materializations"]) == {"sa", "sb"}


# ---------------------------------------------------------------------------
# PERF-070
# ---------------------------------------------------------------------------


def _identity_for(tmp_path: Path) -> ExecutionIdentity:
    from factor_engine.runtime.config import load_config

    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    cfg = _write_factor_yaml(tmp_path, "id", "close", root)
    config = load_config(cfg)
    return execution_identity_from_config(config)


def _factor(name: str):
    from factor_engine.api.dsl_parser import parse_factor

    return parse_factor("close", name=name)


def test_factor_campaign_session_compile_reuse(tmp_path):
    """(d) 2 因子一个 session 编译 1 次 < 2 个 session 各编译 1 次。"""
    identity = _identity_for(tmp_path)
    f1 = _factor("cf1")
    f2 = _factor("cf2")

    # 一个 session、两个因子 → 一次 CSE 批编译。
    engine = FakeEngine()
    session = FactorCampaignSession(identity=identity, engine=engine)
    session.add_factor(None, f1, "cf1.yaml", None)
    session.add_factor(None, f2, "cf2.yaml", None)
    session.compile()
    session.compile()  # 命中缓存，不再编译
    assert engine.compile_calls == [["cf1", "cf2"]]
    assert session.compile_count == 1

    # 两个 session、各一个因子 → 两次独立编译。
    engine2 = FakeEngine()
    s_a = FactorCampaignSession(identity=identity, engine=engine2)
    s_a.add_factor(None, f1, "cf1.yaml", None)
    s_a.compile()
    s_b = FactorCampaignSession(identity=identity, engine=engine2)
    s_b.add_factor(None, f2, "cf2.yaml", None)
    s_b.compile()
    assert engine2.compile_calls == [["cf1"], ["cf2"]]

    # 复用收益：1 次 < 2 次。
    assert len(engine.compile_calls) < len(engine2.compile_calls)


def test_execution_identity_hashable_and_strict():
    """ExecutionIdentity 可哈希；任一字段不同即不同 identity。"""
    from factor_engine.runtime.execution_identity import freeze_options

    identity = ExecutionIdentity(
        source_snapshot=(("type", "parquet_kline"),),
        dataset_config=(("type", "parquet_kline"), ("options", freeze_options(None))),
        market="ashare",
        universe=None,
        frequency="1d",
        calendar=None,
        pit_policy=(False, False),
        writer_target="local",
    )
    assert hash(identity) == hash(identity)
    diff = ExecutionIdentity(
        source_snapshot=(("type", "parquet_kline"),),
        dataset_config=(("type", "parquet_kline"), ("options", freeze_options(None))),
        market="us",
        universe=None,
        frequency="1d",
        calendar=None,
        pit_policy=(False, False),
        writer_target="local",
    )
    assert identity != diff
    assert hash(identity) != hash(diff)
