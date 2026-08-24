from __future__ import annotations

from dataclasses import dataclass

import pytest

from factor_engine.api.dsl_parser import parse_factor
from factor_engine.mining.campaign import (
    MiningCampaignSession,
    MiningCampaignSnapshot,
    NegativeCompileCache,
    candidate_semantic_hash,
    dependency_signature,
    group_source_first,
)


@dataclass(frozen=True)
class _Field:
    field_id: str
    source_name: str
    table: str = ""


@dataclass
class _Analysis:
    lookback: int
    referenced_columns: set[str]
    referenced_fields: dict[str, _Field]
    referenced_field_ids: set[str]


class _Engine:
    def __init__(self) -> None:
        self.analyze_calls: list[list[str]] = []
        self.stream_calls = 0

    def analyze_batch(self, factors, *, enable_cse=True):
        names = [factor.name for factor in factors]
        self.analyze_calls.append(names)
        if any(name.startswith("bad") for name in names):
            raise ValueError("invalid deterministic candidate")
        return {
            "dag": "dag:" + ",".join(names),
            "analyses": {
                factor.name: _Analysis(4, {"close"}, {}, set())
                for factor in factors
            },
        }

    def run_many_iter(self, factors, **kwargs):
        self.stream_calls += 1
        for factor in factors:
            yield factor.name, [factor.name], {"backend": "fake"}


def _snapshot(generation: str = "compiler-v1") -> MiningCampaignSnapshot:
    return MiningCampaignSnapshot("source-v1", "universe-v1", generation)


def test_candidate_hash_is_name_independent_but_argument_order_sensitive():
    left = parse_factor("add(close, volume)", name="left")
    renamed = parse_factor("add(close, volume)", name="renamed")
    reversed_args = parse_factor("add(volume, close)", name="reversed")

    assert candidate_semantic_hash(left) == candidate_semantic_hash(renamed)
    assert candidate_semantic_hash(left) != candidate_semantic_hash(reversed_args)


def test_negative_compile_cache_skips_repeat_only_within_generation():
    engine = _Engine()
    cache = NegativeCompileCache()
    bad = parse_factor("close", name="bad_one")

    first = MiningCampaignSession(engine, _snapshot("v1"), negative_cache=cache)
    assert "bad_one" in first.compile_many([bad])["failures"]
    assert engine.analyze_calls == [["bad_one"]]
    assert "bad_one" in first.compile_many([bad])["failures"]
    assert engine.analyze_calls == [["bad_one"]]

    second = MiningCampaignSession(engine, _snapshot("v2"), negative_cache=cache)
    assert "bad_one" in second.compile_many([bad])["failures"]
    assert engine.analyze_calls == [["bad_one"], ["bad_one"]]
    assert len(cache) == 2


def test_compile_many_batches_only_survivors_and_exposes_immutable_results():
    engine = _Engine()
    good = parse_factor("close", name="good")
    bad = parse_factor("volume", name="bad_invalid")
    session = MiningCampaignSession(engine, _snapshot())

    result = session.compile_many([good, bad])

    assert result["plan"] == "dag:good"
    assert [factor.name for factor in result["factors"]] == ["good"]
    assert engine.analyze_calls == [["good", "bad_invalid"], ["good"], ["bad_invalid"]]
    assert set(result["analyses"]) == {"good"}
    assert set(result["failures"]) == {"bad_invalid"}
    with pytest.raises(TypeError):
        result["failures"]["x"] = object()


def test_dependency_signature_and_source_first_grouping_are_deterministic():
    fa = parse_factor("ts_mean(close, 5)", name="z_factor")
    fb = parse_factor("ts_std(close, 5)", name="a_factor")
    fc = parse_factor("volume", name="other_source")
    common = _Analysis(
        4,
        {"close"},
        {"close": _Field("bars.close", "daily-bars")},
        {"bars.close"},
    )
    other = _Analysis(
        0,
        {"volume"},
        {"volume": _Field("trades.volume", "daily-trades")},
        {"trades.volume"},
    )
    sa = dependency_signature(fa, common, backend_capability={"polars", "pandas"})
    sb = dependency_signature(fb, common, backend_capability={"pandas", "polars"})
    sc = dependency_signature(fc, other)

    groups = group_source_first([(fa, sa), (fc, sc), (fb, sb)])

    assert sa.source_first_key == sb.source_first_key
    assert sa.backend_capability == ("pandas", "polars")
    grouped_names = [[factor.name for factor in factors] for _, factors in groups]
    assert ["a_factor", "z_factor"] in grouped_names
    assert ["other_source"] in grouped_names


def test_snapshot_is_frozen_and_stream_evaluator_receives_pinned_identity():
    engine = _Engine()
    seen = []

    def evaluator(name, block, snapshot):
        seen.append((name, block, dict(snapshot)))
        return {"accepted": True, "rows": len(block)}

    session = MiningCampaignSession(engine, _snapshot(), evaluator=evaluator)
    factor = parse_factor("close", name="streamed")

    rows = list(session.evaluate_many_iter([factor]))

    assert rows == [("streamed", {"accepted": True, "rows": 1}, {"backend": "fake"})]
    assert seen[0][2] == {
        "source_snapshot": "source-v1",
        "universe_snapshot": "universe-v1",
        "compiler_generation": "compiler-v1",
    }
    assert engine.stream_calls == 1
    with pytest.raises(Exception):
        session.snapshot.source_snapshot = "changed"


@pytest.mark.parametrize("field", ["source_snapshot", "universe_snapshot", "compiler_generation"])
def test_snapshot_rejects_unpinned_identity(field):
    values = {
        "source_snapshot": "source-v1",
        "universe_snapshot": "universe-v1",
        "compiler_generation": "compiler-v1",
    }
    values[field] = ""
    with pytest.raises(ValueError, match=field):
        MiningCampaignSnapshot(**values)
