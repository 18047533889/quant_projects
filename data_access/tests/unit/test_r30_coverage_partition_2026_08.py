"""
R30-P0-007 / R30-P0-006 / R30-P1-022/023 单测。

覆盖：
    - PartitionMetadataIndex.objects_for_range / object_uris / index_fingerprint；
    - build_prune_paths_via_index 第二次命中缓存不再 glob（验收）；
    - CoverageService.describe 的 by_year/by_date/quantiles 结构与 authority 标记；
    - coverage_parity_fe 与 FE HistoricalCoverageContract 的差异对比；
    - MiningFieldProfile.from_semantic 默认类规则；
    - FieldCapabilityCatalog.search 过滤 + from_store 防御性遍历。

全部使用 fake store（stub _resolve_raw_paths / build_dataset_manifest / metadata_plane），
不触碰真实文件系统 / registry 全局。
"""
from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from data_access.r30.coverage_service import (
    CoverageService,
    coverage_parity_fe,
)
from data_access.r30.mining_profile import (
    FieldCapabilityCatalog,
    MiningFieldProfile,
)
from data_access.r30.partition_index import (
    PartitionMetadataIndex,
    PartitionMeta,
    assert_no_rescan_on_repeat,
    build_index,
    build_prune_paths_via_index,
    clear_index_cache,
)


# ---------------------------------------------------------------------------
# fake store / fake manifest
# ---------------------------------------------------------------------------
class _FakeManifestFile:
    def __init__(self, path, rows, min_time, max_time, bytes_=0,
                 min_instrument=None, max_instrument=None, schema_hash=None, etag=None):
        self.path = path
        self.rows = rows
        self.bytes = bytes_
        self.min_time = min_time
        self.max_time = max_time
        self.min_instrument = min_instrument
        self.max_instrument = max_instrument
        self.schema_hash = schema_hash
        self.etag = etag


class _FakeManifest:
    def __init__(self, dataset, files, *, epoch="e1", time_column="date",
                 instrument_column="symbol", time_dtype="string"):
        self.dataset = dataset
        self.files = tuple(files)
        self.time_column = time_column
        self.instrument_column = instrument_column
        self.time_dtype = time_dtype
        self.source_epoch = epoch
        self.manifest_built_epoch = epoch
        self.manifest_epoch = epoch

    @property
    def dataset_version(self) -> str:
        return f"dv-{len(self.files)}"

    @property
    def partition_version(self) -> str:
        return f"pv-{self.source_epoch}"

    @property
    def total_rows(self) -> int:
        return sum(f.rows or 0 for f in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)


class _FakePlane:
    """模拟 DatasetMetadataPlane：manifest() 走 _resolve_raw_paths（计数）后返回。"""

    def __init__(self, store, manifest):
        self._store = store
        self._manifest = manifest

    def manifest(self):
        if self._manifest is None:
            return None
        # 模拟真实 metadata_plane：先解析 raw paths 再返回 manifest
        ds = self._store._registry.get(self._store._dataset)
        self._store._resolve_raw_paths(ds, time_range=None, params={})
        return self._manifest


class _FakeEngine:
    """compute_coverage 用：glob(?) 返回该路径本身（模拟本地 1 文件/路径）。"""

    def execute_arrow(self, query, params, deadline_ms=None):
        # query = "SELECT file FROM glob(?) LIMIT ?"
        return pa.table({"file": [params[0]]})


class _FakeRegistry:
    def __init__(self, dataset, schema=None):
        self._dataset = dataset
        self._schema = schema or {"date": "string", "symbol": "string"}

    def get(self, name):
        return SimpleNamespace(
            name=name,
            format="parquet",
            time_column="date",
            instrument_column="symbol",
            schema=self._schema,
            semantic="panel",
        )


class _FakeStore:
    """可配置的 fake store。

    - ``manifest`` 传给 metadata_plane().manifest()（None 表示没有 manifest）；
    - ``_resolve_raw_paths`` 计数调用次数，返回 manifest file 路径或 fallback_paths；
    - ``_engine`` 只够 ``compute_coverage`` 的 glob 模拟。
    """

    def __init__(self, dataset, manifest=None, *, fallback_paths=None, schema=None):
        self._dataset = dataset
        self._manifest = manifest
        self._fallback_paths = fallback_paths or []
        self._resolve_calls = 0
        self._registry = _FakeRegistry(dataset, schema=schema)
        self._engine = _FakeEngine()

    # ---- store 公共面 ----
    def metadata_plane(self, dataset, **params):
        return _FakePlane(self, self._manifest)

    def build_dataset_manifest(self, dataset, *, include_row_groups=False, force=False, **params):
        # 返回 None 表示没有可构建的 manifest（fake：直接走 metadata_plane）
        return None

    def coverage(self, dataset, **params):
        from data_access.read.coverage import CoverageReport

        files = self._manifest.files if self._manifest is not None else []
        lo = min((f.min_time for f in files if f.min_time), default=None)
        hi = max((f.max_time for f in files if f.max_time), default=None)
        return CoverageReport(
            dataset=dataset,
            observed_start=lo,
            observed_end=hi,
            observed_files=len(files),
            status="complete" if files else "unavailable",
        )

    def manifest_version(self, dataset, **params):
        if self._manifest is None:
            return {"has_manifest": False}
        return {
            "has_manifest": True,
            "source_epoch": self._manifest.source_epoch,
            "manifest_built_epoch": self._manifest.manifest_built_epoch,
            "manifest_epoch": self._manifest.manifest_epoch,
            "file_count": self._manifest.file_count,
        }

    # ---- 被计数/被 scan 的原始路径解析 ----
    def _resolve_raw_paths(self, ds, *, time_range=None, params=None, instrument_filter=None):
        self._resolve_calls += 1
        if self._manifest is not None:
            return [f.path for f in self._manifest.files]
        return list(self._fallback_paths)


def _make_store(dataset="ds", *, files=None, epoch="e1", fallback_paths=None):
    manifest = None
    if files is not None:
        manifest = _FakeManifest(dataset, files, epoch=epoch)
    return _FakeStore(dataset, manifest=manifest, fallback_paths=fallback_paths)


def _hive_path(date_str):
    return f"/data/{date_str}/date={date_str}/part.parquet"


# ---------------------------------------------------------------------------
# R30-P0-006 Partition Metadata Index
# ---------------------------------------------------------------------------
def test_objects_for_range_exact_prune():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-01"), 10, "2024-01-01", "2024-01-01"),
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 30, "2024-01-03", "2024-01-03"),
        _FakeManifestFile(_hive_path("2025-01-01"), 40, "2025-01-01", "2025-01-01"),
    ])
    index = build_index(store, "ds")
    objs = index.objects_for_range(("2024-01-02", "2024-01-03"))
    uris = [o.object_uri for o in objs]
    assert len(uris) == 2
    assert any("2024-01-02" in u for u in uris)
    assert any("2024-01-03" in u for u in uris)
    assert not any("2024-01-01" in u for u in uris)
    assert not any("2025-01-01" in u for u in uris)


def test_objects_for_range_predicates_partition_values():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 30, "2024-01-03", "2024-01-03"),
    ])
    index = build_index(store, "ds")
    # 谓词过滤：date=2024-01-02
    uris = index.object_uris(("2024-01-01", "2024-12-31"), predicates={"date": "2024-01-02"})
    assert len(uris) == 1 and "2024-01-02" in uris[0]
    # 集合谓词
    uris2 = index.object_uris(
        None, predicates={"date": ["2024-01-02", "2024-01-03"]}
    )
    assert len(uris2) == 2


def test_partition_meta_fields_carried():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02",
                          bytes_=2048, schema_hash="sh1", etag="etag1"),
    ])
    index = build_index(store, "ds")
    obj = index.objects[0]
    assert obj.row_count == 20
    assert obj.byte_size == 2048
    assert obj.schema_epoch == "sh1"
    assert obj.etag == "etag1"
    assert obj.partition_values.get("date") == "2024-01-02"
    assert index.source_generation == "e1"
    assert index.built_epoch == "e1"


def test_index_fingerprint_stable_and_distinct():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 30, "2024-01-03", "2024-01-03"),
    ])
    i1 = build_index(store, "ds")
    i2 = build_index(store, "ds")
    assert i1.index_fingerprint() == i2.index_fingerprint()
    assert isinstance(i1.index_fingerprint(), str) and len(i1.index_fingerprint()) == 64
    # 不同对象集合 → 指纹不同
    i3 = PartitionMetadataIndex(
        dataset="ds",
        objects=(PartitionMeta(object_uri="x", min_time="2024-01-01", max_time="2024-01-01"),),
        built_epoch="e1",
    )
    assert i1.index_fingerprint() != i3.index_fingerprint()


def test_build_index_fallback_paths_no_manifest():
    store = _make_store(
        fallback_paths=[
            "/data/plain/2024-01-02.parquet",
            "/data/plain/2024-01-03.parquet",
        ]
    )
    index = build_index(store, "ds")
    # 无 manifest → 兜底路径近似：min/max 从路径日期解析
    objs = index.objects_for_range(("2024-01-02", "2024-01-02"))
    assert len(objs) == 1 and "2024-01-02" in objs[0].object_uri
    assert objs[0].min_time == "2024-01-02"
    assert objs[0].byte_size == 0


def test_build_prune_paths_via_index_second_hit_no_rescan(monkeypatch):
    clear_index_cache()
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 30, "2024-01-03", "2024-01-03"),
        _FakeManifestFile(_hive_path("2025-01-01"), 40, "2025-01-01", "2025-01-01"),
    ])
    result = assert_no_rescan_on_repeat(store, "ds", ("2024-01-01", "2024-12-31"))
    assert result["second_from_index"] is True
    assert result["paths_match"] is True
    # 第二次没有重新 glob：resolve 调用次数第一次后不再增加
    assert result["resolve_calls_after_second"] == result["resolve_calls_after_first"]
    assert result["resolve_calls_after_first"] >= 1  # 首次确实发生过至少一次解析
    # 首次 from_index 也应为 True（路径来自索引）
    assert result["first_from_index"] is True
    # 裁剪结果只含 2024 区间（首次构建的索引裁剪）
    paths1, _from = build_prune_paths_via_index(store, "ds", ("2024-01-01", "2024-12-31"))
    assert all("2024" in p for p in paths1)


def test_build_prune_paths_via_index_clear_cache_rebuild():
    clear_index_cache()
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 20, "2024-01-02", "2024-01-02"),
    ])
    paths1, from1 = build_prune_paths_via_index(store, "ds", None)
    clear_index_cache()
    paths2, from2 = build_prune_paths_via_index(store, "ds", None)
    assert from1 is True and from2 is True
    assert paths1 == paths2
    # 重建后对象仍完整
    index = build_index(store, "ds")
    assert len(index.objects) == 1


# ---------------------------------------------------------------------------
# R30-P0-007 CoverageService
# ---------------------------------------------------------------------------
def test_coverage_describe_structure_and_authority():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 100, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 150, "2024-01-03", "2024-01-03"),
        _FakeManifestFile(_hive_path("2025-06-01"), 200, "2025-06-01", "2025-06-01"),
    ])
    desc = CoverageService.describe(
        store, "ds", concept="test_c", market="ashare"
    )
    assert desc.dataset == "ds"
    assert desc.market == "ashare"
    assert desc.concept == "test_c"
    assert desc.first_valid_date == "2024-01-02"
    assert desc.last_valid_date == "2025-06-01"
    # by_date：当日行数 / 最大单日行数
    assert desc.by_date["2024-01-02"] == pytest.approx(0.5)
    assert desc.by_date["2024-01-03"] == pytest.approx(0.75)
    assert desc.by_date["2025-06-01"] == pytest.approx(1.0)
    # by_year：按年行数占比（250 / 450）
    assert desc.by_year[2024] == pytest.approx(250 / 450)
    assert desc.by_year[2025] == pytest.approx(200 / 450)
    # quantiles：p25/p50/p75/p90/max
    assert desc.quantiles["p25"] == pytest.approx(0.5)
    assert desc.quantiles["p50"] == pytest.approx(0.75)
    assert desc.quantiles["max"] == pytest.approx(1.0)
    assert desc.quantiles["p90"] == pytest.approx(1.0)
    # overall：无声明区间 → by_date 均值
    assert desc.overall_coverage == pytest.approx(0.75)
    assert desc.authority == "authoritative"
    assert isinstance(desc.by_instrument, dict)
    # by_instrument 无 instrument 粒度 → problems 注明
    assert any("instrument" in p for p in desc.problems)
    # to_dict 可序列化
    d = desc.to_dict()
    assert d["by_year"]["2024"] == pytest.approx(250 / 450)
    assert set(d["quantiles"]) == {"p25", "p50", "p75", "p90", "max"}


def test_coverage_describe_file_count_approximate():
    # 无每分区行数 → 文件数近似，authority=approximate
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), None, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), None, "2024-01-03", "2024-01-03"),
    ])
    desc = CoverageService.describe(store, "ds")
    assert desc.authority == "approximate"
    assert desc.by_date["2024-01-02"] == pytest.approx(1.0)
    assert any("近似" in p for p in desc.problems)


def test_coverage_describe_no_manifest_empty():
    store = _make_store(fallback_paths=[])
    desc = CoverageService.describe(store, "ds")
    assert desc.by_date == {}
    assert desc.by_year == {}
    assert desc.overall_coverage == 0.0


def test_coverage_parity_fe_diffs():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 100, "2024-01-02", "2024-01-02"),
        _FakeManifestFile(_hive_path("2024-01-03"), 150, "2024-01-03", "2024-01-03"),
        _FakeManifestFile(_hive_path("2025-06-01"), 200, "2025-06-01", "2025-06-01"),
    ])
    contract = {
        "first_valid_date": "2024-01-02",
        "coverage_by_year": {2024: 250 / 450, 2025: 200 / 450},
        "coverage_by_stock": {"AAPL": 1.0},  # DA 无 instrument 粒度 → 差异
        "coverage_by_date": {
            "2024-01-02": 0.5,
            "2024-01-03": 0.75,
            "2025-06-01": 1.0,
        },
        "coverage_ratio": 0.75,
    }
    result = coverage_parity_fe(store, "ds", contract)
    assert result["dataset"] == "ds"
    assert result["fields"]["first_valid_date"]["match"] is True
    assert result["fields"]["coverage_by_date"]["match"] is True
    assert result["fields"]["coverage_by_year"]["match"] is True
    assert result["fields"]["coverage_by_stock"]["match"] is False
    assert result["fields"]["coverage_ratio"]["match"] is True
    assert "coverage_by_stock" in result["differing_fields"]
    assert result["match_all"] is False


def test_coverage_parity_fe_accepts_dataclass():
    store = _make_store(files=[
        _FakeManifestFile(_hive_path("2024-01-02"), 100, "2024-01-02", "2024-01-02"),
    ])
    contract = SimpleNamespace(
        first_valid_date="2024-01-02",
        coverage_by_year={2024: 1.0},
        coverage_by_stock={},
        coverage_by_date={"2024-01-02": 1.0},
        coverage_ratio=1.0,
    )
    result = coverage_parity_fe(store, "ds", contract)
    assert result["fields"]["first_valid_date"]["match"] is True
    assert result["fields"]["coverage_by_date"]["match"] is True


# ---------------------------------------------------------------------------
# R30-P1-022 MiningFieldProfile
# ---------------------------------------------------------------------------
def test_mining_profile_financial_event_default():
    profile = MiningFieldProfile.from_semantic(
        None,
        "ashare_fin",
        {
            "logical_name": "roe",
            "physical_name": "roe",
            "dataset": "ashare_fin",
            "temporal_model": "financial_event",
            "frequency": "quarterly",
            "grain": "instrument",
            "pit_fidelity": "knowledge_date_pit",
        },
    )
    assert profile.mining_allowed is False  # event/financial 默认禁挖
    assert profile.pit_fidelity == "knowledge_date_pit"
    assert profile.coverage_class == "good"
    assert profile.cost_class == "cheap"  # quarterly
    assert "daily_panel" in profile.forbidden_operator_families
    assert "rolling_window_252" in profile.forbidden_operator_families
    d = profile.to_dict()
    assert d["mining_allowed"] is False
    assert d["concept_id"] == "roe"


def test_mining_profile_daily_panel_default():
    profile = MiningFieldProfile.from_semantic(
        None,
        "daily_prices",
        {
            "logical_name": "close",
            "dataset": "daily_prices",
            "frequency": "daily",
            "grain": "instrument",
            "temporal_model": "panel",
        },
    )
    assert profile.mining_allowed is True
    assert profile.pit_fidelity == "unsupported"
    assert profile.cost_class == "medium"
    assert "rolling" in profile.recommended_operator_families
    assert profile.forbidden_operator_families == ()


def test_mining_profile_sparse_event_forbids_daily_panel():
    profile = MiningFieldProfile.from_semantic(
        None,
        "events",
        {
            "logical_name": "announcement",
            "dataset": "events",
            "temporal_model": "sparse_event",
            "frequency": "daily",
            "grain": "event",
        },
    )
    assert profile.mining_allowed is False
    assert profile.pit_fidelity == "effective_only"
    assert profile.coverage_class == "limited"
    assert "rolling_window_252" in profile.forbidden_operator_families
    assert "daily_panel" in profile.forbidden_operator_families


def test_mining_profile_from_semantic_object():
    from data_access.read.semantic_catalog import SemanticField

    field = SemanticField(
        logical_name="turnover",
        dataset="daily_quotes",
        market="ashare",
        frequency="daily",
        grain="instrument",
        temporal_model="panel",
        pit_fidelity="knowledge_date_pit",
        mining_allowed=True,
    )
    profile = MiningFieldProfile.from_semantic(None, "daily_quotes", field)
    assert profile.mining_allowed is True
    assert profile.pit_fidelity == "knowledge_date_pit"
    assert profile.coverage_class == "good"


# ---------------------------------------------------------------------------
# R30-P1-023 FieldCapabilityCatalog
# ---------------------------------------------------------------------------
def test_field_capability_catalog_search():
    rows = [
        {
            "concept": "roe", "market": "ashare", "dataset": "ashare_fin",
            "frequency": "quarterly", "grain": "instrument", "unit": "ratio",
            "pit": "knowledge_date_pit", "coverage": "good", "cost_class": "cheap",
            "mining_allowed": True, "source_availability": "usable",
        },
        {
            "concept": "tick_volume", "market": "us", "dataset": "us_ticks",
            "frequency": "tick", "grain": "instrument", "unit": "shares",
            "pit": "effective_only", "coverage": "limited", "cost_class": "expensive",
            "mining_allowed": True, "source_availability": "usable",
        },
        {
            "concept": "eps_surprise", "market": "ashare", "dataset": "ashare_fin",
            "frequency": "quarterly", "grain": "snapshot", "unit": "ratio",
            "pit": "effective_only", "coverage": "limited", "cost_class": "cheap",
            "mining_allowed": False, "source_availability": "restricted",
        },
    ]
    catalog = FieldCapabilityCatalog(rows)
    assert len(catalog.rows()) == 3
    # market 过滤
    assert len(catalog.search(market="ashare")) == 2
    # pit_min=3 → 只留 knowledge_date_pit / vintage_pit
    assert [r["concept"] for r in catalog.search(pit_min=3)] == ["roe"]
    # cost_max=2（cheap/medium）→ 排除 expensive tick
    assert {r["concept"] for r in catalog.search(cost_max=2)} == {"roe", "eps_surprise"}
    # coverage_min=4（good/high）→ 只留 roe
    assert [r["concept"] for r in catalog.search(coverage_min=4)] == ["roe"]
    # concept 子串
    assert [r["concept"] for r in catalog.search(concept="eps")] == ["eps_surprise"]


def test_field_capability_catalog_from_store(monkeypatch):
    class _Field:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _FakeCatalog:
        def __init__(self, fields):
            self._fields = fields

        def names(self):
            return list(self._fields)

        def resolve_one(self, name, **kw):
            return self._fields[name]

    fields = {
        "roe": _Field(
            logical_name="roe", physical_name="roe", dataset="ashare_fin",
            market="ashare", frequency="quarterly", grain="instrument",
            temporal_model="financial_event", pit_fidelity="knowledge_date_pit",
            mining_allowed=False, source_unit="percent", canonical_unit="ratio",
            currency=None, dimension="ratio", dtype="float64",
        ),
        "close": _Field(
            logical_name="close", physical_name="close", dataset="daily_prices",
            market="ashare", frequency="daily", grain="instrument",
            temporal_model="panel", pit_fidelity=None,
            mining_allowed=True, source_unit="yuan", canonical_unit="yuan",
            currency="CNY", dimension="price", dtype="float64",
        ),
    }
    monkeypatch.setattr(
        "data_access.r30.mining_profile._get_semantic_catalog",
        lambda: _FakeCatalog(fields),
    )
    store = _FakeStore("daily_prices", schema={"date": "string", "symbol": "string", "close": "double"})
    catalog = FieldCapabilityCatalog.from_store(store)
    rows = catalog.rows()
    assert rows, "from_store 应至少产出 1 行"
    by_concept = {r["concept"]: r for r in rows}
    assert by_concept["roe"]["market"] == "ashare"
    assert by_concept["roe"]["mining_allowed"] is False
    assert by_concept["roe"]["source_availability"] == "restricted"
    assert by_concept["close"]["cost_class"] == "medium"
    assert by_concept["close"]["unit"] == "yuan"
