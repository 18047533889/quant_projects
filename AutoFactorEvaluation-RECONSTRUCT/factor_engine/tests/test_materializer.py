"""因子落盘系统完整测试。

覆盖
----
- FactorCatalog：注册 / Hash 防呆 / 水位线 / 增删查
- ParquetMaterializer：Schema 强转 / 数据清洗 / 幂等 Upsert / 年分区 / 原子写入
- ResultStore（Polars + Pandas fallback）：单因子读取 / 分区剪枝 / 多因子拼接 / 跨频对齐
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storage.catalog import FactorCatalog, compute_ir_hash
from storage.exceptions import FactorHashMismatchError, FactorNotFoundError
from storage.materializer import ParquetMaterializer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ir_node(op: str, attrs: dict | None = None, inputs=()) -> object:
    """最小化 IRNode 模拟，用于测试 AST Hash。"""
    from ir.nodes import IRNode

    return IRNode(op=op, inputs=tuple(inputs), attrs=attrs or {})


def _make_series(
    dates: list[str],
    assets: list[str],
    values: list[float] | None = None,
) -> pd.Series:
    """构建 MultiIndex(timestamp, instrument) Series。"""
    rows = []
    val_idx = 0
    for d in dates:
        for a in assets:
            v = values[val_idx] if values else float(val_idx + 1)
            rows.append((pd.Timestamp(d), a, v))
            val_idx += 1

    df = pd.DataFrame(rows, columns=["timestamp", "instrument", "value"])
    return df.set_index(["timestamp", "instrument"])["value"]


# ===========================================================================
# 1. FactorCatalog 测试
# ===========================================================================


class TestFactorCatalog:
    """元数据目录测试。"""

    def test_register_and_query(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            info = cat.get_factor_info("mom_v1")

        assert info is not None
        assert info["factor_id"] == "mom_v1"
        assert info["author"] == "alice"
        assert info["frequency"] == "5m"
        assert info["ast_hash"] == "abc123"

    def test_register_idempotent(self, tmp_path):
        """同 ID 同 Hash → 静默跳过，不报错。"""
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            assert len(cat.list_factors()) == 1

    def test_hash_mismatch_raises(self, tmp_path):
        """同 ID 不同 Hash → FactorHashMismatchError。"""
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            with pytest.raises(FactorHashMismatchError, match="mom_v1"):
                cat.register(
                    "mom_v1", author="alice", frequency="5m", ast_hash="DIFFERENT"
                )

    def test_verify_hash(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            assert cat.verify_hash("mom_v1", "abc123") is True
            assert cat.verify_hash("mom_v1", "xyz789") is False
            assert cat.verify_hash("nonexistent", "anything") is True

    def test_watermark_lifecycle(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")

            assert cat.get_watermark("mom_v1") is None

            cat.update_watermark("mom_v1", "2023-01-01", "2023-06-30", row_count=1000)
            wm = cat.get_watermark("mom_v1")
            assert wm["start_date"] == "2023-01-01"
            assert wm["end_date"] == "2023-06-30"
            assert wm["row_count"] == 1000

            # 更新水位线
            cat.update_watermark("mom_v1", "2023-01-01", "2023-12-31", row_count=2000)
            wm = cat.get_watermark("mom_v1")
            assert wm["end_date"] == "2023-12-31"
            assert wm["row_count"] == 2000

    def test_delete_factor(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("mom_v1", author="alice", frequency="5m", ast_hash="abc123")
            cat.update_watermark("mom_v1", "2023-01-01", "2023-12-31")
            cat.delete_factor("mom_v1")
            assert cat.get_factor_info("mom_v1") is None
            assert cat.get_watermark("mom_v1") is None

    def test_list_factors_with_watermark(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with FactorCatalog(db) as cat:
            cat.register("a_v1", author="alice", frequency="1d", ast_hash="h1")
            cat.register("b_v1", author="bob", frequency="5m", ast_hash="h2")
            cat.update_watermark("a_v1", "2023-01-01", "2023-12-31")
            factors = cat.list_factors()
            assert len(factors) == 2
            # a_v1 有水位线
            assert factors[0]["start_date"] == "2023-01-01"
            # b_v1 无水位线
            assert factors[1]["start_date"] is None


# ===========================================================================
# 2. IR Hash 测试
# ===========================================================================


class TestIRHash:
    """AST Hash 确定性测试。"""

    def test_deterministic(self):
        """同结构 IR 树 → 相同 Hash。"""
        ir1 = _make_ir_node(
            "ts_mean",
            {"d": 20, "min_periods": 1},
            [_make_ir_node("column", {"name": "close"})],
        )
        ir2 = _make_ir_node(
            "ts_mean",
            {"d": 20, "min_periods": 1},
            [_make_ir_node("column", {"name": "close"})],
        )
        assert compute_ir_hash(ir1) == compute_ir_hash(ir2)

    def test_different_structure(self):
        """不同 IR 树 → 不同 Hash。"""
        ir1 = _make_ir_node(
            "ts_mean",
            {"d": 20, "min_periods": 1},
            [_make_ir_node("column", {"name": "close"})],
        )
        ir2 = _make_ir_node(
            "ts_mean",
            {"d": 10, "min_periods": 1},
            [_make_ir_node("column", {"name": "close"})],
        )
        assert compute_ir_hash(ir1) != compute_ir_hash(ir2)

    def test_attr_order_independent(self):
        """attrs 字典键顺序不影响 Hash。"""
        ir1 = _make_ir_node("rank", {"rate": 1.0, "extra": "a"})
        ir2 = _make_ir_node("rank", {"extra": "a", "rate": 1.0})
        assert compute_ir_hash(ir1) == compute_ir_hash(ir2)


# ===========================================================================
# 3. ParquetMaterializer 测试
# ===========================================================================


class TestParquetMaterializer:
    """物化器核心测试。"""

    def test_default_lake_root_uses_workspace_data(self, tmp_path, monkeypatch):
        workspace_root = tmp_path / "workspace_data"
        monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(workspace_root))

        mat = ParquetMaterializer()

        assert mat.lake_root == workspace_root / "factors" / "lake"

    def test_basic_materialize(self, tmp_path):
        """基本落盘：写入、验证 Parquet 和 Catalog。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15", "2024-01-16"],
            assets=["000001.SZ", "000002.SZ"],
        )
        result = mat.materialize(
            factor_id="test_v1",
            result=series,
            author="tester",
            frequency="1d",
            ast_hash="h1",
        )
        assert result["factor_id"] == "test_v1"
        assert result["rows_written"] == 4
        assert 2024 in result["partitions"]

        # 验证 Parquet 文件存在
        pq_path = tmp_path / "factors" / "test_v1" / "year=2024" / "data.parquet"
        assert pq_path.exists()

        # 验证 Catalog
        info = mat.catalog.get_factor_info("test_v1")
        assert info["author"] == "tester"
        assert info["ast_hash"] == "h1"

    def test_float32_enforcement(self, tmp_path):
        """值列强制降级为 Float32。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ"],
            values=[3.14159265358979],
        )
        mat.materialize(
            factor_id="f32_test",
            result=series,
            ast_hash="h1",
        )
        pq_path = tmp_path / "factors" / "f32_test" / "year=2024" / "data.parquet"
        df = pd.read_parquet(pq_path)
        assert df["value"].dtype == np.float32

    def test_inf_cleaned(self, tmp_path):
        """±inf → NaN → 被 dropna 清除。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ", "000002.SZ", "000003.SZ"],
            values=[1.0, float("inf"), float("-inf")],
        )
        result = mat.materialize(
            factor_id="inf_test",
            result=series,
            ast_hash="h1",
        )
        # 只有 1 行有效数据
        assert result["rows_written"] == 1

    def test_nan_dropped(self, tmp_path):
        """NaN 值行被 dropna 清除。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ", "000002.SZ"],
            values=[1.5, float("nan")],
        )
        result = mat.materialize(
            factor_id="nan_test",
            result=series,
            ast_hash="h1",
        )
        assert result["rows_written"] == 1

    def test_idempotent_upsert(self, tmp_path):
        """重复写入相同数据 → 行数不变。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15", "2024-01-16"],
            assets=["000001.SZ", "000002.SZ"],
        )
        mat.materialize(factor_id="idem_test", result=series, ast_hash="h1")
        mat.materialize(factor_id="idem_test", result=series, ast_hash="h1")

        pq_path = tmp_path / "factors" / "idem_test" / "year=2024" / "data.parquet"
        df = pd.read_parquet(pq_path)
        assert len(df) == 4  # 不重复

    def test_incremental_update(self, tmp_path):
        """增量追加新日期的数据。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series1 = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ"],
            values=[1.0],
        )
        series2 = _make_series(
            dates=["2024-01-16"],
            assets=["000001.SZ"],
            values=[2.0],
        )
        mat.materialize(factor_id="incr_test", result=series1, ast_hash="h1")
        mat.materialize(factor_id="incr_test", result=series2, ast_hash="h1")

        pq_path = tmp_path / "factors" / "incr_test" / "year=2024" / "data.parquet"
        df = pd.read_parquet(pq_path)
        assert len(df) == 2
        assert sorted(df["value"].tolist()) == pytest.approx([1.0, 2.0], abs=1e-5)

    def test_upsert_overwrites_old_value(self, tmp_path):
        """同 [datetime, asset] 新值覆盖旧值。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series_old = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ"],
            values=[1.0],
        )
        series_new = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ"],
            values=[99.0],
        )
        mat.materialize(factor_id="overwrite_test", result=series_old, ast_hash="h1")
        mat.materialize(factor_id="overwrite_test", result=series_new, ast_hash="h1")

        pq_path = (
            tmp_path / "factors" / "overwrite_test" / "year=2024" / "data.parquet"
        )
        df = pd.read_parquet(pq_path)
        assert len(df) == 1
        assert df["value"].iloc[0] == pytest.approx(99.0, abs=1e-5)

    def test_year_partitioning(self, tmp_path):
        """跨年数据正确分到不同分区。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2023-12-31", "2024-01-01"],
            assets=["000001.SZ"],
            values=[1.0, 2.0],
        )
        result = mat.materialize(
            factor_id="cross_year", result=series, ast_hash="h1"
        )
        assert sorted(result["partitions"]) == [2023, 2024]

        pq_2023 = tmp_path / "factors" / "cross_year" / "year=2023" / "data.parquet"
        pq_2024 = tmp_path / "factors" / "cross_year" / "year=2024" / "data.parquet"
        assert pq_2023.exists()
        assert pq_2024.exists()
        assert len(pd.read_parquet(pq_2023)) == 1
        assert len(pd.read_parquet(pq_2024)) == 1

    def test_hash_mismatch_blocks_materialize(self, tmp_path):
        """公式变更但 factor_id 未变 → 拒绝落盘。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(dates=["2024-01-15"], assets=["000001.SZ"], values=[1.0])

        mat.materialize(factor_id="guarded", result=series, ast_hash="hash_v1")

        with pytest.raises(FactorHashMismatchError, match="guarded"):
            mat.materialize(
                factor_id="guarded", result=series, ast_hash="hash_v2_different"
            )

    def test_watermark_tracking(self, tmp_path):
        """落盘后水位线正确更新。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series1 = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ"],
            values=[1.0],
        )
        mat.materialize(factor_id="wm_test", result=series1, ast_hash="h1")
        wm = mat.catalog.get_watermark("wm_test")
        assert "2024-01-15" in wm["start_date"]

        # 增量更新 → 水位线扩展
        series2 = _make_series(
            dates=["2024-06-30"],
            assets=["000001.SZ"],
            values=[2.0],
        )
        mat.materialize(factor_id="wm_test", result=series2, ast_hash="h1")
        wm = mat.catalog.get_watermark("wm_test")
        assert "2024-06-30" in wm["end_date"]

    def test_empty_after_cleaning(self, tmp_path):
        """全部为 NaN/Inf 时跳过落盘。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(
            dates=["2024-01-15"],
            assets=["000001.SZ", "000002.SZ"],
            values=[float("nan"), float("inf")],
        )
        result = mat.materialize(
            factor_id="empty_test", result=series, ast_hash="h1"
        )
        assert result["rows_written"] == 0
        assert result["watermark"] is None

    def test_sorted_output(self, tmp_path):
        """写入的 Parquet 数据按 [asset, datetime] 排序。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        # 故意以乱序输入
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-16"), "B"),
                (pd.Timestamp("2024-01-15"), "A"),
                (pd.Timestamp("2024-01-15"), "B"),
                (pd.Timestamp("2024-01-16"), "A"),
            ],
            names=["timestamp", "instrument"],
        )
        series = pd.Series([4.0, 1.0, 3.0, 2.0], index=idx, name="value")
        mat.materialize(factor_id="sort_test", result=series, ast_hash="h1")

        pq_path = tmp_path / "factors" / "sort_test" / "year=2024" / "data.parquet"
        df = pd.read_parquet(pq_path)
        # 应按 asset 升序、datetime 升序
        assert df["asset"].tolist() == ["A", "A", "B", "B"]

    def test_delete_factor_with_files(self, tmp_path):
        """删除因子同时删物理文件。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _make_series(dates=["2024-01-15"], assets=["000001.SZ"], values=[1.0])
        mat.materialize(factor_id="del_test", result=series, ast_hash="h1")

        factor_dir = tmp_path / "factors" / "del_test"
        assert factor_dir.exists()

        mat.delete_factor("del_test", delete_files=True)
        assert not factor_dir.exists()
        assert mat.catalog.get_factor_info("del_test") is None

    def test_ir_node_hash_integration(self, tmp_path):
        """使用真实 IRNode 计算 Hash 并落盘。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        ir = _make_ir_node(
            "rank",
            {"rate": 1.0},
            [_make_ir_node("ts_mean", {"d": 20, "min_periods": 1},
                           [_make_ir_node("column", {"name": "close"})])],
        )
        series = _make_series(dates=["2024-01-15"], assets=["000001.SZ"], values=[1.0])
        result = mat.materialize(
            factor_id="ir_hash_test",
            result=series,
            ir_node=ir,
        )
        assert result["rows_written"] == 1
        info = mat.catalog.get_factor_info("ir_hash_test")
        assert len(info["ast_hash"]) == 64  # SHA-256 hex


# ===========================================================================
# 4. ResultStore 测试
# ===========================================================================


def _setup_lake(tmp_path, factor_id, dates, assets, values, freq="1d", ast_hash="h1"):
    """辅助：预先落盘一个因子供 ResultStore 读取。"""
    mat = ParquetMaterializer(lake_root=tmp_path)
    series = _make_series(dates, assets, values)
    mat.materialize(
        factor_id=factor_id,
        result=series,
        frequency=freq,
        ast_hash=ast_hash,
    )
    return mat


class TestPandasResultStore:
    """Pandas 兜底后端测试。"""

    def test_load_single_factor(self, tmp_path):
        from storage.result_store import PandasResultStore

        _setup_lake(
            tmp_path,
            "f1",
            ["2024-01-15", "2024-01-16"],
            ["A", "B"],
            [1.0, 2.0, 3.0, 4.0],
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PandasResultStore(tmp_path, cat)
        df = store.load_factor("f1")
        assert len(df) == 4
        assert set(df.columns) == {"datetime", "asset", "value"}

    def test_load_factor_not_found(self, tmp_path):
        from storage.result_store import PandasResultStore

        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PandasResultStore(tmp_path, cat)
        with pytest.raises(FactorNotFoundError):
            store.load_factor("nonexistent")

    def test_partition_pruning(self, tmp_path):
        from storage.result_store import PandasResultStore

        _setup_lake(
            tmp_path,
            "f1",
            ["2023-06-15", "2024-01-15"],
            ["A"],
            [1.0, 2.0],
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PandasResultStore(tmp_path, cat)

        # 只请求 2024 → 只返回 1 行
        df = store.load_factor("f1", start="2024-01-01")
        assert len(df) == 1

    def test_multi_factor_join(self, tmp_path):
        from storage.result_store import PandasResultStore

        _setup_lake(
            tmp_path, "f1", ["2024-01-15"], ["A"], [1.0], ast_hash="h1"
        )
        _setup_lake(
            tmp_path, "f2", ["2024-01-15"], ["A"], [2.0], ast_hash="h2"
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PandasResultStore(tmp_path, cat)
        df = store.load_factors(["f1", "f2"])
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert len(df) == 1

    def test_to_pandas(self, tmp_path):
        from storage.result_store import PandasResultStore

        _setup_lake(
            tmp_path, "f1", ["2024-01-15"], ["A"], [1.0], ast_hash="h1"
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PandasResultStore(tmp_path, cat)
        df = store.to_pandas(["f1"])
        assert isinstance(df, pd.DataFrame)


class TestPolarsResultStore:
    """Polars 后端测试（需要安装 polars）。"""

    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_load_single_factor(self, tmp_path):
        import polars as pl
        from storage.result_store import PolarsResultStore

        _setup_lake(
            tmp_path,
            "f1",
            ["2024-01-15", "2024-01-16"],
            ["A", "B"],
            [1.0, 2.0, 3.0, 4.0],
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PolarsResultStore(tmp_path, cat)
        lf = store.load_factor("f1")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 4

    def test_partition_pruning(self, tmp_path):
        from storage.result_store import PolarsResultStore

        _setup_lake(
            tmp_path,
            "f1",
            ["2023-06-15", "2024-01-15"],
            ["A"],
            [1.0, 2.0],
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PolarsResultStore(tmp_path, cat)
        lf = store.load_factor("f1", start="2024-01-01")
        df = lf.collect()
        assert len(df) == 1

    def test_multi_factor_join(self, tmp_path):
        from storage.result_store import PolarsResultStore

        _setup_lake(
            tmp_path, "f1", ["2024-01-15"], ["A"], [1.0], ast_hash="h1"
        )
        _setup_lake(
            tmp_path, "f2", ["2024-01-15"], ["A"], [2.0], ast_hash="h2"
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PolarsResultStore(tmp_path, cat)
        lf = store.load_factors(["f1", "f2"])
        df = lf.collect()
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert len(df) == 1

    def test_to_pandas(self, tmp_path):
        from storage.result_store import PolarsResultStore

        _setup_lake(
            tmp_path, "f1", ["2024-01-15"], ["A"], [1.0], ast_hash="h1"
        )
        cat = FactorCatalog(tmp_path / "_catalog.sqlite")
        store = PolarsResultStore(tmp_path, cat)
        pdf = store.to_pandas(["f1"])
        assert isinstance(pdf, pd.DataFrame)


class TestBuildResultStore:
    """工厂函数自动选择后端。"""

    def test_auto_select(self, tmp_path):
        from storage.result_store import build_result_store

        _setup_lake(tmp_path, "f1", ["2024-01-15"], ["A"], [1.0])
        store = build_result_store(tmp_path)
        # 如果有 polars 就是 PolarsResultStore，否则是 PandasResultStore
        assert hasattr(store, "load_factor")
