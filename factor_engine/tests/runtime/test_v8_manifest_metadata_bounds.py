import itertools
import pickle
import sqlite3

import pytest

import factor_engine.runtime.finite_manifest as module
from factor_engine.runtime.finite_manifest import FiniteFactorManifest, ManifestError
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.manifest_metadata_limits import (
    MAX_DEPENDENCIES_PER_FACTOR,
    MAX_FACTOR_NAME_UTF8,
    MAX_MANIFEST_ROW_METADATA_BYTES,
    bounded_utf8_size,
    controlled_wave_buffer_bytes,
)


class Factor:
    def __init__(self, name, dependencies=()):
        self.name = name
        self.dependencies = dependencies

    def __getstate__(self):
        return {"name": self.name, "padding": getattr(self, "padding", b"")}


class NoEncodeString(str):
    def encode(self, *_args, **_kwargs):
        raise AssertionError("full string encode must not be used for admission")


class RaisingDependenciesProperty(Factor):
    @property
    def dependencies(self):
        raise RuntimeError("dependencies property failed")

    @dependencies.setter
    def dependencies(self, _value):
        pass


class RaisingDependencyIterator:
    def __init__(self):
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            return "partial-must-not-persist"
        raise RuntimeError("dependency iteration failed")


class RaisingNameProperty(Factor):
    @property
    def name(self):
        raise RuntimeError("name property failed")

    @name.setter
    def name(self, _value):
        pass

    def __getstate__(self):
        return {}


class RaisingCloseAttributeIterator:
    def __init__(self, *, pause=0):
        self.pause = pause
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        self.done = True
        if self.pause:
            import time
            time.sleep(self.pause)
        return "source"

    def __getattribute__(self, name):
        if name == "close":
            raise RuntimeError("close attribute failed")
        return object.__getattribute__(self, name)


def test_short_metadata_is_unchanged_and_definition_budget_semantics_remain(tmp_path):
    factors = [Factor("alpha", ("source",)), Factor("beta", ("alpha",))]
    with FiniteFactorManifest.ingest(factors, tmp_path / "manifest.sqlite3") as manifest:
        assert [record.name for record in manifest.records()] == ["alpha", "beta"]
        assert manifest.dependent_ordinals("alpha") == (1,)
        sizes = [record.definition_bytes for record in manifest.records()]
        loaded = manifest.load_wave_bounded(
            start=0, max_items=2, max_definition_bytes=sum(sizes),
        )
        assert [item[3] for item in loaded] == ["alpha", "beta"]
        assert controlled_wave_buffer_bytes(2, sum(sizes)) == (
            sum(sizes) + 2 * MAX_MANIFEST_ROW_METADATA_BYTES
        )


def test_oversized_factor_name_rejects_ordinal_without_pickle_or_full_encode(
    tmp_path, monkeypatch,
):
    name = NoEncodeString("x" * (MAX_FACTOR_NAME_UTF8 + 1))
    monkeypatch.setattr(
        module, "_bounded_pickle",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not pickle")),
    )
    with FiniteFactorManifest.ingest(
        [Factor(name)], tmp_path / "manifest.sqlite3"
    ) as manifest:
        [record] = list(manifest.records())
        assert record.ordinal == 0
        assert record.name == "<invalid:0>"
        assert not record.valid
        assert record.error_code == "FACTOR_NAME_TOO_LARGE"
    assert bounded_utf8_size(name, MAX_FACTOR_NAME_UTF8) is None


def test_multibyte_name_uses_utf8_bytes_not_character_count(tmp_path):
    name = "量" * 400
    with FiniteFactorManifest.ingest(
        [Factor(name)], tmp_path / "manifest.sqlite3"
    ) as manifest:
        [record] = list(manifest.records())
        assert record.ordinal == 0
        assert record.error_code == "FACTOR_NAME_TOO_LARGE"
        assert record.name == "<invalid:0>"


@pytest.mark.parametrize("dependencies,error", [
    (("d" * 1025,), "DEPENDENCY_NAME_TOO_LARGE"),
    (itertools.repeat("dep"), "TOO_MANY_DEPENDENCIES"),
])
def test_dependency_metadata_rejects_ordinal_without_partial_rows(
    tmp_path, dependencies, error,
):
    with FiniteFactorManifest.ingest(
        [Factor("alpha", dependencies)], tmp_path / "manifest.sqlite3",
        deadline_seconds=5,
    ) as manifest:
        [record] = list(manifest.records())
        assert not record.valid
        assert record.error_code == error
        assert manifest._db.execute("select count(*) from dependencies").fetchone() == (0,)
        assert manifest._db.execute(
            "select definition,definition_bytes from factors where ordinal=0"
        ).fetchone() == (None, 0)
        if error == "TOO_MANY_DEPENDENCIES":
            assert MAX_DEPENDENCIES_PER_FACTOR == 4096


def test_invalid_definition_never_enumerates_unbounded_dependencies(tmp_path):
    factor = Factor("alpha", itertools.repeat("dep"))
    factor.__dict__["unpickleable"] = lambda: None
    factor.__getstate__ = None
    with FiniteFactorManifest.ingest(
        [factor], tmp_path / "manifest.sqlite3", max_definition_bytes=1,
    ) as manifest:
        [record] = list(manifest.records())
        assert not record.valid
        assert record.error_code in {"INVALID_FACTOR_DEFINITION", "DEFINITION_TOO_LARGE"}
        assert manifest._db.execute("select count(*) from dependencies").fetchone() == (0,)


@pytest.mark.parametrize("factor", [
    RaisingDependenciesProperty("bad"),
    Factor("bad", RaisingDependencyIterator()),
])
def test_dependency_access_failure_rejects_only_its_ordinal_and_clears_payload(
    tmp_path, factor,
):
    with FiniteFactorManifest.ingest(
        [factor, Factor("good", ("source",))], tmp_path / "manifest.sqlite3"
    ) as manifest:
        records = list(manifest.records())
        assert [(row.name, row.valid, row.error_code) for row in records] == [
            ("bad", False, "INVALID_FACTOR_DEPENDENCIES"),
            ("good", True, None),
        ]
        assert manifest._db.execute(
            "select definition,definition_bytes from factors where ordinal=0"
        ).fetchone() == (None, 0)
        assert manifest._db.execute(
            "select ordinal,dependency_name from dependencies order by ordinal"
        ).fetchall() == [(1, "source")]


def test_dependency_close_attribute_failure_is_per_ordinal(tmp_path):
    factor = Factor("bad", RaisingCloseAttributeIterator())
    with FiniteFactorManifest.ingest(
        [factor, Factor("good")], tmp_path / "manifest.sqlite3"
    ) as manifest:
        records = list(manifest.records())
        assert [(row.name, row.error_code) for row in records] == [
            ("bad", "INVALID_FACTOR_DEPENDENCIES"), ("good", None),
        ]
        assert manifest._db.execute(
            "select definition,definition_bytes from factors where ordinal=0"
        ).fetchone() == (None, 0)


def test_dependency_close_failure_does_not_replace_input_timeout(tmp_path):
    factor = Factor("slow", RaisingCloseAttributeIterator(pause=0.02))
    with pytest.raises(module.ManifestInputTimeout):
        FiniteFactorManifest.ingest(
            [factor], tmp_path / "manifest.sqlite3", deadline_seconds=0.005,
        )


def test_name_property_failure_rejects_only_its_ordinal(tmp_path):
    with FiniteFactorManifest.ingest(
        [RaisingNameProperty("ignored"), Factor("good")],
        tmp_path / "manifest.sqlite3",
    ) as manifest:
        records = list(manifest.records())
        assert [(row.name, row.error_code) for row in records] == [
            ("<invalid:0>", "INVALID_FACTOR_NAME"), ("good", None),
        ]


@pytest.mark.parametrize("column", ["name", "definition"])
def test_bounded_load_rejects_giant_existing_cells_before_deserialization(
    tmp_path, monkeypatch, column,
):
    path = tmp_path / "manifest.sqlite3"
    manifest = FiniteFactorManifest(path)
    payload = pickle.dumps(Factor("safe"))
    with manifest._db:
        manifest._db.execute(
            "insert into factors values(0,?,?,?,?,?,?)",
            ("safe", payload, len(payload), "a" * 64, 1, None),
        )
        if column == "name":
            manifest._db.execute("update factors set name=?", ("x" * 1_000_000,))
        else:
            manifest._db.execute(
                "update factors set definition=?,definition_bytes=1",
                (b"x" * 1_000_000,),
            )
    monkeypatch.setattr(
        module.pickle, "loads",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not deserialize")),
    )
    with pytest.raises(ManifestError, match="native SQLite bound"):
        manifest.load_wave_bounded(start=0, max_items=1, max_definition_bytes=1024)
    manifest.close()


def test_records_projection_rejects_giant_existing_text_without_returning_it(tmp_path):
    path = tmp_path / "manifest.sqlite3"
    with FiniteFactorManifest(path) as manifest:
        payload = pickle.dumps(Factor("safe"))
        with manifest._db:
            manifest._db.execute(
                "insert into factors values(0,?,?,?,?,?,?)",
                ("x" * 1_000_000, payload, len(payload), "a" * 64, 1, None),
            )
        with pytest.raises(ManifestError, match="metadata"):
            list(manifest.records())


@pytest.mark.parametrize("column", ["name", "error_code", "definition_digest"])
def test_nul_prefixed_long_tail_cannot_bypass_blob_length_projection(
    tmp_path, column,
):
    with FiniteFactorManifest(tmp_path / "manifest.sqlite3") as manifest:
        payload = pickle.dumps(Factor("safe"))
        with manifest._db:
            manifest._db.execute(
                "insert into factors values(0,?,?,?,?,?,?)",
                ("safe", payload, len(payload), "a" * 64, 0, "INVALID"),
            )
            manifest._db.execute(
                f"update factors set {column}=?",
                ("ok\0" + "x" * 100_000,),
            )
        with pytest.raises(ManifestError, match="metadata"):
            list(manifest.records())


@pytest.mark.parametrize("padding", [20 * 1024, 128 * 1024])
def test_records_and_state_registration_accept_legal_large_definitions(
    tmp_path, padding,
):
    factor = Factor("large")
    factor.padding = b"x" * padding
    with FiniteFactorManifest.ingest(
        [factor], tmp_path / "manifest.sqlite3",
        max_definition_bytes=256 * 1024,
    ) as manifest:
        [record] = list(manifest.records())
        assert record.valid and record.definition_bytes >= padding
        state = PersistentRunState(tmp_path / "state.sqlite3")
        try:
            for item in manifest.records():
                state.register(item.ordinal, item.name)
            assert state.counts() == {"ACCEPTED": 1}
        finally:
            state.close()


def test_records_ignore_oversized_invalid_definition_blob(tmp_path):
    with FiniteFactorManifest(tmp_path / "manifest.sqlite3") as manifest:
        with manifest._db:
            manifest._db.execute(
                "insert into factors values(0,?,?,?,?,?,?)",
                ("bad", b"x" * 128_000, 131_073, "", 0,
                 "DEFINITION_TOO_LARGE"),
            )
        [record] = list(manifest.records())
        assert not record.valid
        assert record.error_code == "DEFINITION_TOO_LARGE"


def test_nested_bounded_load_restores_paused_records_length_limit(tmp_path):
    factors = [Factor("alpha"), Factor("beta")]
    with FiniteFactorManifest.ingest(
        factors, tmp_path / "manifest.sqlite3"
    ) as manifest:
        original = manifest._db.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
        records = manifest.records(limit=2)
        assert next(records).name == "alpha"
        # records() now uses projection only, so it must not tighten a limit
        # across yield and accidentally constrain a nested wave read.
        assert manifest._db.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) == original
        manifest.load_wave_bounded(
            start=0, max_items=1, max_definition_bytes=1024,
        )
        assert manifest._db.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) == original
        records.close()


def test_dependency_lookup_rejects_oversized_query_without_sqlite_bind(
    tmp_path, monkeypatch,
):
    with FiniteFactorManifest(tmp_path / "manifest.sqlite3") as manifest:
        name = NoEncodeString("x" * 1025)
        with pytest.raises(ManifestError, match="dependency name"):
            manifest.dependent_ordinals(name)


def test_serialization_lease_scope_distinguishes_runs_for_same_input_object(tmp_path):
    lease_ids = []

    class Lease:
        def release(self):
            pass

    class Broker:
        def acquire_memory(self, _kind, _size, *, lease_id):
            lease_ids.append(lease_id)
            return Lease()

    factors = [Factor("alpha")]
    for run_name in ("run-a", "run-b", "run-a"):
        run_dir = tmp_path / run_name
        run_dir.mkdir(exist_ok=True)
        with FiniteFactorManifest.ingest_supervised(
            factors, run_dir / "manifest.sqlite3", process_context="fork",
            serialization_broker=Broker(), deadline_seconds=2,
        ):
            pass
    assert len(lease_ids) == 3
    assert lease_ids[0] != lease_ids[1]
    assert lease_ids[0] == lease_ids[2]
    assert all(item.startswith("manifest-ingestion:") for item in lease_ids)
