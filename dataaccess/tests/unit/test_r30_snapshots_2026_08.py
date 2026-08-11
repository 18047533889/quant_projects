# -*- coding: utf-8 -*-
"""R30-P0-008 / R30-P0-009 / R30-P1-009 —— 快照身份测试。

    R30-P0-008  CalendarSnapshot 全量 digest（中间交易日 / early-close 敏感）
    R30-P1-009  UniverseSnapshot（members / policy_version 敏感）
    R30-P0-009  ExperimentDataSnapshot（dataset / calendar / universe / contract /
                registry / build SHA 任一变化 → snapshot_id 变化）

全部用假 store + 真实 MarketCalendar/MarketSession 对象构造，不需要真实数据。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

from data_access.read.session_calendar import (
    MarketCalendar,
    build_ashare_session,
    build_us_session,
)
from data_access.r30.calendar_snapshot import CalendarSnapshot, freeze_calendar_world
from data_access.r30.experiment_snapshot import (
    ExperimentDataSnapshot,
    current_experiment_snapshot,
    dataset_source_id,
)
from data_access.r30.universe_snapshot import UniverseSnapshot


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ashare_calendar(days, source="test"):
    return MarketCalendar(
        "ashare",
        trading_days=days,
        timezone="Asia/Shanghai",
        source=source,
        session=build_ashare_session(),
    )


def _us_calendar(days, early_close=None, source="test"):
    sess = build_us_session(early_close_dates=early_close or [])
    return MarketCalendar(
        "us",
        trading_days=days,
        timezone="America/New_York",
        source=source,
        session=sess,
    )


def _mv(source_epoch="epoch-1", has_manifest=True, gen=None):
    d = {
        "dataset": "ashare_stock_daily",
        "has_manifest": has_manifest,
        "mutation_owner": "dataaccess",
        "fresh": True,
        "source_epoch": source_epoch,
        "manifest_built_epoch": source_epoch,
        "manifest_epoch": source_epoch,
    }
    if gen:
        d["manifest_generation_id"] = gen
    return d


class FakeStore:
    """最小假 store：日历注入 + manifest + 三个指纹 + universe 解析。"""

    def __init__(
        self,
        calendars=None,
        manifest_map=None,
        registry_fp="REG-v1",
        contract_fp="CONTRACT-v1",
        sec_digest="SEC-v1",
    ):
        self._calendars = dict(calendars or {})
        self._locked = False
        self._manifest = dict(manifest_map or {})
        self._registry_fp = registry_fp
        self._contract_fp = contract_fp
        self._sec_digest = sec_digest

    def set_calendar(self, market, calendar):
        self._calendars[str(market).strip().lower()] = calendar

    def get_calendar(self, market):
        if not market:
            return None
        return self._calendars.get(str(market).strip().lower())

    def lock_calendars(self):
        self._locked = True

    @property
    def calendars_locked(self):
        return self._locked

    def manifest_version(self, dataset, **params):
        return self._manifest.get(
            dataset, {"dataset": dataset, "has_manifest": False}
        )

    def registry_fingerprint(self):
        return self._registry_fp

    def contract_ir_fingerprint(self):
        return self._contract_fp

    def _effective_security_digest(self):
        return self._sec_digest

    def _resolve_universe_instruments(self, universe, time_range=None, instruments=None):
        return list(self._manifest.get("_universe_" + universe, []))


def _base_exp_store():
    store = FakeStore(
        manifest_map={
            "ashare_stock_daily": _mv("epoch-1"),
            "ashare_stock_daily_eod": _mv("epoch-1"),
        }
    )
    store.set_calendar(
        "ashare",
        _ashare_calendar([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]),
    )
    return store


# ---------------------------------------------------------------------------
# R30-P0-008 CalendarSnapshot
# ---------------------------------------------------------------------------


def test_calendar_snapshot_middle_trading_day_sensitive():
    """中间某交易日改变（count/first/last 不变）→ snapshot_id 改变。"""
    days1 = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
    days2 = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 7), date(2024, 1, 5), date(2024, 1, 8)]
    a = CalendarSnapshot.build("ashare", _ashare_calendar(days1))
    b = CalendarSnapshot.build("ashare", _ashare_calendar(days2))
    # 旧粗 digest 只覆盖 count/first/last —— 这里显式证明它们全相等：
    assert len(a.trading_days) == len(b.trading_days)
    assert a.trading_days[0] == b.trading_days[0]
    assert a.trading_days[-1] == b.trading_days[-1]
    assert a.first_trading_day == b.first_trading_day
    assert a.last_trading_day == b.last_trading_day
    # 但全量 digest 变了：
    assert a.snapshot_id != b.snapshot_id


def test_calendar_snapshot_early_close_sensitive():
    """中间某个 early-close 日期改变 → snapshot_id 改变。"""
    ec1 = [date(2024, 7, 3), date(2024, 11, 28), date(2024, 12, 24)]
    ec2 = [date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)]
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    a = CalendarSnapshot.build("us", _us_calendar(days, early_close=ec1))
    b = CalendarSnapshot.build("us", _us_calendar(days, early_close=ec2))
    assert a.snapshot_id != b.snapshot_id
    # early-close 时间也进 digest（标记成 early_close_time:HH:MM）。
    assert "early_close_time:13:00" in a.early_close


def test_calendar_snapshot_session_and_tz_source_sensitive():
    """session schedule / timezone / source 任一变化 → snapshot_id 改变。"""
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    a = CalendarSnapshot.build("ashare", _ashare_calendar(days, source="registry"))
    b = CalendarSnapshot.build("ashare", _ashare_calendar(days, source="explicit"))
    assert a.snapshot_id != b.snapshot_id
    tz_swap = MarketCalendar(
        "ashare",
        trading_days=days,
        timezone="UTC",
        source="test",
        session=build_ashare_session(),
    )
    c = CalendarSnapshot.build("ashare", tz_swap)
    assert c.snapshot_id != a.snapshot_id
    # 午休段在 sessions 里（A 股 morning+afternoon）：
    assert len(a.sessions) == 2
    assert a.sessions[0][0] == "morning"
    assert a.sessions[1][0] == "afternoon"


def test_calendar_snapshot_from_store_none():
    store = FakeStore()
    assert CalendarSnapshot.from_store(store, "ashare") is None


def test_calendar_snapshot_from_store_and_digest():
    store = FakeStore()
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    store.set_calendar("ashare", _ashare_calendar(days))
    snap = CalendarSnapshot.from_store(store, "ashare")
    assert snap is not None
    assert snap.market == "ashare"
    direct = CalendarSnapshot.build("ashare", _ashare_calendar(days))
    assert snap.snapshot_id == direct.snapshot_id
    assert snap.to_dict()["trading_day_count"] == 3


def test_calendar_snapshot_source_version_includes_manifest_token():
    store = FakeStore(manifest_map={"ashare_calendar": _mv("epoch-9")})
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    store.set_calendar("ashare", _ashare_calendar(days, source="registry"))
    snap = CalendarSnapshot.from_store(store, "ashare")
    assert snap.source_version == "registry:epoch-9"
    # 日历数据版本变了 → source_version 变 → snapshot 变（内容相同也变）。
    store2 = FakeStore(manifest_map={"ashare_calendar": _mv("epoch-10")})
    store2.set_calendar("ashare", _ashare_calendar(days, source="registry"))
    snap2 = CalendarSnapshot.from_store(store2, "ashare")
    assert snap.snapshot_id != snap2.snapshot_id


def test_freeze_calendar_world_locks_and_snapshots():
    store = FakeStore()
    store.set_calendar("ashare", _ashare_calendar([date(2024, 1, 2)]))
    store.set_calendar("us", _us_calendar([date(2024, 1, 2)]))
    world = freeze_calendar_world(store)
    assert store.calendars_locked is True
    assert set(world) == {"ashare", "us"}
    assert all(isinstance(s, CalendarSnapshot) for s in world.values())
    assert world["ashare"].snapshot_id == CalendarSnapshot.build(
        "ashare", _ashare_calendar([date(2024, 1, 2)])
    ).snapshot_id


# ---------------------------------------------------------------------------
# R30-P1-009 UniverseSnapshot
# ---------------------------------------------------------------------------


def test_universe_snapshot_members_change():
    a = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002", "600000"], "v1", "v1")
    b = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002", "600001"], "v1", "v1")
    assert a.snapshot_id != b.snapshot_id


def test_universe_snapshot_policy_versions_change():
    a = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002"], "v1", "v1")
    b = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002"], "v2", "v1")
    c = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002"], "v1", "v2")
    d = UniverseSnapshot.build("csi300", "ashare", ["000001", "000002"], "v2", "v2")
    assert len({a.snapshot_id, b.snapshot_id, c.snapshot_id, d.snapshot_id}) == 4


def test_universe_snapshot_membership_digest_order_insensitive():
    d1 = UniverseSnapshot.membership_digest(["600000", "000001", "000002"])
    d2 = UniverseSnapshot.membership_digest(["000001", "000002", "600000"])
    assert d1 == d2
    assert d1 != UniverseSnapshot.membership_digest(["000001", "000002", "600999"])


def test_universe_snapshot_from_store_resolves():
    store = FakeStore(manifest_map={"_universe_csi300": ["000001", "600000"]})
    snap = UniverseSnapshot.from_store(store, "csi300", market="ashare")
    assert snap is not None
    assert snap.member_count == 2
    assert snap.members == ("000001", "600000")
    assert snap.snapshot_id == UniverseSnapshot.build(
        "csi300", "ashare", ["600000", "000001"], "1", "1"
    ).snapshot_id


def test_universe_snapshot_to_dict():
    snap = UniverseSnapshot.build("csi300", "ashare", ["000001", "600000"], "v1", "v2")
    d = snap.to_dict()
    assert d["universe_id"] == "csi300"
    assert d["member_count"] == 2
    assert d["members"] == ["000001", "600000"]
    assert "membership_digest" in d


# ---------------------------------------------------------------------------
# R30-P0-009 ExperimentDataSnapshot
# ---------------------------------------------------------------------------


def test_experiment_snapshot_basic():
    store = _base_exp_store()
    snap = ExperimentDataSnapshot.build(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}
    )
    assert snap.experiment_id == "exp1"
    assert snap.market == "ashare"
    assert snap.snapshot_id
    assert snap.calendar_snapshot is not None
    assert snap.registry_digest == "REG-v1"
    assert snap.semantic_contract_digest == "CONTRACT-v1"
    assert snap.datasets["ashare_stock_daily"] == "ashare_stock_daily:epoch-1"


def test_experiment_snapshot_dataset_change():
    store1 = _base_exp_store()
    store2 = _base_exp_store()
    store2._manifest["ashare_stock_daily"] = _mv("epoch-2")
    s1 = ExperimentDataSnapshot.build(store1, "exp1", "ashare", {"ashare_stock_daily": ""})
    s2 = ExperimentDataSnapshot.build(store2, "exp1", "ashare", {"ashare_stock_daily": ""})
    assert s1.snapshot_id != s2.snapshot_id


def test_experiment_snapshot_calendar_change():
    store1 = _base_exp_store()
    store2 = _base_exp_store()
    store2.set_calendar(
        "ashare",
        _ashare_calendar([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 7)]),
    )
    s1 = ExperimentDataSnapshot.build(store1, "exp1", "ashare", {"ashare_stock_daily": ""})
    s2 = ExperimentDataSnapshot.build(store2, "exp1", "ashare", {"ashare_stock_daily": ""})
    assert s1.snapshot_id != s2.snapshot_id


def test_experiment_snapshot_universe_change():
    store = _base_exp_store()
    u1 = UniverseSnapshot.build("csi300", "ashare", ["000001", "600000"], "v1", "v1")
    u2 = UniverseSnapshot.build("csi300", "ashare", ["000001", "600001"], "v1", "v1")
    s1 = ExperimentDataSnapshot.build(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}, universe=u1
    )
    s2 = ExperimentDataSnapshot.build(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}, universe=u2
    )
    assert s1.snapshot_id != s2.snapshot_id
    # universe 用 str id 时从 store 解析
    store._manifest["_universe_csi300"] = ["000001", "600000"]
    s3 = ExperimentDataSnapshot.build(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}, universe="csi300"
    )
    assert s3.universe_snapshot is not None
    assert s3.universe_snapshot.members == ("000001", "600000")


def test_experiment_snapshot_contract_change():
    store1 = _base_exp_store()
    store2 = _base_exp_store()
    store2._contract_fp = "CONTRACT-v2"
    s1 = ExperimentDataSnapshot.build(store1, "exp1", "ashare", {"ashare_stock_daily": ""})
    s2 = ExperimentDataSnapshot.build(store2, "exp1", "ashare", {"ashare_stock_daily": ""})
    assert s1.snapshot_id != s2.snapshot_id


def test_experiment_snapshot_registry_change():
    store1 = _base_exp_store()
    store2 = _base_exp_store()
    store2._registry_fp = "REG-v2"
    s1 = ExperimentDataSnapshot.build(store1, "exp1", "ashare", {"ashare_stock_daily": ""})
    s2 = ExperimentDataSnapshot.build(store2, "exp1", "ashare", {"ashare_stock_daily": ""})
    assert s1.snapshot_id != s2.snapshot_id


def test_experiment_snapshot_build_sha_sensitive():
    store = _base_exp_store()
    s1 = ExperimentDataSnapshot.build(store, "exp1", "ashare", {"ashare_stock_daily": ""})
    with patch(
        "data_access._build_meta.build_sha",
        return_value="0123456789abcdef0123456789abcdef01234567",
    ):
        s2 = ExperimentDataSnapshot.build(store, "exp1", "ashare", {"ashare_stock_daily": ""})
    assert s1.snapshot_id != s2.snapshot_id


def test_experiment_snapshot_build_sha_none_tolerated():
    store = _base_exp_store()
    with patch("data_access._build_meta.build_sha", return_value=None):
        snap = ExperimentDataSnapshot.build(
            store, "exp1", "ashare", {"ashare_stock_daily": ""}
        )
    assert snap.code_build_sha is None
    assert snap.snapshot_id


def test_experiment_snapshot_to_dict_traceable():
    store = _base_exp_store()
    snap = ExperimentDataSnapshot.build(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}
    )
    d = snap.to_dict()
    assert d["experiment_id"] == "exp1"
    assert d["market"] == "ashare"
    assert "ashare_stock_daily" in d["datasets"]
    assert d["calendar_snapshot"] is not None
    assert "snapshot_id" in d["calendar_snapshot"]
    assert d["semantic_contract_digest"] == "CONTRACT-v1"
    assert d["registry_digest"] == "REG-v1"
    assert "version_gate" in d and "api" in d["version_gate"]
    assert d["snapshot_id"] == snap.snapshot_id


def test_current_experiment_snapshot():
    store = _base_exp_store()
    snap = current_experiment_snapshot(
        store, "exp1", "ashare", {"ashare_stock_daily": ""}
    )
    assert isinstance(snap, ExperimentDataSnapshot)
    assert snap.experiment_id == "exp1"


def test_dataset_source_id_fallback_no_manifest():
    """无 manifest → stable_digest 兜底（不抛、确定）。"""
    store = FakeStore()
    sid = dataset_source_id(store, "some_dataset")
    assert sid is not None
    assert sid == dataset_source_id(store, "some_dataset")
