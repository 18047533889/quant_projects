"""data_access.r30.training —— R30-P2-001 ``TrainingDatasetSpec`` 训练数据规格。

铺接口,不过度工程化:
  - 描述「训练数据如何切分 / 清洗 / 归一化」;
  - **DataAccess 负责 dataset assembly / snapshot**(``assemble`` 委托 store);
  - **Model layer 负责训练**(本模块不引入 sklearn / numpy 等训练依赖)。

``experiment_snapshot`` 由其他 agent 建设的 ``data_access.r30.experiment_snapshot``
承载,可能尚未就绪 —— 所有对它的访问都走懒加载 + try/except,绝不阻塞本模块。
"""
from __future__ import annotations

from typing import Any

__all__ = ["TrainingDatasetSpec"]


def _experiment_snapshot_digest(snapshot: Any) -> str | None:
    """防御性折叠 ``experiment_snapshot`` 摘要。

    优先用对象自带的 ``digest()``(测试里可用假对象);否则懒加载官方模块的
    ``snapshot_digest`` 辅助函数(模块未建成 → 静默返回 None)。
    """
    if snapshot is None:
        return None
    dig = getattr(snapshot, "digest", None)
    if callable(dig):
        try:
            out = dig()
            if out is not None:
                return str(out)
        except Exception:  # noqa: BLE001 —— 防御性
            pass
    try:
        import importlib

        mod = importlib.import_module("data_access.r30.experiment_snapshot")
        helper = getattr(mod, "snapshot_digest", None)
        if callable(helper):
            out = helper(snapshot)
            if out is not None:
                return str(out)
    except Exception:  # noqa: BLE001 —— 模块可能尚未建成
        pass
    return None


def _as_tuple(value: Any, what: str) -> tuple:
    """把 ``features/labels`` 收成 tuple;单个 str 视作一个成员。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _segments_overlap(seg_a: Any, seg_b: Any) -> bool:
    """两个 ``(start, end)`` 区间是否重叠(端点类型不可比较 → 保守判重叠)。"""
    for seg in (seg_a, seg_b):
        if not isinstance(seg, (tuple, list)) or len(seg) != 2:
            return False
        if seg[0] is None or seg[1] is None:
            return False
    try:
        return bool(seg_a[1] > seg_b[0] and seg_b[1] > seg_a[0])
    except TypeError:
        return True


class TrainingDatasetSpec:
    """训练数据集规格 —— 描述数据窗口、划分、清洗与归一化策略。

    参数:
        features: 特征列(非空);labels: 标签列(非空);
        universe: 标的/宇宙(作为 read_joined anchor);
        time_range: 全局时间窗口 ``(start, end)``;
        train_valid_test: ``((train_start, train_end), (valid_start, valid_end),
            (test_start, test_end))`` —— 三段互不重叠;
        purge / embargo: 数据泄漏防护(>= 0);
        missing_policy: 缺失值策略(drop/fill…);normalization_policy: 归一化;
        experiment_snapshot: ``ExperimentDataSnapshot``(懒访问,可能未建成)。
    """

    def __init__(
        self,
        features: tuple,
        labels: tuple,
        universe: str,
        time_range: tuple | None = None,
        train_valid_test: tuple | None = None,
        purge: int = 0,
        embargo: int = 0,
        missing_policy: str = "drop",
        normalization_policy: str = "none",
        experiment_snapshot: Any | None = None,
    ) -> None:
        self.features = _as_tuple(features, "features")
        self.labels = _as_tuple(labels, "labels")
        self.universe = universe
        self.time_range = tuple(time_range) if time_range is not None else None
        self.train_valid_test = (
            tuple(tuple(seg) for seg in train_valid_test)
            if train_valid_test is not None
            else None
        )
        self.purge = int(purge)
        self.embargo = int(embargo)
        self.missing_policy = missing_policy
        self.normalization_policy = normalization_policy
        self.experiment_snapshot = experiment_snapshot

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """返回全部违规项(空 list = 合法)。"""
        errors: list[str] = []
        if self.purge < 0:
            errors.append(f"purge 必须 >= 0 (got {self.purge})")
        if self.embargo < 0:
            errors.append(f"embargo 必须 >= 0 (got {self.embargo})")
        if not self.features:
            errors.append("features 不能为空")
        if not self.labels:
            errors.append("labels 不能为空")
        if not self.universe:
            errors.append("universe 不能为空")

        tvt = self.train_valid_test
        if tvt is not None:
            if len(tvt) != 3:
                errors.append(f"train_valid_test 必须为 3 段 (现 {len(tvt)} 段)")
            for i, seg in enumerate(tvt):
                if not isinstance(seg, (tuple, list)) or len(seg) != 2:
                    errors.append(
                        f"train_valid_test[{i}] 必须是 (start, end) 二元组"
                    )
                    continue
                start, end = seg[0], seg[1]
                if start is None or end is None:
                    errors.append(f"train_valid_test[{i}] 端点不能为 None")
                elif start > end:
                    errors.append(f"train_valid_test[{i}] start 不能晚于 end")
            for i in range(len(tvt) - 1):
                if _segments_overlap(tvt[i], tvt[i + 1]):
                    errors.append(f"train_valid_test 段 {i} 与 {i + 1} 重叠")
        return errors

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """把规格折叠成纯数据 dict(snapshot 只取摘要,不携带对象本身)。"""
        return {
            "features": list(self.features),
            "labels": list(self.labels),
            "universe": self.universe,
            "time_range": (
                list(self.time_range) if self.time_range is not None else None
            ),
            "train_valid_test": (
                [list(seg) for seg in self.train_valid_test]
                if self.train_valid_test is not None
                else None
            ),
            "purge": self.purge,
            "embargo": self.embargo,
            "missing_policy": self.missing_policy,
            "normalization_policy": self.normalization_policy,
            "experiment_snapshot_digest": _experiment_snapshot_digest(
                self.experiment_snapshot
            ),
        }

    # ------------------------------------------------------------------
    def assemble(self, store: Any) -> Any:
        """返回 dataset assembly(接口桩)。

        DataAccess 负责 dataset assembly / snapshot;Model layer 负责训练。
        若 ``store`` 提供 ``read_joined`` 则防御性委托;否则返回描述性 dict
        (不抛异常)。真实实现在 store 接入后替换。
        """
        fn = getattr(store, "read_joined", None)
        if callable(fn):
            try:
                return fn(
                    anchor=self.universe,
                    fields={self.universe: list(self.features) + list(self.labels)},
                    time_range=self.time_range,
                    universe=self.universe,
                )
            except Exception as exc:  # noqa: BLE001 —— 防御性降级
                return self._assembly_description(exception=exc)
        return self._assembly_description()

    def _assembly_description(self, exception: BaseException | None = None) -> dict:
        desc: dict = {
            "universe": self.universe,
            "features": list(self.features),
            "labels": list(self.labels),
            "time_range": self.time_range,
            "train_valid_test": self.train_valid_test,
            "purge": self.purge,
            "embargo": self.embargo,
            "missing_policy": self.missing_policy,
            "normalization_policy": self.normalization_policy,
            "snapshot_digest": _experiment_snapshot_digest(self.experiment_snapshot),
            "assembled": False,
            "note": "TrainingDatasetSpec.assemble 为接口桩;真实 assembly "
                    "由 store.read_joined 完成(DataAccess 负责)。",
        }
        if exception is not None:
            desc["delegate_error"] = repr(exception)
        return desc
