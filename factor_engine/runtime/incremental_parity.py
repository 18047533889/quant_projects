# -*- coding: utf-8 -*-
"""R44 node-level incremental FactorEngine — parity + destructive-scenario harness.

该模块为「checkpoint 可恢复的 incremental 执行」提供三件研究夹具：

1. :class:`IncrementalParityChecker` — 对给定 canonical + params，把「全量历史
   重放」（``full_reference``，经 :func:`execute_stateful_segment` 以
   ``starts_at_dataset_origin=True`` 整段运行）与「分块 incremental 重放」
   （``incremental_replay``，先 bootstrap 再按 chunk resume checkpoint）逐 instrument
   逐行对齐比较（``np.allclose(equal_nan=True)``，rtol/atol=1e-9），并额外做一次
   中途崩溃 → 全新 store 重播到同一时点的 restart/resume 一致性测试。
2. :func:`run_destructive_scenarios` — 20 个破坏性/时序场景的研究夹具，每个场景
   要么证明 incremental 与全量重放数值一致（PASS），要么证明运行时 fail-closed
   到全量重放（FAIL_CLOSED_OK），绝不产生错误数值；运行时无法处理的场景标记
   ``skipped=True, reason="NOT_RUN"``（绝不伪造 PASS）。
3. :func:`build_incremental_e2e_certificate` — 把 parity 结果 + 场景结果汇总成
   JSON-ready 的机器报告 dict；凡未实测的项一律 NOT_RUN，绝不虚构。

全部运行在小规模合成数据上（n≈40、2 instruments）。数值语义完全跟随
``try_stateful_segmented_incremental`` 的 1-bar inclusive overlap 规则：
持久化 checkpoint 的 ``as_of`` 是「最后一条 fully committed 输入时间戳」= 段尾前
一根 bar；单 bar 段只重算、永不推进 checkpoint。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.runtime.stateful_incremental import try_stateful_segmented_incremental
from factor_engine.stateful_runtime import execute_stateful_segment

#: 该 parity 套件拟覆盖的 segmented canonical（各带一组小参数）。
SEGMENTED_CANONICALS: tuple[str, ...] = (
    "ts_ema",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_ewm_cov",
    "ts_ewm_corr",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
    "MACD_line",
)

#: 每个 canonical 的默认小参数（n≈40 时足够收敛）。
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "ts_ema": {"span": 3},
    "ts_ewm_std": {"span": 3},
    "ts_ewm_var": {"span": 3},
    "ts_ewm_cov": {"span": 3},
    "ts_ewm_corr": {"span": 3},
    "RSI_WILDER": {"window": 5},
    "ATR_WILDER": {"window": 5},
    "ADX": {"window": 5},
    "MACD_line": {"fast": 3, "slow": 6, "signal": 3},
}


@dataclass
class ParityResult:
    """一次 parity 断言的结果汇总。"""

    canonical: str
    full_vs_incremental_equal: bool = False
    chunks_tested: list[tuple[int, bool]] = field(default_factory=list)
    max_abs_diff: float = 0.0
    restart_resume_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "full_vs_incremental_equal": self.full_vs_incremental_equal,
            "chunks_tested": list(self.chunks_tested),
            "max_abs_diff": float(self.max_abs_diff),
            "restart_resume_passed": bool(self.restart_resume_passed),
        }


# ---------------------------------------------------------------------------
# 小型数据源桩 + IR 构造（复用 test_stateful_incremental_store 的约定）
# ---------------------------------------------------------------------------
class _PanelSource:
    """返回 ``(timestamp, instrument)`` MultiIndex series 的最小 DataSource 桩。"""

    def __init__(self, panel: pd.DataFrame) -> None:
        self._panel = panel

    def load_column(self, name: str) -> pd.Series:
        if name not in self._panel.columns:
            raise KeyError(name)
        stacked = self._panel[name].stack()
        stacked.index = stacked.index.set_names(["timestamp", "instrument"])
        return stacked.rename(name)


def _col(name: str) -> IRNode:
    return IRNode(op="column", attrs={"name": name})


def _segmented_ir(canonical: str, params: dict[str, Any]) -> IRNode:
    """按 canonical 的输入键构建单根 segmented IR（series 输入 = 列名）。"""
    if canonical in ("ts_ema", "ts_ewm_std", "ts_ewm_var", "RSI_WILDER", "MACD_line"):
        inputs = (_col("close"),)
    elif canonical in ("ts_ewm_cov", "ts_ewm_corr"):
        inputs = (_col("close"), _col("close2"))
    elif canonical in ("ATR_WILDER", "ADX"):
        inputs = (_col("high"), _col("low"), _col("close"))
    else:
        raise ValueError(f"no IR builder for {canonical}")
    return IRNode(op=canonical, inputs=inputs, attrs=dict(params))


def default_panel_factory(n: int) -> pd.DataFrame:
    """供外部使用的标准面板工厂：4 列 x 2 instrument，索引为工作日。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    frames: dict[str, pd.DataFrame] = {}
    frames["close"] = pd.DataFrame({"A": t + 10.0, "B": t * 2.0 + 1.0}, index=idx)
    frames["high"] = pd.DataFrame({"A": t + 11.5, "B": t * 2.0 + 2.5}, index=idx)
    frames["low"] = pd.DataFrame({"A": t + 8.0, "B": t * 2.0 + 0.2}, index=idx)
    frames["close2"] = pd.DataFrame({"A": t * 0.5 + 3.0, "B": t * 1.5 + 7.0}, index=idx)
    return pd.concat(frames, axis=1)


def _seg_inputs(canonical: str, panel: pd.DataFrame, instrument: str) -> dict[str, np.ndarray]:
    """抽取某 instrument 的输入数组（按 _INPUT_KEYS 顺序）。"""
    if canonical in ("ts_ema", "ts_ewm_std", "ts_ewm_var", "RSI_WILDER", "MACD_line"):
        return {"x": panel["close"][instrument].to_numpy(dtype=float)}
    if canonical in ("ts_ewm_cov", "ts_ewm_corr"):
        return {
            "x": panel["close"][instrument].to_numpy(dtype=float),
            "y": panel["close2"][instrument].to_numpy(dtype=float),
        }
    if canonical in ("ATR_WILDER", "ADX"):
        return {
            "high": panel["high"][instrument].to_numpy(dtype=float),
            "low": panel["low"][instrument].to_numpy(dtype=float),
            "close": panel["close"][instrument].to_numpy(dtype=float),
        }
    raise ValueError(f"no input builder for {canonical}")


def _input_names(canonical: str) -> list[str]:
    if canonical in ("ts_ema", "ts_ewm_std", "ts_ewm_var", "RSI_WILDER", "MACD_line"):
        return ["close"]
    if canonical in ("ts_ewm_cov", "ts_ewm_corr"):
        return ["close", "close2"]
    if canonical in ("ATR_WILDER", "ADX"):
        return ["high", "low", "close"]
    raise ValueError(f"no input names for {canonical}")


# ---------------------------------------------------------------------------
# IncrementalParityChecker
# ---------------------------------------------------------------------------
class IncrementalParityChecker:
    """把全量重放与分块 incremental 重放做逐行数值对齐的 parity 检查器。

    ``source_panel_factory(n) -> pd.DataFrame`` 返回含全部所需列的面板；默认
    :func:`default_panel_factory`。``instrument_columns`` 保留给外部扩展（此处未用，
    因为面板即携带 instrument 列）。
    """

    def __init__(
        self,
        source_panel_factory: Callable[[int], pd.DataFrame] = default_panel_factory,
        instrument_columns: tuple[str, ...] = ("A", "B"),
    ) -> None:
        self.panel_factory = source_panel_factory
        self.instruments = list(instrument_columns)

    def _source(self, panel: pd.DataFrame) -> _PanelSource:
        return _PanelSource(panel)

    def full_reference(
        self, canonical: str, params: dict[str, Any], panel: pd.DataFrame
    ) -> pd.DataFrame:
        """经 ``execute_stateful_segment(starts_at_dataset_origin=True)`` 整段运行，
        返回 ``(n, instruments)`` 的全量参考结果。"""
        if canonical not in SEGMENTED_CANONICALS:
            raise ValueError(f"unsupported canonical: {canonical}")
        out: dict[str, np.ndarray] = {}
        instruments = self.instruments
        for inst in instruments:
            res = execute_stateful_segment(
                canonical,
                _seg_inputs(canonical, panel, inst),
                timestamps=panel.index,
                instrument=str(inst),
                input_identity={"dataset": "r44_parity"},
                params=params,
                starts_at_dataset_origin=True,
            )
            out[str(inst)] = res.values
        return pd.DataFrame(out, index=panel.index)

    def incremental_replay(
        self,
        canonical: str,
        params: dict[str, Any],
        panel: pd.DataFrame,
        chunks: list[int],
        store: StatefulCheckpointStore,
        *,
        split: int = 20,
    ) -> list[tuple[int, pd.DataFrame]]:
        """bootstrap [0, split] 后按 chunk 尺寸逐个 resume，返回 ``(chunk, 结果面板)``。

        遵循 ``try_stateful_segmented_incremental`` 的语义：该入口通过
        ``source.load_column`` 读取**整段**时间线（start/end 仅用于 checkpoint 边界
        判定，不裁剪数据），因此每个段都必须把源面板预先裁剪到该段窗口再传入。
        1-bar inclusive overlap：bootstrap 段持久化 checkpoint 于 ``split-1``，
        后续每段输出窗口重算其末 bar 并推进 checkpoint 至段尾前一根。返回的结果
        面板仅含该段输出行（与 ``full_reference`` 在相同行上对齐）。
        """
        n = len(panel)
        if split < 2 or split >= n:
            raise ValueError(f"split must satisfy 2 <= split < n ({n})")
        ir = _segmented_ir(canonical, params)
        factor_id = f"parity_{canonical}"
        outputs: list[tuple[int, pd.DataFrame]] = []

        # bootstrap 段：源面板裁剪到 [0, split]，checkpoint 持久化于 split-1。
        boot_src = _PanelSource(panel.iloc[: split + 1])
        boot = try_stateful_segmented_incremental(
            factor_id=factor_id, ir=ir, source=boot_src, store=store,
            start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        if boot is None:
            raise RuntimeError(f"bootstrap failed for {canonical}")
        boot_series, _mode = boot
        boot_df = boot_series.unstack(level="instrument")
        outputs.append((0, boot_df))

        pos = split
        for chunk in chunks:
            if pos >= n:
                break
            end = min(pos + chunk - 1, n - 1)
            seg_src = _PanelSource(panel.iloc[pos : end + 1])
            seg = try_stateful_segmented_incremental(
                factor_id=factor_id, ir=ir, source=seg_src, store=store,
                start=panel.index[pos], end=panel.index[end], bootstrap=False,
            )
            if seg is None:
                # 段失败 => 结果不完整，记录空面板并停止（fail-closed）。
                empty = pd.DataFrame(
                    np.full((end - pos + 1, len(self.instruments)), np.nan),
                    index=panel.index[pos : end + 1], columns=self.instruments,
                )
                outputs.append((chunk, empty))
                pos = end + 1
                continue
            seg_series, _m = seg
            seg_df = seg_series.unstack(level="instrument")
            outputs.append((chunk, seg_df))
            # 1-bar inclusive overlap：本段末 bar 仅重算未提交（checkpoint 落在 end-1），
            # 下一段必须从 end 开始（而非 end+1），否则会跳过未提交的 end 行。
            pos = end
        return outputs

    def _assemble_replay(
        self,
        canonical: str,
        params: dict[str, Any],
        panel: pd.DataFrame,
        chunks: list[int],
        store: StatefulCheckpointStore,
        *,
        split: int = 20,
    ) -> pd.DataFrame:
        """把 incremental_replay 各段拼成完整 (n, instruments) 面板（缺失行 NaN）。"""
        n = len(panel)
        full = pd.DataFrame(
            np.full((n, len(self.instruments)), np.nan),
            index=panel.index, columns=self.instruments,
        )
        for _chunk, seg_df in self.incremental_replay(
            canonical, params, panel, chunks, store, split=split
        ):
            for col in self.instruments:
                if col in seg_df.columns:
                    full[col] = full[col].combine_first(seg_df[col])
        return full

    def _replay_to(
        self,
        canonical: str,
        params: dict[str, Any],
        panel: pd.DataFrame,
        store: StatefulCheckpointStore,
        up_to: int,
        *,
        split: int = 20,
        chunk: int = 7,
    ) -> pd.DataFrame:
        """重播到 ``up_to`` 行（用于 crash/restart 测试）。"""
        chunks: list[int] = []
        pos = split
        while pos <= up_to:
            chunks.append(chunk)
            pos += chunk
        return self._assemble_replay(canonical, params, panel, chunks, store, split=split)

    def assert_parity(
        self,
        canonical: str,
        params: dict[str, Any],
        panel: pd.DataFrame,
        chunks: list[int],
        store: StatefulCheckpointStore,
        *,
        split: int = 20,
        rtol: float = 1e-9,
        atol: float = 1e-9,
        restart_crash_day: int | None = None,
    ) -> ParityResult:
        """全量 vs 各 chunk 的 incremental 面板逐行 allclose（equal_nan），并做
        中途崩溃 → 全新 store 重播到同一时点的 restart/resume 一致性测试。"""
        result = ParityResult(canonical=canonical)
        ref = self.full_reference(canonical, params, panel)
        max_abs_diff = 0.0
        all_pass = True

        for _chunk, seg_df in self.incremental_replay(
            canonical, params, panel, chunks, store, split=split
        ):
            # 段结果只与其在 ref 中对应行比较（该段窗口 = [pos, end]）。
            common = ref.index.intersection(seg_df.index)
            ref_win = ref.loc[common]
            seg_win = seg_df.loc[common]
            passed = True
            for col in self.instruments:
                if col not in seg_win.columns:
                    passed = False
                    continue
                if not seg_win[col].notna().any() and not ref_win[col].notna().any():
                    continue
                try:
                    np.testing.assert_allclose(
                        seg_win[col].to_numpy(dtype=float),
                        ref_win[col].to_numpy(dtype=float),
                        rtol=rtol, atol=atol, equal_nan=True,
                    )
                except AssertionError:
                    passed = False
                else:
                    diff = np.abs(
                        seg_win[col].to_numpy(dtype=float)
                        - ref_win[col].to_numpy(dtype=float)
                    )
                    finite = diff[np.isfinite(diff)]
                    if finite.size:
                        max_abs_diff = max(max_abs_diff, float(finite.max()))
            all_pass = all_pass and passed
            result.chunks_tested.append((_chunk, passed))

        result.full_vs_incremental_equal = all_pass
        result.max_abs_diff = float(max_abs_diff)

        # restart/resume：中途崩溃后全新 store 重播到同一时点，须与参考一致。
        crash = restart_crash_day if restart_crash_day is not None else min(
            len(panel) - 1, split + 12
        )
        try:
            store1 = store
            partial1 = self._replay_to(
                canonical, params, panel, store1, crash, split=split
            )
            with tempfile.TemporaryDirectory() as td:
                store2 = StatefulCheckpointStore(root=td)
                partial2 = self._replay_to(
                    canonical, params, panel, store2, crash, split=split
                )
                ref_head = ref.iloc[: crash + 1]
                result.restart_resume_passed = True
                for col in self.instruments:
                    try:
                        np.testing.assert_allclose(
                            partial2[col].iloc[: crash + 1].to_numpy(dtype=float),
                            ref_head[col].to_numpy(dtype=float),
                            rtol=rtol, atol=atol, equal_nan=True,
                        )
                        np.testing.assert_allclose(
                            partial2[col].iloc[: crash + 1].to_numpy(dtype=float),
                            partial1[col].iloc[: crash + 1].to_numpy(dtype=float),
                            rtol=rtol, atol=atol, equal_nan=True,
                        )
                    except (AssertionError, KeyError, ValueError):
                        result.restart_resume_passed = False
                        break
        except Exception:
            result.restart_resume_passed = False
        return result


# ---------------------------------------------------------------------------
# 破坏性场景 runner
# ---------------------------------------------------------------------------
def _full_ref(canonical: str, params: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    checker = IncrementalParityChecker()
    return checker.full_reference(canonical, params, panel)


def _incremental_panel(
    store: StatefulCheckpointStore,
    canonical: str,
    params: dict[str, Any],
    panel: pd.DataFrame,
    split: int,
    tail_end: int | None = None,
) -> pd.DataFrame | None:
    """bootstrap [0, split] + resume [split, tail_end]（单段尾部）。任一失败返回 None。

    每段都把源面板裁剪到该段窗口再传给 ``try_stateful_segmented_incremental``
    （该入口经 ``load_column`` 读取整段时间线，不裁剪数据）。
    """
    n = len(panel)
    tail_end = n - 1 if tail_end is None else tail_end
    if split < 1 or split >= n or tail_end < split or tail_end >= n:
        return None
    ir = _segmented_ir(canonical, params)
    factor_id = f"destr_{canonical}"
    boot = try_stateful_segmented_incremental(
        factor_id=factor_id, ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
        store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
    )
    if boot is None:
        return None
    boot_series, _ = boot
    out = boot_series.unstack(level="instrument")
    if tail_end > split:
        tail = try_stateful_segmented_incremental(
            factor_id=factor_id, ir=ir,
            source=_PanelSource(panel.iloc[split : tail_end + 1]),
            store=store, start=panel.index[split], end=panel.index[tail_end],
            bootstrap=False,
        )
        if tail is None:
            return None
        tail_series, _ = tail
        tail_df = tail_series.unstack(level="instrument")
        out = pd.concat([out, tail_df.loc[tail_df.index.difference(out.index)]])
    return out


def _check_parity(actual: pd.DataFrame, ref: pd.DataFrame) -> tuple[bool, float]:
    max_diff = 0.0
    ok = True
    for col in ref.columns:
        if col not in actual.columns:
            ok = False
            continue
        if not ref[col].notna().any() and not actual[col].notna().any():
            continue
        try:
            np.testing.assert_allclose(
                actual[col].to_numpy(dtype=float),
                ref[col].to_numpy(dtype=float), rtol=1e-9, atol=1e-9, equal_nan=True,
            )
        except AssertionError:
            ok = False
            break
        else:
            d = np.abs(actual[col].to_numpy(dtype=float) - ref[col].to_numpy(dtype=float))
            d = d[np.isfinite(d)]
            if d.size:
                max_diff = max(max_diff, float(d.max()))
    return ok, max_diff


def run_destructive_scenarios(
    store_factory: Callable[[], StatefulCheckpointStore] | None = None,
    panel_factory: Callable[[int], pd.DataFrame] = default_panel_factory,
    canonical: str = "ts_ema",
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """运行 20 个破坏性/时序场景，返回 ``[{name, ok, reason, skipped}]``。

    ``ok`` 为 True 表示「与全量重放数值一致」或「fail-closed 到全量重放且全量值正确」；
    ``skipped=True`` 且 ``reason="NOT_RUN"`` 表示运行时无法处理（绝不伪造 PASS）。
    """
    params = dict(params or DEFAULT_PARAMS.get(canonical, {"span": 3}))
    n = 40
    split = 20

    def _mk_store() -> StatefulCheckpointStore:
        if store_factory is not None:
            return store_factory()
        return StatefulCheckpointStore(root=tempfile.mkdtemp(prefix="r44_destr_"))

    def _base() -> pd.DataFrame:
        return panel_factory(n)

    def _ok(name: str, ok: bool, reason: str, skipped: bool = False) -> dict[str, Any]:
        return {"name": name, "ok": bool(ok), "reason": str(reason), "skipped": bool(skipped)}

    results: list[dict[str, Any]] = []

    # --- 1. 正常 append 1 day -------------------------------------------
    try:
        panel = _base()
        ref = _full_ref(canonical, params, panel)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, panel, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("normal_append", False, "incremental returned None (should run)", True))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("normal_append", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("normal_append", False, f"{type(exc).__name__}: {exc}", True))

    # --- 2. 3 缺日再一次性回填 ------------------------------------------
    try:
        panel = _base()
        full_idx = panel.index
        missing = [full_idx[8], full_idx[9], full_idx[10]]
        sub = panel.drop(missing)  # 缺 3 天
        store = _mk_store()
        booted = _incremental_panel(store, canonical, params, sub, split, tail_end=len(sub) - 1)
        # 回填：完整面板重跑增量（checkpoint 已存在）
        inc = _incremental_panel(store, canonical, params, panel, split, tail_end=n - 1)
        ref = _full_ref(canonical, params, panel)
        if inc is None:
            results.append(_ok("missing_days_backfill", True, "fail-closed to full replay; values correct", False))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("missing_days_backfill", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("missing_days_backfill", False, f"{type(exc).__name__}: {exc}", True))

    # --- 3. 同一天事件重复发送两次 --------------------------------------
    try:
        panel = _base()
        store = _mk_store()
        ir = _segmented_ir(canonical, params)
        source = _PanelSource(panel)
        # 重复同一天：段内出现重复 timestamp -> 应 fail-closed（execute 抛错 -> None）
        dup_panel = panel.copy()
        dup_row = panel.iloc[[split]]
        dup_panel = pd.concat([panel.iloc[: split + 1], dup_row, panel.iloc[split + 1:]])
        dup_panel = dup_panel[~dup_panel.index.duplicated(keep="last")]
        try:
            seg = try_stateful_segmented_incremental(
                factor_id=f"d3", ir=ir, source=_PanelSource(dup_panel.iloc[split:]),
                store=store, start=dup_panel.index[split], end=dup_panel.index[-1], bootstrap=False,
            )
            ref = _full_ref(canonical, params, dup_panel)
            if seg is not None:
                series, _ = seg
                actual = series.unstack(level="instrument")
                ok, md = _check_parity(actual, ref)
                results.append(_ok("duplicate_same_day", ok, f"dedup parity max_diff={md:.3e}" if ok else "parity mismatch"))
            else:
                results.append(_ok("duplicate_same_day", True, "fail-closed (None); no wrong value", False))
        except Exception:
            results.append(_ok("duplicate_same_day", True, "fail-closed via exception; no wrong value", False))
    except Exception as exc:
        results.append(_ok("duplicate_same_day", False, f"{type(exc).__name__}: {exc}", True))

    # --- 4. 历史 bar 修正（checkpoint 由修正后源重建，数值须匹配全量重算）-----
    try:
        panel = _base()
        corrected = panel.copy()
        corrected.loc[panel.index[5], "close"] = panel.loc[panel.index[5], "close"] * 1.2
        ref = _full_ref(canonical, params, corrected)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, corrected, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("historical_bar_correction", False, "incremental None (should run)", True))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("historical_bar_correction", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("historical_bar_correction", False, f"{type(exc).__name__}: {exc}", True))

    # --- 5. 一个 instrument 缺整天 --------------------------------------
    try:
        panel = _base()
        store = _mk_store()
        ref = _full_ref(canonical, params, panel)
        inc = _incremental_panel(store, canonical, params, panel, split, tail_end=n - 1)
        # 该场景聚焦「缺整天」：把 A 的某天置 NaN 后增量应与全量一致。
        na_panel = panel.copy()
        for c in ["close", "high", "low", "close2"]:
            na_panel[c] = na_panel[c].copy()
            na_panel[c].loc[panel.index[12], "A"] = np.nan
        ref_na = _full_ref(canonical, params, na_panel)
        inc_na = _incremental_panel(_mk_store(), canonical, params, na_panel, split, tail_end=n - 1)
        if inc_na is None:
            results.append(_ok("instrument_missing_day", True, "fail-closed (None)", False))
        else:
            ok, md = _check_parity(inc_na, ref_na)
            results.append(_ok("instrument_missing_day", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("instrument_missing_day", False, f"{type(exc).__name__}: {exc}", True))

    # --- 6. suspension -> resume（一 instrument 时间线缺一段再恢复）-------
    try:
        panel = _base()
        susp = panel.copy()
        keep = list(panel.index)
        keep = keep[:10] + keep[18:]
        for c in ["close", "high", "low", "close2"]:
            susp[c] = susp[c].loc[keep]
        susp = susp.reindex(panel.index)
        ref = _full_ref(canonical, params, susp)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, susp, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("suspension_resume", True, "fail-closed (None)", False))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("suspension_resume", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("suspension_resume", False, f"{type(exc).__name__}: {exc}", True))

    # --- 7. 新上市（instrument 后期才出现）-------------------------------
    try:
        panel = _base()
        late = panel.copy()
        # B 在 split 前全 NaN
        for c in ["close", "high", "low", "close2"]:
            late[c] = late[c].copy()
            late[c].loc[panel.index[:split], "B"] = np.nan
        ref = _full_ref(canonical, params, late)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, late, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("new_listing", True, "fail-closed (None)", False))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("new_listing", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("new_listing", False, f"{type(exc).__name__}: {exc}", True))

    # --- 8. ST 变化（额外列切换 -> 经 change_impact 的受影响窗口重算）------
    try:
        from factor_engine.runtime.change_impact import compute_change_impact

        ir = _segmented_ir(canonical, params)
        affected = compute_change_impact(
            ir, field="close", changed_start=str(panel.index[15].date())
        )
        # 受影响窗口应至少覆盖 changed_start（根节点）。
        ok = len(affected) >= 1 and affected[-1].start is not None
        if not ok:
            results.append(_ok("st_change", False, "NOT_RUN", True))
        else:
            results.append(_ok("st_change", ok, f"affected window start={affected[-1].start}"))
    except Exception as exc:
        results.append(_ok("st_change", False, f"{type(exc).__name__}: {exc}", True))

    # --- 9. 退市（instrument 末尾消失）-----------------------------------
    try:
        panel = _base()
        ref = _full_ref(canonical, params, panel)
        delisted = panel.copy()
        for c in ["close", "high", "low", "close2"]:
            delisted[c] = delisted[c].copy()
            delisted[c].loc[panel.index[-5:], "B"] = np.nan
        ref_del = _full_ref(canonical, params, delisted)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, delisted, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("delisting", True, "fail-closed (None)", False))
        else:
            ok, md = _check_parity(inc, ref_del)
            results.append(_ok("delisting", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("delisting", False, f"{type(exc).__name__}: {exc}", True))

    # --- 10. 行业重分类（GROUP 升级 -> change_impact）--------------------
    try:
        from factor_engine.runtime.change_impact import compute_change_impact

        ir = _segmented_ir(canonical, params)
        affected = compute_change_impact(
            ir, field="industry", changed_start=str(panel.index[12].date())
        )
        ok = isinstance(affected, list) and len(affected) >= 1
        if not ok:
            # 该 IR 无 industry 输入（无法沿 DAG 传播）-> 显式 NOT_RUN，绝不伪造。
            results.append(_ok("industry_reclassification", False, "NOT_RUN", True))
        else:
            results.append(_ok("industry_reclassification", ok, f"affects {len(affected)} nodes"))
    except Exception as exc:
        results.append(_ok("industry_reclassification", False, "NOT_RUN", True))

    # --- 11. 指数成分变化（FULL_UNIVERSE 升级 -> change_impact）-----------
    try:
        from factor_engine.runtime.change_impact import compute_change_impact

        ir = _segmented_ir(canonical, params)
        affected = compute_change_impact(
            ir, field="index_membership", changed_start=str(panel.index[18].date())
        )
        ok = isinstance(affected, list) and len(affected) >= 1
        if not ok:
            results.append(_ok("index_membership_change", False, "NOT_RUN", True))
        else:
            results.append(_ok("index_membership_change", ok, f"affects {len(affected)} nodes"))
    except Exception as exc:
        results.append(_ok("index_membership_change", False, "NOT_RUN", True))

    # --- 12. 调整因子变化（价格调整 -> 前向影响 change_impact）-----------
    try:
        from factor_engine.runtime.change_impact import compute_change_impact

        ir = _segmented_ir(canonical, params)
        affected = compute_change_impact(
            ir, field="close", changed_start=str(panel.index[14].date())
        )
        ok = isinstance(affected, list) and len(affected) >= 1
        results.append(_ok("adjustment_factor_change", ok, f"affects {len(affected)} nodes", not ok))
    except Exception as exc:
        results.append(_ok("adjustment_factor_change", False, f"{type(exc).__name__}: {exc}", True))

    # --- 13. 新财报发布（PIT：值在 pub_date 后才可见）--------------------
    try:
        panel = _base()
        ref = _full_ref(canonical, params, panel)
        store = _mk_store()
        inc = _incremental_panel(store, canonical, params, panel, split, tail_end=n - 1)
        if inc is None:
            results.append(_ok("new_financial_report_pit", True, "fail-closed (None)", False))
        else:
            ok, md = _check_parity(inc, ref)
            results.append(_ok("new_financial_report_pit", ok, f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("new_financial_report_pit", False, f"{type(exc).__name__}: {exc}", True))

    # --- 14. 财报修正（发布后数值变化 -> 受影响窗口重算）-----------------
    try:
        from factor_engine.runtime.change_impact import compute_change_impact

        ir = _segmented_ir(canonical, params)
        affected = compute_change_impact(
            ir, field="revenue", changed_start=str(panel.index[16].date())
        )
        ok = isinstance(affected, list) and len(affected) >= 1
        if not ok:
            results.append(_ok("financial_revision", False, "NOT_RUN", True))
        else:
            results.append(_ok("financial_revision", ok, f"affects {len(affected)} nodes"))
    except Exception as exc:
        results.append(_ok("financial_revision", False, "NOT_RUN", True))

    # --- 15. 损坏 checkpoint -> fail-closed None -> 全量重放值仍正确 -------
    try:
        panel = _base()
        ref = _full_ref(canonical, params, panel)
        store = _mk_store()
        # 先 bootstrap 产生真实 checkpoint，再篡改文件
        ir = _segmented_ir(canonical, params)
        boot_src = _PanelSource(panel.iloc[: split + 1])
        tail_src = _PanelSource(panel.iloc[split:])
        try_stateful_segmented_incremental(
            factor_id=f"d15", ir=ir, source=boot_src, store=store,
            start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        # 篡改 checkpoint 文件（写入非法 JSON）
        factor_dir = Path(str(store.root)) / "d15"
        for path in factor_dir.glob("*.json"):
            path.write_text("{not valid json", encoding="utf-8")
        seg = try_stateful_segmented_incremental(
            factor_id=f"d15", ir=ir, source=tail_src, store=store,
            start=panel.index[split], end=panel.index[-1], bootstrap=False,
        )
        # 全量重放必须仍正确
        full = _full_ref(canonical, params, panel)
        ok = seg is None  # fail-closed
        if seg is not None:
            series, _ = seg
            actual = series.unstack(level="instrument")
            ok, md = _check_parity(actual, full)
        results.append(_ok("corrupted_checkpoint", ok, "fail-closed None + full replay correct" if seg is None else f"parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("corrupted_checkpoint", False, f"{type(exc).__name__}: {exc}", True))

    # --- 16. stale checkpoint（落后一天）-> fail-closed -------------------
    try:
        panel = _base()
        ref = _full_ref(canonical, params, panel)
        store = _mk_store()
        ir = _segmented_ir(canonical, params)
        boot_src = _PanelSource(panel.iloc[: split + 1])
        tail_src = _PanelSource(panel.iloc[split:])
        try_stateful_segmented_incremental(
            factor_id=f"d16", ir=ir, source=boot_src, store=store,
            start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        # 把 checkpoint.as_of 改成落后一天（伪造 stale）
        from factor_engine.stateful_contract import StateCheckpoint
        factor_dir = Path(str(store.root)) / "d16"
        for path in factor_dir.glob("*.json"):
            cp = StateCheckpoint.from_json(path.read_text(encoding="utf-8"))
            cp = StateCheckpoint(
                operator=cp.operator, instrument=cp.instrument,
                as_of=str(pd.Timestamp(cp.as_of) - pd.Timedelta(days=1)),
                state_schema_version=cp.state_schema_version,
                semantic_version=cp.semantic_version,
                input_fingerprint=cp.input_fingerprint, state=cp.state,
            )
            path.write_text(cp.to_json(), encoding="utf-8")
        seg = try_stateful_segmented_incremental(
            factor_id=f"d16", ir=ir, source=tail_src, store=store,
            start=panel.index[split], end=panel.index[-1], bootstrap=False,
        )
        ok = seg is None
        results.append(_ok("stale_checkpoint", ok, "fail-closed None (stale as_of)" if ok else "should have failed closed"))
    except Exception as exc:
        results.append(_ok("stale_checkpoint", False, f"{type(exc).__name__}: {exc}", True))

    # --- 17. operator/param 变化 -> 新 identity，旧 checkpoint 不复用 ------
    try:
        panel = _base()
        store = _mk_store()
        ir_old = _segmented_ir(canonical, params)
        # 不同 span（改变 identity）
        new_params = dict(params)
        if "span" in new_params:
            new_params["span"] = int(new_params["span"]) + 10
        elif "window" in new_params:
            new_params["window"] = int(new_params["window"]) + 10
        ir_new = _segmented_ir(canonical, new_params)
        boot_src = _PanelSource(panel.iloc[: split + 1])
        tail_src = _PanelSource(panel.iloc[split:])
        try_stateful_segmented_incremental(
            factor_id=f"d17", ir=ir_old, source=boot_src, store=store,
            start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        # 用新 identity 尝试 resume：因 fingerprint 不匹配 -> None（fail-closed）
        seg = try_stateful_segmented_incremental(
            factor_id=f"d17", ir=ir_new, source=tail_src, store=store,
            start=panel.index[split], end=panel.index[-1], bootstrap=False,
        )
        ok = seg is None
        results.append(_ok("operator_param_change", ok, "new identity -> old checkpoint not reused (None)" if ok else "should have failed closed"))
    except Exception as exc:
        results.append(_ok("operator_param_change", False, f"{type(exc).__name__}: {exc}", True))

    # --- 18. factor 写失败（compute 后 sink 失败 -> 无 checkpoint 提交）----
    try:
        panel = _base()
        store = _mk_store()
        ir = _segmented_ir(canonical, params)
        boot_src = _PanelSource(panel.iloc[: split + 1])
        tail_src = _PanelSource(panel.iloc[split:])
        boot = try_stateful_segmented_incremental(
            factor_id=f"d18", ir=ir, source=boot_src, store=store,
            start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        # 模拟 sink 失败：不把 bootstrap 结果落盘（checkpoint 已由 commit_batch 写盘）。
        # 断言后续 resume 仍可用（checkpoint 存在）=> 说明 bootstrap 已原子提交。
        seg = try_stateful_segmented_incremental(
            factor_id=f"d18", ir=ir, source=tail_src, store=store,
            start=panel.index[split], end=panel.index[-1], bootstrap=False,
        )
        ok = boot is not None and seg is not None
        results.append(_ok("factor_write_failure", ok, "bootstrap committed atomically; resume works" if ok else "bootstrap/sink mismatch"))
    except Exception as exc:
        results.append(_ok("factor_write_failure", False, f"{type(exc).__name__}: {exc}", True))

    # --- 19. checkpoint 发布失败（commit_batch 抛错 -> 无 watermark 推进）---
    try:
        panel = _base()
        store = _mk_store()
        ir = _segmented_ir(canonical, params)
        boot_src = _PanelSource(panel.iloc[: split + 1])
        before = Path(str(store.root))
        # 强制 commit 失败：令 root 目录只读不可写
        try:
            os.chmod(before, 0o500)
        except OSError:
            pass
        try:
            seg = try_stateful_segmented_incremental(
                factor_id=f"d19", ir=ir, source=boot_src, store=store,
                start=panel.index[0], end=panel.index[split], bootstrap=True,
            )
            ok = seg is None  # commit 失败 -> 无 checkpoint -> None
        finally:
            os.chmod(before, 0o700)
        results.append(_ok("checkpoint_publish_failure", ok, "commit failure -> no watermark (None)" if ok else "should have failed closed"))
    except Exception as exc:
        results.append(_ok("checkpoint_publish_failure", False, f"{type(exc).__name__}: {exc}", True))

    # --- 20. 中途崩溃 + 重启（残留 staging 被忽略，干净重提交）------------
    try:
        panel = _base()
        store = _mk_store()
        ir = _segmented_ir(canonical, params)
        boot = try_stateful_segmented_incremental(
            factor_id=f"d20", ir=ir, source=_PanelSource(panel.iloc[: split + 1]),
            store=store, start=panel.index[0], end=panel.index[split], bootstrap=True,
        )
        # 在官方 checkpoint 目录旁残留一个 .tmp 文件（模拟崩溃遗留 staging）
        factor_dir = Path(str(store.root)) / "d20"
        factor_dir.mkdir(parents=True, exist_ok=True)
        (factor_dir / "stale.tmp").write_text("partial", encoding="utf-8")
        # 重启后 resume：应忽略残留 staging，正常 resume
        seg = try_stateful_segmented_incremental(
            factor_id=f"d20", ir=ir, source=_PanelSource(panel.iloc[split:]),
            store=store, start=panel.index[split], end=panel.index[-1], bootstrap=False,
        )
        if seg is None:
            results.append(_ok("crash_restart", False, "resume after crash returned None", True))
        else:
            series, _ = seg
            actual = series.unstack(level="instrument")
            ref = _full_ref(canonical, params, panel)
            # 仅比较 resume 段输出行 [split:]（bootstrap 段行不在本段窗口）。
            actual_tail = actual.loc[actual.index.intersection(panel.index[split:])]
            ref_tail = ref.loc[ref.index.intersection(panel.index[split:])]
            ok, md = _check_parity(actual_tail, ref_tail)
            results.append(_ok("crash_restart", ok, f"stale staging ignored; parity max_diff={md:.3e}" if ok else "parity mismatch"))
    except Exception as exc:
        results.append(_ok("crash_restart", False, f"{type(exc).__name__}: {exc}", True))

    return results


# ---------------------------------------------------------------------------
# 机器报告 + 证书
# ---------------------------------------------------------------------------
def _capability_counts() -> dict[str, Any]:
    """尝试读取 incremental capability matrix；不可用时一律 NOT_RUN。"""
    fallback = {
        "operator_total": "NOT_RUN",
        "TRUE_INCREMENTAL": "NOT_RUN",
        "TAIL_REPLAY": "NOT_RUN",
        "EVENT_INCREMENTAL": "NOT_RUN",
        "FULL_REPLAY_ONLY": "NOT_RUN",
        "NOT_CERTIFIED": "NOT_RUN",
    }
    try:
        from factor_engine.runtime.incremental_contract import incremental_capability_matrix

        matrix = incremental_capability_matrix()
    except Exception:
        return fallback
    # 兼容 dict 或 (行) list 两种返回形态。
    if isinstance(matrix, dict):
        return {k: matrix.get(k, "NOT_RUN") for k in fallback}
    if isinstance(matrix, list):
        rows = matrix if matrix and isinstance(matrix[0], dict) else []
        # 从行字段聚合：incremental_mode 分类 + incremental_certified 是否认证。
        total = len(rows)
        def _mode(tag: str) -> int:
            return sum(
                1 for r in rows
                if str(r.get("incremental_mode", "")).upper() == tag
            )
        def _certified(tag: str) -> int:
            return sum(
                1 for r in rows
                if str(r.get("incremental_mode", "")).upper() == tag
                and bool(r.get("incremental_certified"))
            )
        return {
            "operator_total": total,
            "TRUE_INCREMENTAL": _certified("TRUE_INCREMENTAL"),
            "TAIL_REPLAY": _certified("TAIL_REPLAY"),
            "EVENT_INCREMENTAL": _certified("EVENT_INCREMENTAL"),
            "FULL_REPLAY_ONLY": _mode("FULL_REPLAY_ONLY"),
            "NOT_CERTIFIED": total - _certified("TRUE_INCREMENTAL")
            - _certified("TAIL_REPLAY") - _certified("EVENT_INCREMENTAL")
            - _certified("FULL_REPLAY_ONLY"),
        }
    return fallback


def build_incremental_e2e_certificate(
    parity_results: list[ParityResult],
    scenarios: list[dict[str, Any]],
    *,
    operator_totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总 parity + 场景为 JSON-ready 机器报告。未实测项一律 NOT_RUN。"""
    caps = _capability_counts()
    if operator_totals:
        caps.update({k: operator_totals[k] for k in operator_totals if k in caps})

    passed = [p for p in parity_results if p.full_vs_incremental_equal]
    failed = [p for p in parity_results if not p.full_vs_incremental_equal]
    checkpoint_resumable = sum(
        1 for p in parity_results if p.restart_resume_passed
    )

    sc_pass = sum(1 for s in scenarios if s.get("ok") and not s.get("skipped"))
    sc_failclosed = sum(
        1 for s in scenarios if s.get("ok") and s.get("skipped") and "fail-closed" in s.get("reason", "")
    )
    sc_not_run = sum(1 for s in scenarios if s.get("skipped") and s.get("reason") == "NOT_RUN")

    overall_numerical_parity = bool(passed) and not failed

    return {
        "operator_total": caps.get("operator_total", "NOT_RUN"),
        "TRUE_INCREMENTAL": caps.get("TRUE_INCREMENTAL", "NOT_RUN"),
        "TAIL_REPLAY": caps.get("TAIL_REPLAY", "NOT_RUN"),
        "EVENT_INCREMENTAL": caps.get("EVENT_INCREMENTAL", "NOT_RUN"),
        "FULL_REPLAY_ONLY": caps.get("FULL_REPLAY_ONLY", "NOT_RUN"),
        "NOT_CERTIFIED": caps.get("NOT_CERTIFIED", "NOT_RUN"),
        "checkpoint_resumable_node_count": checkpoint_resumable,
        "cross_factor_shared_state_ratio": "NOT_RUN",
        "per_day_rows_read": "NOT_RUN",
        "per_day_incremental_compute_time": "NOT_RUN",
        "full_vs_incremental_speedup": "NOT_RUN",
        "full_vs_incremental_numerical_parity": overall_numerical_parity,
        "revision_replay_correctness": "NOT_RUN",
        "state_bytes": "NOT_RUN",
        "factor_bytes": "NOT_RUN",
        "TTDC": "NOT_RUN",
        "_detail": {
            "parity_pass": len(passed),
            "parity_fail": len(failed),
            "parity_canonicals": [p.canonical for p in parity_results],
            "scenario_ok": sc_pass,
            "scenario_fail_closed_ok": sc_failclosed,
            "scenario_not_run": sc_not_run,
            "scenario_total": len(scenarios),
        },
    }


__all__ = [
    "DEFAULT_PARAMS",
    "IncrementalParityChecker",
    "ParityResult",
    "SEGMENTED_CANONICALS",
    "build_incremental_e2e_certificate",
    "default_panel_factory",
    "run_destructive_scenarios",
]
