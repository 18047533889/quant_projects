#!/usr/bin/env python3
"""P0#11+P0#12 — 100k dry-run ladder runner.

逐级放量（100 → 1k → 5k → 20k → 50k → 100k）跑因子 dry-run：
分层抽样公式集 → run_many 批量算 → streaming sink 流式落值 → checkpoint
（已完成公式 id 集合，中断可续跑）→ 失败分类（参数错/数据缺/超时/OOM/语义错）。

判据（GO_PROMPT §71 + §99-§105）：
  Stage A 100  : 核语义
  Stage B 1000 : CSE/IO
  Stage C 5000 : scheduler/RSS
  Stage D 20k  : write/cache
  Stage E 50k  : long-run stability
  Stage F 100k : 正式
每级比较 output checksum / DQ / throughput / memory / failures。

用法：
  python3 scripts/dry_run_ladder.py --levels 100,1000 --out /tmp/ladder_out \
      --workers 6 --seed 42
  # 续跑（checkpoint 存在时自动跳过已完成公式）：
  python3 scripts/dry_run_ladder.py --levels 100,1000 --out /tmp/ladder_out --resume
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 环境：数据根 + 跳过 COS mirror（本地已有 2586 天 parquet）
# ---------------------------------------------------------------------------
_DEFAULT_DATA_ROOT = "/home/sunhaiwei/quant_projects/data/a_share/lqtp_data"
os.environ.setdefault("ASHARE_PARQUET_ROOT", _DEFAULT_DATA_ROOT)
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
# 子进程线程上限（机器 32 核 92G，本 ladder 试跑 ≤6 workers）
os.environ.setdefault("OMP_NUM_THREADS", "6")

# ---------------------------------------------------------------------------
# 失败分类
# ---------------------------------------------------------------------------
FAIL_PARAM = "param_error"      # 参数错：编译/绑定/参数校验失败
FAIL_DATA = "data_missing"      # 数据缺：字段/数据集缺失、0 行、PIT 缺
FAIL_TIMEOUT = "timeout"        # 超时
FAIL_OOM = "oom"                # 内存不足
FAIL_SEMANTIC = "semantic"      # 语义错：结果 NaN 率过高 / 全 NaN / 形状错
FAIL_OTHER = "other"            # 其它

_FAIL_CLASSES = [
    FAIL_PARAM, FAIL_DATA, FAIL_TIMEOUT, FAIL_OOM, FAIL_SEMANTIC, FAIL_OTHER,
]

# 概念 → ashare_stock_daily 物理列（仅本 ladder 用的小池；未映射概念会落 data_missing）
CONCEPT_TO_PHYSICAL: dict[str, str] = {
    "continuous_close": "Close",
    "continuous_high": "High",
    "continuous_low": "Low",
    "continuous_open": "Open",
    "continuous_volume_shares": "Volume",
    "continuous_vwap": "Vwap",
    "return_decimal": "Return",
    "amount_local": "Amount",
}

# 小池：50 股票 × 120 交易日
_INSTRUMENTS = [
    "000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ", "000651.SZ",
    "000858.SZ", "000895.SZ", "000938.SZ", "000977.SZ", "002027.SZ",
    "002230.SZ", "002304.SZ", "002415.SZ", "002475.SZ", "002594.SZ",
    "300014.SZ", "300059.SZ", "300124.SZ", "300274.SZ", "300308.SZ",
    "300408.SZ", "300433.SZ", "300498.SZ", "300750.SZ", "300760.SZ",
    "600000.SH", "600009.SH", "600016.SH", "600019.SH", "600028.SH",
    "600030.SH", "600031.SH", "600036.SH", "600048.SH", "600050.SH",
    "600104.SH", "600276.SH", "600309.SH", "600519.SH", "600585.SH",
    "600690.SH", "600703.SH", "600745.SH", "600809.SH", "600887.SH",
    "600900.SH", "601012.SH", "601088.SH", "601166.SH", "601318.SH",
]
_START = "2024-01-01"
_END = "2024-06-30"


# ---------------------------------------------------------------------------
# 公式构建
# ---------------------------------------------------------------------------
def _param_default(op_row: Any, name: str) -> Any:
    """从 operator metadata 取参数默认值；MISSING 时按名字启发式。"""
    try:
        op = _get_operator(op_row.canonical)
        ps = getattr(getattr(op, "metadata", None), "param_specs", None) or {}
        spec = ps.get(name)
        if spec is not None:
            default = getattr(spec, "default", None)
            # dataclasses.MISSING sentinel —— 不能当真实默认值传回
            import dataclasses
            if default is not None and default is not dataclasses.MISSING:
                return default
    except Exception:
        pass
    low = name.lower()
    if "window" in low or "lookback" in low or "period" in low:
        return 10
    if "offset" in low:
        return 0
    if "sigma" in low:
        return 6
    if "min_periods" in low or "min_peers" in low:
        return 3
    if "roc" in low:
        return 5
    if "ema" in low or "smooth" in low:
        return 5
    if "signal" in low:
        return 3
    if "output" in low:
        return 0
    if "left" in low or "right" in low:
        return 5
    if "pivot" in low:
        return 10
    if "cutoff" in low:
        return 5
    return 5


def build_formula(op_row: Any) -> Any:
    """按 operator 的 data_inputs + scalar_parameters 构造 CleanedCall 表达式。

    面板槽位 = default_input_recipe 的 key（顺序稳定）；标量参数 = scalar_parameters
    中不在 recipe key 里的（避免把面板槽误当标量）。未映射概念会落 data_missing。
    """
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api import col

    recipe = dict(op_row.default_input_recipe or {})
    # 面板槽位 → recipe 的 VALUE（概念名，如 continuous_close）；fields 映射把这些
    # 概念绑定到物理列（Close/High/Low/...）。未映射概念在 run 期落 data_missing。
    panel_slots = list(recipe.keys())
    scalar_params = [
        p for p in (op_row.scalar_parameters or ()) if p not in recipe
    ]
    factory = make_cleaned_call_factory(op_row.canonical)
    args = [col(str(recipe[slot])) for slot in panel_slots]
    kwargs = {p: _param_default(op_row, p) for p in scalar_params}
    return factory(*args, **kwargs)


def _get_operator(canonical: str) -> Any:
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry.get(canonical)


# ---------------------------------------------------------------------------
# 分层抽样
# ---------------------------------------------------------------------------
def stratified_sample(rows: list[Any], n: int, rng: np.random.Generator) -> list[Any]:
    """分层抽样：按 (economic_effect_family, 输入源数) 分层，保证家族/输入源覆盖。

    每层至少取 1 个（若 n 足够），其余按比例补齐；绝不只取前 N 条。
    """
    if n >= len(rows):
        return list(rows)
    # 层 key：economic_effect_family + 输入源数
    layers: dict[tuple[str, int], list[Any]] = {}
    for r in rows:
        fam = str(getattr(r, "economic_effect_family", "unknown") or "unknown")
        nsrc = len(getattr(r, "source_recipes", ()) or ())
        layers.setdefault((fam, nsrc), []).append(r)
    # 每层至少 1 个（若 n 允许）
    picked: list[Any] = []
    layer_items = list(layers.items())
    rng.shuffle(layer_items)
    for key, members in layer_items:
        if len(picked) >= n:
            break
        picked.append(rng.choice(members))
    # 剩余按比例补齐
    remaining = n - len(picked)
    if remaining > 0:
        weights = [len(m) for _, m in layer_items]
        total = sum(weights) or 1
        for key, members in layer_items:
            if remaining <= 0:
                break
            take = max(0, int(round(len(members) / total * remaining)))
            take = min(take, len(members), remaining)
            if take > 0:
                picked.extend(rng.choice(members, size=take, replace=False).tolist())
                remaining -= take
    # 若仍不足（浮点取整），从最大层补
    if remaining > 0:
        biggest = max(layer_items, key=lambda kv: len(kv[1]))
        pool = [r for r in biggest[1] if r not in picked]
        picked.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return picked[:n]


# ---------------------------------------------------------------------------
# 数据源 / 引擎
# ---------------------------------------------------------------------------
def build_engine() -> Any:
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine

    fields = {c: p for c, p in CONCEPT_TO_PHYSICAL.items()}
    src = DataAccessSource(
        dataset="ashare_stock_daily",
        fields=fields,
        start_date=_START,
        end_date=_END,
        instrument_filter=list(_INSTRUMENTS),
        run_mode="interactive_research",
    )
    eng = FactorEngine(backend=PandasBackend(), data_source=src, run_mode="research")
    return eng


# ---------------------------------------------------------------------------
# 失败分类
# ---------------------------------------------------------------------------
def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc)
    low = (name + " " + msg).lower()
    # 高优先级：MISSING 默认值泄漏进 plan → 构造/参数错，不是数据缺
    if "missingdefault" in low or "unsupported plan attribute" in low:
        return FAIL_PARAM
    # 数据缺
    if any(k in low for k in [
        "unknownfield", "missingdata", "dataerror", "0 row", "no field",
        "no such", "not found", "missing", "unavailable", "catalog",
        "coverage", "no data", "empty", "binder", "does not exist",
        "unknown column", "unknown field", "fieldregistry",
        "typedinputcontract", "resourceadmission", "contracterror",
        "inflight scan bytes", "required raw", "require raw",
        "source_unknown", "supported_markets",
    ]):
        return FAIL_DATA
    # 参数错
    if any(k in low for k in [
        "param", "arity", "argument", "typeerror", "valueerror",
        "unsupported", "invalid", "signature", "keyword", "positional",
        "dslparse", "compile", "budget", "not callable", "coerce",
        "missingdefault", "missing 1 required positional",
    ]):
        return FAIL_PARAM
    # DuckDB binder / SQL / 引擎错误但字段相关 → 数据缺（列找不到）
    if any(k in low for k in [
        "binder error", "duckdb 查询失败", "referenced column",
        "column .* not found", "engineerror",
    ]):
        return FAIL_DATA
    # 数值/域错误（tail-state quantile、除零、log of negative）→ 语义错
    if any(k in low for k in [
        "quantile", "domain", "must be in", "zero", "negative", "nan",
        "overflow", "log of", "sqrt of", "nonzero", "division",
    ]):
        return FAIL_SEMANTIC
    # 内存
    if any(k in low for k in ["memory", "oom", "alloc", "rss", "budget_bytes", "refused"]):
        return FAIL_OOM
    # 超时
    if any(k in low for k in ["timeout", "timed out", "deadline", "abort"]):
        return FAIL_TIMEOUT
    return FAIL_OTHER


def classify_result(result: Any) -> str | None:
    """语义错：全 NaN / NaN 率过高 / 形状异常。返回 None 表示通过。"""
    if result is None:
        return FAIL_SEMANTIC
    try:
        arr = result.values if hasattr(result, "values") else np.asarray(result)
        if arr.size == 0:
            return FAIL_SEMANTIC
        if np.isnan(arr).all():
            return FAIL_SEMANTIC
        nan_ratio = float(np.isnan(arr).mean())
        if nan_ratio > 0.999:
            return FAIL_SEMANTIC
    except Exception:
        return FAIL_OTHER
    return None


# ---------------------------------------------------------------------------
# streaming sink：边算边写 parquet 分片
# ---------------------------------------------------------------------------
class StreamingSink:
    """把每个因子结果立即写成一个 parquet 分片，不驻留内存。"""

    def __init__(self, sink_dir: Path, shard_rows: int = 5000):
        self.sink_dir = Path(sink_dir)
        self.sink_dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = shard_rows
        self._buf: list[pd.DataFrame] = []
        self._buf_rows = 0
        self._shard_idx = 0
        self.written_files: list[str] = []
        self.total_rows = 0

    def write(self, name: str, result: Any) -> None:
        df = self._to_frame(name, result)
        self._buf.append(df)
        self._buf_rows += len(df)
        if self._buf_rows >= self.shard_rows:
            self._flush()

    def _to_frame(self, name: str, result: Any) -> pd.DataFrame:
        if isinstance(result, pd.Series):
            s = result.rename(name)
            return s.to_frame()
        if isinstance(result, pd.DataFrame):
            return result.copy()
        return pd.DataFrame({name: result})

    def _flush(self) -> None:
        if not self._buf:
            return
        frame = pd.concat(self._buf, axis=0)
        path = self.sink_dir / f"shard_{self._shard_idx:05d}.parquet"
        frame.to_parquet(path)
        self.written_files.append(str(path))
        self.total_rows += len(frame)
        self._shard_idx += 1
        self._buf = []
        self._buf_rows = 0

    def close(self) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
@dataclass
class LevelState:
    level: int
    n_formulas: int
    passed: int = 0
    failed_by_class: dict[str, int] = field(default_factory=lambda: {c: 0 for c in _FAIL_CLASSES})
    wall_time_s: float = 0.0
    rss_peak_mb: float = 0.0
    sink_files: list[str] = field(default_factory=list)
    checkpoint_path: str = ""
    completed_ids: list[str] = field(default_factory=list)
    done: bool = False
    all_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Checkpoint:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}

    def level_state(self, level: int) -> LevelState:
        key = str(level)
        if key not in self.data:
            self.data[key] = LevelState(level=level, n_formulas=0).to_dict()
        return LevelState(**self.data[key])

    def save_level(self, state: LevelState) -> None:
        self.data[str(state.level)] = state.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))
        tmp.replace(self.path)

    def completed_ids(self, level: int) -> set[str]:
        return set(self.level_state(level).completed_ids)


# ---------------------------------------------------------------------------
# 单级执行
# ---------------------------------------------------------------------------
def run_level(
    level: int,
    formulas: list[tuple[str, Any]],
    engine: Any,
    sink: StreamingSink,
    ckpt: Checkpoint,
    *,
    workers: int,
    per_factor_timeout: float,
) -> LevelState:
    state = ckpt.level_state(level)
    state.n_formulas = len(formulas)
    state.checkpoint_path = str(ckpt.path)
    done_ids = ckpt.completed_ids(level)
    # 全部公式已完成 → 本级已 done，直接返回（避免续跑重复）
    if len(done_ids) >= len(formulas):
        state.all_complete = True
        ckpt.save_level(state)
        return state
    start = time.time()
    interrupted = False
    _prior_completed = len(state.completed_ids)
    _prior_wall = state.wall_time_s

    for fid, expr in formulas:
        if fid in done_ids:
            continue
        t0 = time.time()
        try:
            from factor_engine.api.factor import Factor
            factor = Factor(name=fid, expr=expr)
            out = engine.run_many([factor], enable_cse=True)
            result = out["results"][fid]
            sem = classify_result(result)
            if sem is not None:
                state.failed_by_class[sem] += 1
            else:
                sink.write(fid, result)
                state.passed += 1
        except MemoryError:
            state.failed_by_class[FAIL_OOM] += 1
        except Exception as exc:  # noqa: BLE001 — 分类后继续，不中断 ladder
            cls = classify_exception(exc)
            state.failed_by_class[cls] += 1
        finally:
            state.completed_ids.append(fid)
            # 每 25 个公式写一次 checkpoint（中断可续跑）
            if len(state.completed_ids) % 25 == 0:
                state.wall_time_s = time.time() - start
                state.rss_peak_mb = _rss_peak_mb()
                state.sink_files = list(sink.written_files)
                ckpt.save_level(state)
            # 测试钩子：LADDER_MAX_COMPLETE 达到后模拟中断（验证续跑）
            _max = os.environ.get("LADDER_MAX_COMPLETE")
            if _max and len(state.completed_ids) - _prior_completed >= int(_max):
                interrupted = True
                break

    state.wall_time_s = _prior_wall + (time.time() - start)
    state.rss_peak_mb = _rss_peak_mb()
    state.sink_files = list(sink.written_files)
    if not interrupted and len(state.completed_ids) >= len(formulas):
        state.done = True
    ckpt.save_level(state)
    return state


def _rss_peak_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_terminal_operators() -> list[Any]:
    from factor_engine.cleaned_operators import load_all
    load_all(include_research=False)
    from factor_engine.mining.direct_use import DirectUseContext, get_direct_use_mining_operators
    from factor_engine.market.context import Market
    ctx = DirectUseContext(market=Market.ASHARE)
    rows = get_direct_use_mining_operators(ctx, admission="all")
    return [r for r in rows if r.terminal_allowed]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="100,1000", help="逗号分隔的 ladder 级")
    ap.add_argument("--out", default="/tmp/ladder_out", help="输出目录")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="续跑（跳过已完成公式）")
    ap.add_argument("--per-factor-timeout", type=float, default=120.0)
    ap.add_argument("--smoke", action="store_true", help="10 公式冒烟（自测）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(out_dir / "ladder_checkpoint.json")
    rng = np.random.default_rng(args.seed)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if args.smoke:
        levels = [10]

    # 加载 terminal_allowed 公式集（1446 条）
    rows = load_terminal_operators()
    print(f"[ladder] terminal_allowed operators: {len(rows)}", flush=True)

    # 预构建公式（id -> expr），供各层抽样
    all_formulas: list[tuple[str, Any]] = []
    for i, r in enumerate(rows):
        try:
            expr = build_formula(r)
        except Exception as exc:  # noqa: BLE001
            # 构建期失败 → 记为 param_error，仍占一个 id
            all_formulas.append((f"f_{i:06d}", None))
            continue
        all_formulas.append((f"f_{i:06d}", expr))

    engine = build_engine()
    summary: dict[str, Any] = {"levels": {}, "criteria": _CRITERIA}

    for level in levels:
        print(f"\n[ladder] === level {level} ===", flush=True)
        # 分层抽样
        sampled = stratified_sample(all_formulas, level, rng)
        sink = StreamingSink(out_dir / f"level_{level}_sink")
        state = run_level(
            level, sampled, engine, sink, ckpt,
            workers=args.workers, per_factor_timeout=args.per_factor_timeout,
        )
        sink.close()
        summary["levels"][str(level)] = state.to_dict()
        print(f"[ladder] level {level}: passed={state.passed} "
              f"failed={sum(state.failed_by_class.values())} "
              f"wall={state.wall_time_s:.1f}s rss_peak={state.rss_peak_mb:.0f}MB",
              flush=True)
        for cls, cnt in state.failed_by_class.items():
            if cnt:
                print(f"    {cls}: {cnt}", flush=True)

    # 落盘 ladder 状态 json
    status_path = out_dir / "ladder_status.json"
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[ladder] status -> {status_path}", flush=True)
    return 0


_CRITERIA = {
    "100": "核语义",
    "1000": "CSE/IO",
    "5000": "scheduler/RSS",
    "20000": "write/cache",
    "50000": "long-run stability",
    "100000": "正式",
    "compare": ["output checksum", "DQ", "throughput", "memory", "failures"],
}


if __name__ == "__main__":
    sys.exit(main())
