"""data_access.r30.data_quality —— DataQualityService 立面（R30-P1-025）。

包装既有 quality/contracts.py 的 ``run_quality_checks``（11 种 Arrow 层检查），
输出统一的 ``DataQualityResult``（dataset / source_snapshot / checks / severity /
problems），并内建一组最小检查集作为降级路径：

    - row-count drift（对比 manifest 期望行数 vs 实际）
    - missing partition（时间范围空洞，启发式）
    - OHLC consistency（high>=low、close<=high、open 在 [low, high] 内）
    - negative volume
    - financial period <= publish_time（PIT leakage）
    - currency enum（已知币种枚举）
    - schema/semantic epoch（manifest schema_epoch 存在时校验表 schema）

severity 聚合：PASS（全过）/ WARN（有 warn 无 block）/ BLOCK（有 block）。

**永不写数据**：本模块只读，绝不改表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from data_access.r30._shared import stable_digest_full

__all__ = [
    "BLOCK",
    "PASS",
    "WARN",
    "DataQualityResult",
    "DataQualityService",
    "aggregate_severity",
    "qualify_source",
]

PASS = "PASS"
WARN = "WARN"
BLOCK = "BLOCK"

_SEVERITY_RANK = {PASS: 0, WARN: 1, BLOCK: 2}

_KNOWN_CURRENCIES: frozenset[str] = frozenset({
    "CNY", "CNH", "USD", "HKD", "JPY", "EUR", "GBP", "AUD", "CAD",
    "SGD", "NZD", "KRW", "INR", "CHF", "TWD", "MYR", "THB",
})

_ROW_COUNT_DRIFT_WARN_RATIO = 0.20  # 行数相对漂移 >20% → WARN

_CURRENCY_COLUMNS = ("currency", "ccy", "currency_code")

_TIME_COLUMN_CANDIDATES = (
    "date", "time", "ts", "timestamp", "datetime", "trade_date", "report_date",
)


@dataclass
class DataQualityResult:
    """一次数据质量检查的结果。"""

    dataset: str
    source_snapshot: Any = None
    checks: dict[str, str] = field(default_factory=dict)   # check 名 -> PASS/WARN/BLOCK
    severity: str = PASS
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_snapshot": _snapshot_to_dict(self.source_snapshot),
            "checks": dict(self.checks),
            "severity": self.severity,
            "problems": list(self.problems),
        }

    def ok(self) -> bool:
        """是否可用：severity != BLOCK。"""
        return self.severity != BLOCK


def aggregate_severity(checks: Mapping[str, str]) -> str:
    """PASS（全过）/ WARN（有 warn 无 block）/ BLOCK（有 block）。"""
    if any(v == BLOCK for v in checks.values()):
        return BLOCK
    if any(v == WARN for v in checks.values()):
        return WARN
    return PASS


class DataQualityService:
    """数据质量立面：读数据 → 调既有 run_quality_checks + 内置最小检查集。"""

    def run(
        self,
        store: Any,
        dataset: str,
        params: Mapping[str, Any] | None = None,
        options: Any = None,
    ) -> DataQualityResult:
        """对 dataset 跑数据质量检查，返回 DataQualityResult。

        - 先调既有 quality/contracts.run_quality_checks（防御性 import；不可用/
          抛异常则跳过，不影响内置检查）；
        - 内置最小检查集**总是**运行（即降级路径）；
        - 永不写数据。
        """
        params = dict(params or {})
        table, snapshot = self._read(store, dataset, params)

        checks: dict[str, str] = {}
        problems: list[str] = []

        if table is None:
            checks["read"] = BLOCK
            problems.append(f"无法读取数据集 {dataset} 的数据")
        else:
            # 1) 既有 quality/contracts.run_quality_checks（包装）
            external = self._run_external_checks(dataset, table, options)
            if external is not None:
                ext_checks, ext_problems = external
                checks.update(ext_checks)
                problems.extend(ext_problems)

            # 2) 内置最小检查集
            expected_rows, schema_epoch = self._manifest_info(store, dataset, params)
            builtin = self._builtin_checks(
                dataset, table, expected_rows=expected_rows, schema_epoch=schema_epoch
            )
            builtin_checks, builtin_problems = builtin
            checks.update(builtin_checks)
            problems.extend(builtin_problems)

        severity = aggregate_severity(checks)
        return DataQualityResult(
            dataset=dataset,
            source_snapshot=snapshot,
            checks=checks,
            severity=severity,
            problems=problems,
        )

    def qualify_source(self, store: Any, dataset: str) -> str:
        """便捷返回 PASS/WARN/BLOCK。"""
        return self.run(store, dataset).severity

    # ---- 外部质量检查包装 ----

    def _run_external_checks(
        self, dataset: str, table: Any, options: Any
    ) -> tuple[dict[str, str], list[str]] | None:
        """调 quality/contracts.run_quality_checks；不可用/抛异常时 fail closed。"""
        try:
            from data_access.quality.contracts import QualityOptions, run_quality_checks
        except Exception as exc:
            return (
                {"quality_contracts": BLOCK},
                [f"quality_contracts checker unavailable: {exc}"],
            )
        opts = options if isinstance(options, QualityOptions) else None
        try:
            report = run_quality_checks(dataset, table, options=opts)
        except Exception as exc:
            return (
                {"quality_contracts": BLOCK},
                [f"quality_contracts check failed: {exc}"],
            )
        failures = list(getattr(report, "failures", ()) or ())
        checks = {
            "quality_contracts": BLOCK if failures else PASS,
        }
        return checks, failures

    # ---- 内置最小检查集 ----

    def _builtin_checks(
        self,
        dataset: str,
        table: Any,
        *,
        expected_rows: int | None,
        schema_epoch: Any,
    ) -> tuple[dict[str, str], list[str]]:
        checks: dict[str, str] = {}
        problems: list[str] = []

        # row-count drift
        if expected_rows is not None and table is not None:
            actual = int(getattr(table, "num_rows", 0) or 0)
            drift = _relative_drift(actual, expected_rows)
            if drift > _ROW_COUNT_DRIFT_WARN_RATIO:
                checks["row_count_drift"] = WARN
                problems.append(
                    f"row count drift: manifest~{expected_rows} actual={actual}"
                )
            else:
                checks["row_count_drift"] = PASS

        if table is not None:
            checks, problems = self._merge(
                checks, problems, self._check_ohlc(table)
            )
            checks, problems = self._merge(
                checks, problems, self._check_negative_volume(table)
            )
            checks, problems = self._merge(
                checks, problems, self._check_financial_period(table)
            )
            checks, problems = self._merge(
                checks, problems, self._check_currency(table)
            )
            checks, problems = self._merge(
                checks, problems, self._check_missing_partition(table)
            )
            checks, problems = self._merge(
                checks, problems, self._check_schema_epoch(table, schema_epoch)
            )
        return checks, problems

    @staticmethod
    def _merge(
        checks: dict[str, str],
        problems: list[str],
        result: tuple[str, str, list[str]],
    ) -> tuple[dict[str, str], list[str]]:
        name, severity, msgs = result
        if name is not None:
            checks[name] = severity
        problems.extend(msgs)
        return checks, problems

    def _check_ohlc(
        self, table: Any
    ) -> tuple[str, str, list[str]]:
        """OHLC consistency：high>=low、close<=high、open 在 [low, high]。

        R32 P0-068..072：异常 → BLOCK（fail-closed）。
        """
        cols = set(getattr(table, "column_names", []) or [])
        if not {"high", "low"}.issubset(cols):
            return "ohlc", PASS, []
        try:
            import pyarrow.compute as pc

            hi = table.column("high")
            lo = table.column("low")
            bad = int(pc.sum(pc.less(hi, lo)).as_py() or 0)
            msgs: list[str] = []
            if bad:
                msgs.append(f"OHLC inconsistency: high < low in {bad} rows")
            if "close" in cols:
                close_bad = int(
                    pc.sum(pc.greater(table.column("close"), hi)).as_py() or 0
                )
                if close_bad:
                    msgs.append(
                        f"OHLC inconsistency: close > high in {close_bad} rows"
                    )
            if "open" in cols:
                o = table.column("open")
                below = int(pc.sum(pc.less(o, lo)).as_py() or 0)
                above = int(pc.sum(pc.greater(o, hi)).as_py() or 0)
                if below or above:
                    msgs.append(
                        f"OHLC inconsistency: open outside [low, high] in "
                        f"{below + above} rows"
                    )
            return "ohlc", (BLOCK if msgs else PASS), msgs
        except Exception as exc:
            return "ohlc", BLOCK, [f"ohlc check failed: {exc}"]

    def _check_negative_volume(self, table: Any) -> tuple[str, str, list[str]]:
        """R32 P0-068..072：异常 → BLOCK（fail-closed）。"""
        cols = set(getattr(table, "column_names", []) or [])
        if "volume" not in cols:
            return "negative_volume", PASS, []
        try:
            import pyarrow as pa
            import pyarrow.compute as pc

            v = table.column("volume")
            if not (pa.types.is_floating(v.type) or pa.types.is_integer(v.type)):
                v = pc.cast(v, pa.float64())
            neg = int(pc.sum(pc.less(v, 0)).as_py() or 0)
            if neg:
                return "negative_volume", BLOCK, [f"negative volume: {neg} rows"]
            return "negative_volume", PASS, []
        except Exception as exc:
            return "negative_volume", BLOCK, [f"negative_volume check failed: {exc}"]

    def _check_financial_period(self, table: Any) -> tuple[str, str, list[str]]:
        """financial period <= publish_time（PIT leakage）。

        R32 P0-068..072：异常 → BLOCK（fail-closed）。
        """
        cols = set(getattr(table, "column_names", []) or [])
        if not {"report_period", "publish_time"}.issubset(cols):
            return "financial_period", PASS, []
        try:
            import pyarrow as pa
            import pyarrow.compute as pc

            rp = pc.cast(table.column("report_period"), pa.timestamp("us"))
            pt = pc.cast(table.column("publish_time"), pa.timestamp("us"))
            leak = int(
                pc.sum(pc.and_kleene(pc.is_valid(rp), pc.greater(rp, pt))).as_py()
                or 0
            )
            if leak:
                return (
                    "financial_period",
                    BLOCK,
                    [f"financial period > publish_time (PIT leakage): {leak} rows"],
                )
            return "financial_period", PASS, []
        except Exception as exc:
            return "financial_period", BLOCK, [f"financial_period check failed: {exc}"]

    def _check_currency(self, table: Any) -> tuple[str, str, list[str]]:
        """R32 P0-068..072：异常 → WARN（currency 非致命）。"""
        cols = set(getattr(table, "column_names", []) or [])
        for name in _CURRENCY_COLUMNS:
            if name in cols:
                try:
                    import pyarrow.compute as pc

                    values = set(pc.unique(table.column(name)).to_pylist())
                    unknown = sorted(
                        str(v).upper()
                        for v in values
                        if v is not None and str(v).upper() not in _KNOWN_CURRENCIES
                    )
                    if unknown:
                        return (
                            "currency_enum",
                            WARN,
                            [f"currency enum: unknown currency values {unknown}"],
                        )
                    return "currency_enum", PASS, []
                except Exception as exc:
                    return "currency_enum", WARN, [f"currency_enum check failed: {exc}"]
        return "currency_enum", PASS, []

    def _check_missing_partition(self, table: Any) -> tuple[str, str, list[str]]:
        """时间范围空洞（启发式）：日频/离散时间列上 distinct 天数远小于跨度。

        R32 P0-073..077：coverage grain 改为 session-based（非自然日），但本检查
        保守启发式（不强制交易日历），异常 → WARN。
        """
        cols = set(getattr(table, "column_names", []) or [])
        time_col = next((c for c in _TIME_COLUMN_CANDIDATES if c in cols), None)
        if time_col is None:
            return "missing_partition", PASS, []
        try:
            import pyarrow as pa
            import pyarrow.compute as pc

            arr = table.column(time_col)
            if pa.types.is_timestamp(arr.type) or pa.types.is_date(arr.type):
                d = pc.cast(arr, pa.date32())
            else:
                d = pc.cast(arr, pa.date32())
            distinct = int(pc.count_distinct(d).as_py() or 0)
            if distinct == 0 or int(d.length()) < 10:
                return "missing_partition", PASS, []
            lo = pc.min(d).as_py()
            hi = pc.max(d).as_py()
            span = (hi - lo).days + 1
            if span > 0 and distinct < span * 0.9:
                return (
                    "missing_partition",
                    WARN,
                    [f"missing partition: {distinct} distinct dates < span {span}"],
                )
            return "missing_partition", PASS, []
        except Exception as exc:
            return "missing_partition", WARN, [f"missing_partition check failed: {exc}"]

    def _check_schema_epoch(
        self, table: Any, schema_epoch: Any
    ) -> tuple[str, str, list[str]]:
        """manifest schema_epoch 存在时，用表 schema 的实际摘要核对。

        R32 P0-068..072：异常 → WARN（schema drift 非致命）。
        """
        if schema_epoch is None or table is None:
            return "schema_epoch", PASS, []
        try:
            actual = stable_digest_full(
                tuple(getattr(table, "column_names", []) or []),
                tuple(str(t) for t in table.schema.types),
            )
            ref = str(schema_epoch)
            if len(ref) >= 8 and ref.isalnum():
                if ref.startswith(actual[:16]) or actual.startswith(ref[:16]):
                    return "schema_epoch", PASS, []
                return (
                    "schema_epoch",
                    WARN,
                    ["schema/semantic epoch mismatch with manifest"],
                )
            return "schema_epoch", PASS, []
        except Exception as exc:
            return "schema_epoch", WARN, [f"schema_epoch check failed: {exc}"]

    # ---- 数据读取（只读，防御性） ----

    def _read(self, store: Any, dataset: str, params: Mapping[str, Any]):
        """从 store 读取数据，返回 (pyarrow.Table | None, snapshot | None)。"""
        read = getattr(store, "read", None)
        if callable(read):
            try:
                handle = read(dataset, **params)
            except TypeError:
                try:
                    handle = read(dataset)
                except Exception:
                    handle = None
            except Exception:
                handle = None
            if handle is not None:
                return self._handle_to_arrow(handle)
        read_uri = getattr(store, "read_uri", None)
        if callable(read_uri):
            try:
                handle = read_uri(dataset, **params)
            except Exception:
                handle = None
            if handle is not None:
                return self._handle_to_arrow(handle)
        return None, None

    def _handle_to_arrow(self, handle: Any):
        """把 ReadHandle / Arrow Table / polars DataFrame 归一化成 (pa.Table, snapshot)。"""
        snapshot = getattr(handle, "snapshot", None)
        to_arrow = getattr(handle, "to_arrow", None)
        if callable(to_arrow):
            try:
                return to_arrow(), snapshot
            except Exception:
                return None, snapshot
        # 已是 pyarrow.Table 或兼容对象
        if hasattr(handle, "column_names") and hasattr(handle, "num_rows"):
            return handle, snapshot
        return None, snapshot

    def _manifest_info(
        self, store: Any, dataset: str, params: Mapping[str, Any]
    ) -> tuple[int | None, Any]:
        """从 manifest 读期望行数 / schema_epoch；不可用 → (None, None)。"""
        try:
            from data_access.read.manifest import load_manifest_for_dataset

            manifest = load_manifest_for_dataset(store, dataset, **params)
        except Exception:
            return None, None
        if manifest is None:
            return None, None
        rows = getattr(manifest, "total_rows", None)
        if rows is None:
            files = getattr(manifest, "files", None) or []
            rows = sum(int(getattr(f, "rows", 0) or 0) for f in files)
        schema_epoch = getattr(manifest, "schema_epoch", None)
        return (int(rows) if rows else None), schema_epoch


def qualify_source(store: Any, dataset: str) -> str:
    """便捷：返回 PASS/WARN/BLOCK。"""
    return DataQualityService().qualify_source(store, dataset)


# ---- 内部工具 ----

def _relative_drift(actual: int, expected: int) -> float:
    if expected <= 0:
        return 0.0
    return abs(actual - expected) / expected


def _snapshot_to_dict(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    to_dict = getattr(snapshot, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return str(snapshot)
    return str(snapshot)
