"""R30-P0-001 —— BenchmarkReport：指标采集 + 对比门 + 证据落盘。

记录全部 R30-P0-001 要求的指标：

    TTFC(time-to-first-content) / TTDC(time-to-data-ready) / wall time /
    rows/s / GB/s / physical_object_count / physical_scan_count /
    bytes_scanned / bytes_returned / resolution_ms / snapshot_ms / schema_ms /
    calendar_ms / remote_list_ms / remote_head_ms / governor_wait_ms /
    duckdb_wait_ms / duckdb_execute_ms / polars_execute_ms / cache_hit /
    session_reuse / peak_rss_mb / spill_bytes

其中不能由当前本地合成路径测到的子项（如 remote_list_ms / remote_head_ms /
spill_bytes）按 None 记录并在 SUMMARY 里标 n/a——绝不假填充 0。
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

# 每个指标一个 slot；None = 未采集（SUMMARY 显示 n/a）。
_METRIC_SLOTS: tuple[str, ...] = (
    "ttfc_ms",
    "ttdc_ms",
    "wall_ms",
    "rows_per_sec",
    "gb_per_sec",
    "physical_object_count",
    "physical_scan_count",
    "bytes_scanned",
    "bytes_returned",
    "resolution_ms",
    "snapshot_ms",
    "schema_ms",
    "calendar_ms",
    "remote_list_ms",
    "remote_head_ms",
    "governor_wait_ms",
    "duckdb_wait_ms",
    "duckdb_execute_ms",
    "polars_execute_ms",
    "cache_hit",
    "session_reuse",
    "peak_rss_mb",
    "spill_bytes",
    "status",
)

_PASS, _FAIL, _CHECK, _SKIP = "PASS", "FAIL", "CHECK", "SKIP"


def peak_rss_mb() -> float | None:
    """进程当前峰值 RSS（MB）。Linux ru_maxrss 单位 KB；macOS 单位 bytes。"""
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if platform.system() == "Darwin":
        return round(kb / 1024.0 / 1024.0, 2)
    return round(kb / 1024.0, 2)


def benchmark_env() -> dict[str, Any]:
    """固定可复现的环境快照：commit SHA / CPU / RAM / 各引擎版本。"""
    env: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": None,
        "ram_gb": None,
        "version": None,
        "commit_sha": None,
    }
    try:
        from data_access import __version__ as _ver
        from data_access._build_meta import build_sha

        env["version"] = str(_ver)
        env["commit_sha"] = build_sha()
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    env["ram_gb"] = round(kb / 1024.0 / 1024.0, 1)
                    break
    except Exception:
        pass
    for mod, key in (
        ("duckdb", "duckdb_version"),
        ("polars", "polars_version"),
        ("pyarrow", "pyarrow_version"),
        ("pandas", "pandas_version"),
        ("numpy", "numpy_version"),
    ):
        try:
            m = __import__(mod)
            env[key] = getattr(m, "__version__", None)
        except Exception:
            env[key] = None
    return env


class BenchmarkReport:
    """一次固定 workload 的指标容器 + 证据写出。"""

    def __init__(self, name: str, workload: str, scale: str) -> None:
        self.name = name
        self.workload = workload
        self.scale = scale
        self.metrics: dict[str, Any] = {k: None for k in _METRIC_SLOTS}
        self.extra: dict[str, Any] = {}
        self._marks: dict[str, float] = {}
        self._gates: list[dict[str, Any]] = []
        self._started = time.perf_counter()

    # -- 里程碑 -----------------------------------------------------------
    def mark(self, label: str) -> None:
        """记录相对创建时刻的里程碑（用于 TTFC/TTDC）。"""
        self._marks[label] = (time.perf_counter() - self._started) * 1000.0

    def record_metric(self, name: str, value: Any) -> None:
        if name in self.metrics:
            self.metrics[name] = value
        else:
            self.extra[name] = value

    def record_extra(self, name: str, value: Any) -> None:
        self.extra[name] = value

    def finish(self) -> "BenchmarkReport":
        """收尾：把里程碑与峰值 RSS 固化进 metrics。"""
        wall = (time.perf_counter() - self._started) * 1000.0
        self.metrics["wall_ms"] = wall
        if "content" in self._marks:
            self.metrics["ttfc_ms"] = self._marks["content"]
        if "data" in self._marks:
            self.metrics["ttdc_ms"] = self._marks["data"]
        if self.metrics.get("peak_rss_mb") is None:
            self.metrics["peak_rss_mb"] = peak_rss_mb()
        return self

    # -- 派生吞吐 ---------------------------------------------------------
    def compute_throughput(self, rows: int, bytes_: int) -> None:
        """由行数/字节数 + wall_ms 派生 rows_per_sec / gb_per_sec。"""
        wall_s = (self.metrics.get("wall_ms") or 0.0) / 1000.0
        if wall_s > 0:
            self.metrics["rows_per_sec"] = round(rows / wall_s, 1)
            self.metrics["gb_per_sec"] = round((bytes_ / wall_s) / 1e9, 3)
        self.metrics["bytes_returned"] = int(bytes_)

    # -- gate -------------------------------------------------------------
    def add_gate(self, metric: str, rule: str, verdict: str, delta: float | None) -> None:
        self._gates.append(
            {"metric": metric, "rule": rule, "verdict": verdict, "delta": delta}
        )

    def gate_verdict(self) -> str:
        """FAIL > SKIP > CHECK > PASS。无 gate 时 PASS。"""
        if not self._gates:
            return _PASS
        if any(g.get("verdict") == _FAIL for g in self._gates):
            return _FAIL
        if any(g.get("verdict") == _SKIP for g in self._gates):
            return _SKIP
        if any(g.get("verdict") == _CHECK for g in self._gates):
            return _CHECK
        return _PASS

    def mark_skipped(self, reason: str) -> None:
        """COS 等不可用场景：记录 SKIP 状态，绝不假通过。"""
        self._gates.append(
            {"metric": "status", "rule": "workload runnable", "verdict": _SKIP, "delta": None}
        )
        self.record_metric("status", "SKIP")
        self.record_extra("skip_reason", reason)

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "workload": self.workload,
            "scale": self.scale,
            "verdict": self.gate_verdict(),
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
            "gates": list(self._gates),
        }
        return d

    # -- 证据 -------------------------------------------------------------
    def write_evidence(self, out_dir: str | Path) -> Path:
        """写 BENCHMARK_ENV.json / BENCHMARK_RESULTS.parquet / BENCHMARK_SUMMARY.md。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        env = benchmark_env()
        (out / "BENCHMARK_ENV.json").write_text(
            json.dumps(env, indent=2, default=str), encoding="utf-8"
        )
        results = self.to_dict()
        try:
            import pandas as pd

            df = pd.DataFrame([results])
            df.to_parquet(out / "BENCHMARK_RESULTS.parquet", index=False)
        except Exception:
            (out / "BENCHMARK_RESULTS.json").write_text(
                json.dumps(results, indent=2, default=str), encoding="utf-8"
            )
        (out / "BENCHMARK_SUMMARY.md").write_text(
            self._summary_markdown(env), encoding="utf-8"
        )
        return out

    def _summary_markdown(self, env: dict[str, Any]) -> str:
        lines: list[str] = [
            f"# Benchmark {self.name}  ({self.workload})",
            "",
            f"- scale: `{self.scale}`  verdict: **{self.gate_verdict()}**",
            f"- commit: {env.get('commit_sha') or 'n/a'}  version: {env.get('version') or 'n/a'}",
            f"- platform: {env.get('platform')}  cpu: {env.get('cpu_model') or env.get('cpu_count')}",
            f"- ram_gb: {env.get('ram_gb')}  python: {env.get('python')}",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for k, v in self.metrics.items():
            if v is None:
                lines.append(f"| {k} | n/a |")
            else:
                lines.append(f"| {k} | {v} |")
        for k, v in self.extra.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        if self._gates:
            lines.append("## Gates")
            lines.append("")
            for g in self._gates:
                lines.append(f"- **{g['verdict']}**  {g['metric']}: {g['rule']} (delta={g['delta']})")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BenchmarkReport(name={self.name}, workload={self.workload}, "
            f"scale={self.scale}, verdict={self.gate_verdict()})"
        )


def _gate_verdict_for(deltas: dict[str, float | None]) -> list[dict[str, Any]]:
    """按 R30-P0-001 门规产出一组 gate 判定。

    规则：
      - throughput（warm_throughput / rows_per_sec）：after 相对 before 回退 >10% → FAIL
      - P95 类指标：after 比 before 上涨 >10% → FAIL（延迟上涨=回退）
      - peak_rss_mb：after 比 before 上涨 >15% → CHECK
    """
    gates: list[dict[str, Any]] = []
    for tk in ("warm_throughput", "rows_per_sec", "gb_per_sec"):
        if tk in deltas and deltas[tk] is not None:
            verdict = _FAIL if deltas[tk] < -0.10 else _PASS
            gates.append(
                {
                    "metric": tk,
                    "rule": "throughput >= 90% of baseline",
                    "verdict": verdict,
                    "delta": round(float(deltas[tk]), 4),
                }
            )
            break
    for k in sorted(deltas):
        if "p95" in k and deltas[k] is not None:
            verdict = _FAIL if deltas[k] > 0.10 else _PASS
            gates.append(
                {
                    "metric": k,
                    "rule": "P95 regression <= +10%",
                    "verdict": verdict,
                    "delta": round(float(deltas[k]), 4),
                }
            )
    if "peak_rss_mb" in deltas and deltas["peak_rss_mb"] is not None:
        if deltas["peak_rss_mb"] > 0.15:
            gates.append(
                {
                    "metric": "peak_rss_mb",
                    "rule": "peak RSS <= +15%",
                    "verdict": _CHECK,
                    "delta": round(float(deltas["peak_rss_mb"]), 4),
                }
            )
    return gates


def compare(before: BenchmarkReport, after: BenchmarkReport) -> dict[str, Any]:
    """对比两次运行：相对差 + gate 判定。

    ``deltas`` 为 {metric: (after-before)/|before|}；被除数为 0 或任一为 None 的
    指标记 None（不参与 gate）。返回结构含 ``verdict`` 字符串。
    """
    keys = sorted(set(before.metrics) | set(after.metrics) | set(before.extra) | set(after.extra))
    deltas: dict[str, float | None] = {}
    for k in keys:
        b = before.metrics.get(k)
        if b is None:
            b = before.extra.get(k)
        a = after.metrics.get(k)
        if a is None:
            a = after.extra.get(k)
        if b is None or a is None or isinstance(b, (str, bool)) or isinstance(a, (str, bool)):
            continue
        try:
            bf, af = float(b), float(a)
        except (TypeError, ValueError):
            continue
        if bf == 0:
            deltas[k] = None
        else:
            deltas[k] = (af - bf) / abs(bf)

    gates = _gate_verdict_for(deltas)
    verdict = _PASS
    if any(g.get("verdict") == _FAIL for g in gates):
        verdict = _FAIL
    elif any(g.get("verdict") == _SKIP for g in gates):
        verdict = _SKIP
    elif any(g.get("verdict") == _CHECK for g in gates):
        verdict = _CHECK
    return {
        "workload": before.name,
        "before": before.name,
        "after": after.name,
        "deltas": {k: (round(float(v), 4) if v is not None else None) for k, v in deltas.items()},
        "gates": gates,
        "verdict": verdict,
    }
