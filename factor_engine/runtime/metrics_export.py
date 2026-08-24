"""factor_engine 运行指标导出：factor_run + pipeline summary + data_access telemetry。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factor_engine.storage.catalog import FactorCatalog


def _iso_to_unix_nano(value: str | None) -> str:
    """将 ISO 时间戳转为 OTLP JSON 要求的 Unix 纳秒字符串。"""
    if not value:
        dt = datetime.now(timezone.utc)
    else:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp() * 1_000_000_000))


def collect_data_access_telemetry() -> dict[str, Any]:
    """读取 data_access 进程内 telemetry 快照（可选依赖）。"""
    try:
        from data_access.read.telemetry import get_counters_snapshot
    except ImportError:
        return {"available": False, "operators": {}}

    snapshot = get_counters_snapshot()
    operators: dict[str, Any] = {}
    for name, counters in snapshot.items():
        operators[name] = {
            "total_queries": counters.total_queries,
            "slow_queries": counters.slow_queries,
            "total_elapsed_ms": round(counters.total_elapsed_ms, 3),
        }
    return {"available": True, "operators": operators}


def summarize_pipeline_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """从 pipeline 单 config 结果列表聚合指标。"""
    success = [item for item in results if item.get("status") == "success"]
    failed = [item for item in results if item.get("status") == "failed"]
    total_rows = 0
    total_non_null = 0
    materialized_rows = 0
    for item in success:
        result_block = item.get("result") or {}
        total_rows += int(result_block.get("row_count") or 0)
        total_non_null += int(result_block.get("non_null_count") or 0)
        mat = item.get("materialization") or {}
        materialized_rows += int(mat.get("rows_written") or 0)

    return {
        "configs_total": len(results),
        "configs_success": len(success),
        "configs_failed": len(failed),
        "total_result_rows": total_rows,
        "total_non_null_rows": total_non_null,
        "total_materialized_rows": materialized_rows,
        "incremental_runs": sum(
            1 for item in success if item.get("mode") == "materialize_incremental"
        ),
        "failed_config_names": [item.get("config_name") for item in failed],
    }


def enrich_pipeline_summary(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """为 run_summary.json 附加 metrics 块。"""
    enriched = dict(summary)
    enriched["metrics"] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": summarize_pipeline_results(results),
        "data_access": collect_data_access_telemetry(),
        "per_config": [
            {
                "config_name": item.get("config_name"),
                "status": item.get("status"),
                "mode": item.get("mode"),
                "factor_name": item.get("factor_name"),
                "row_count": (item.get("result") or {}).get("row_count"),
                "rows_written": (item.get("materialization") or {}).get("rows_written"),
                "run_id": (item.get("materialization") or {}).get("run_id"),
                "partitions_failed": (item.get("materialization") or {}).get(
                    "partitions_failed"
                ),
            }
            for item in results
        ],
    }
    return enriched


def export_factor_runs_jsonl(
    catalog: FactorCatalog,
    output_path: str | Path,
    *,
    factor_id: str | None = None,
    limit: int = 500,
) -> int:
    """将 factor_run 记录导出为 JSONL。返回写入行数。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if factor_id is not None:
        runs = catalog.list_runs(factor_id, limit=limit)
    else:
        runs = catalog.list_recent_runs(limit=limit)

    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
    return len(runs)


def write_pipeline_metrics_file(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """将 enrich 后的 summary 写入 metrics JSON 文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _prom_label(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


def to_prometheus_text(summary: dict[str, Any]) -> str:
    """将 pipeline summary 转为 Prometheus text exposition 格式。"""
    metrics = summary.get("metrics") or {}
    pipeline = metrics.get("pipeline") or {}
    lines: list[str] = []

    def gauge(name: str, value: float | int, labels: dict[str, str] | None = None) -> None:
        label_text = ""
        if labels:
            parts = [f'{key}={_prom_label(val)}' for key, val in sorted(labels.items())]
            label_text = "{" + ",".join(parts) + "}"
        lines.append(f"factor_engine_{name}{label_text} {value}")

    gauge("configs_total", int(pipeline.get("configs_total") or summary.get("configs_total") or 0))
    gauge("configs_success", int(pipeline.get("configs_success") or summary.get("configs_success") or 0))
    gauge("configs_failed", int(pipeline.get("configs_failed") or summary.get("configs_failed") or 0))
    gauge("total_result_rows", int(pipeline.get("total_result_rows") or 0))
    gauge("total_materialized_rows", int(pipeline.get("total_materialized_rows") or 0))
    gauge("incremental_runs", int(pipeline.get("incremental_runs") or 0))

    da = metrics.get("data_access") or {}
    if da.get("available"):
        for operator, block in (da.get("operators") or {}).items():
            labels = {"operator": operator}
            gauge("data_access_queries_total", int(block.get("total_queries") or 0), labels)
            gauge("data_access_slow_queries_total", int(block.get("slow_queries") or 0), labels)
            gauge(
                "data_access_elapsed_ms_total",
                float(block.get("total_elapsed_ms") or 0.0),
                labels,
            )

    for item in metrics.get("per_config") or []:
        labels = {
            "config_name": str(item.get("config_name") or "unknown"),
            "status": str(item.get("status") or "unknown"),
        }
        gauge(
            "config_rows_written",
            int(item.get("rows_written") or item.get("row_count") or 0),
            labels,
        )

    return "\n".join(lines) + "\n"


def to_otlp_json(summary: dict[str, Any]) -> dict[str, Any]:
    """生成 OTLP 兼容的 resourceMetrics JSON（供 collector 桥接）。"""
    metrics = summary.get("metrics") or {}
    pipeline = metrics.get("pipeline") or {}
    exported_at = metrics.get("exported_at") or datetime.now(timezone.utc).isoformat()

    def _gauge(name: str, value: float | int) -> dict[str, Any]:
        return {
            "name": name,
            "gauge": {
                "dataPoints": [
                    {
                        "timeUnixNano": _iso_to_unix_nano(exported_at),
                        "asDouble": float(value),
                    }
                ]
            },
        }

    metric_items = [
        _gauge("factor_engine.configs_total", pipeline.get("configs_total") or 0),
        _gauge("factor_engine.configs_success", pipeline.get("configs_success") or 0),
        _gauge("factor_engine.configs_failed", pipeline.get("configs_failed") or 0),
        _gauge("factor_engine.total_result_rows", pipeline.get("total_result_rows") or 0),
        _gauge("factor_engine.total_materialized_rows", pipeline.get("total_materialized_rows") or 0),
    ]

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "factor_engine.service.name", "value": {"stringValue": "factor_engine"}},
                    ]
                },
                "scopeMetrics": [{"metrics": metric_items}],
            }
        ]
    }


def write_prometheus_metrics_file(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """将 pipeline summary 转为 Prometheus text 并写入文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_prometheus_text(summary), encoding="utf-8")
    return path


def write_otlp_metrics_file(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """将 pipeline summary 转为 OTLP JSON 并写入文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_otlp_json(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def push_otlp_http(
    summary: dict[str, Any],
    endpoint: str,
    *,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """通过 OTLP/HTTP JSON 推送指标（默认 collector :4318/v1/metrics）。"""
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/")
    if not url.endswith("/v1/metrics"):
        url = f"{url}/v1/metrics"

    payload = json.dumps(to_otlp_json(summary), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "endpoint": url,
                "status": response.status,
                "body": body[:500],
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "endpoint": url,
            "status": exc.code,
            "error": body[:500],
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "endpoint": url, "error": str(exc)}


def push_otlp_grpc(
    summary: dict[str, Any],
    endpoint: str,
    *,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """OTLP/gRPC 推送；若安装 opentelemetry SDK 则走原生 gRPC，否则 HTTP 回退。"""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        import socket
        host = endpoint.replace("http://", "").replace("https://", "")
        if ":" not in host:
            host = f"{host}:4317"
        probe_host, probe_port = host.rsplit(":", 1)
        with socket.create_connection((probe_host, int(probe_port)), timeout=timeout_sec):
            pass
        exporter = OTLPMetricExporter(endpoint=host, insecure=True)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
        provider = MeterProvider(resource=Resource.create({"factor_engine.service.name": "factor_engine"}), metric_readers=[reader])
        meter = provider.get_meter("factor_engine")
        metrics = summary.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                meter.create_counter(key).add(float(value))
        provider.force_flush(timeout_millis=int(timeout_sec * 1000))
        provider.shutdown()
        return {"ok": True, "endpoint": host, "transport": "grpc_native"}
    except (ImportError, OSError, RuntimeError, TimeoutError) as exc:
        http_endpoint = endpoint.replace(":4317", ":4318")
        if "://" not in http_endpoint:
            http_endpoint = f"http://{http_endpoint}"
        result = push_otlp_http(summary, http_endpoint, timeout_sec=timeout_sec)
        result["transport"] = "http_fallback_from_grpc"
        return result
