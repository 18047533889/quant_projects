#!/usr/bin/env python3
"""
Generate visual performance regression reports.

Creates charts showing performance trends over time with regression highlighting.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tracker import BenchmarkTracker


@dataclass
class TimeSeriesPoint:
    """Single point in performance time series."""
    timestamp: datetime
    commit_sha: str
    value: float
    commit_date: datetime


class RegressionReporter:
    """Generate regression reports with visualizations."""

    def __init__(self, tracker: BenchmarkTracker):
        self.tracker = tracker

    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime from various formats."""
        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass

        # Try git format: "2026-08-14 01:13:01 +0800"
        try:
            # Parse without timezone, git format is complex
            date_part = date_str.rsplit("+", 1)[0].rsplit("-", 1)[0].strip()
            return datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass

        # Fallback to now
        return datetime.now()

    def build_time_series(self, benchmark_name: str) -> list[TimeSeriesPoint]:
        """Build time series data for a benchmark."""
        results = self.tracker.load_results(benchmark_name)

        points = []
        for result in results:
            points.append(TimeSeriesPoint(
                timestamp=datetime.fromisoformat(result.timestamp),
                commit_sha=result.commit_sha,
                value=result.value,
                commit_date=self._parse_datetime(result.commit_date),
            ))

        return sorted(points, key=lambda p: p.timestamp)

    def generate_html_report(
        self,
        comparisons: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Generate interactive HTML report with charts."""
        # Group comparisons by benchmark suite
        by_suite = defaultdict(list)
        for comp in comparisons:
            suite = comp["benchmark_name"].split("_")[0]
            by_suite[suite].append(comp)

        # Build time series data for each benchmark
        time_series_data = {}
        for comp in comparisons:
            bench_name = comp["benchmark_name"]
            if bench_name not in time_series_data:
                series = self.build_time_series(bench_name)
                if series:
                    time_series_data[bench_name] = [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "commit": p.commit_sha[:8],
                            "value": p.value,
                        }
                        for p in series
                    ]

        html_content = self._build_html_template(
            comparisons=comparisons,
            by_suite=by_suite,
            time_series_data=time_series_data,
        )

        output_path.write_text(html_content, encoding="utf-8")

    def _build_html_template(
        self,
        comparisons: list[dict[str, Any]],
        by_suite: dict[str, list[dict[str, Any]]],
        time_series_data: dict[str, list[dict[str, Any]]],
    ) -> str:
        """Build HTML report template."""
        regressions = [c for c in comparisons if c["is_regression"]]
        improvements = [c for c in comparisons if c["is_improvement"]]

        # Build summary section
        summary_html = f"""
<div class="summary">
    <h2>Summary</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(comparisons)}</div>
            <div class="stat-label">Total Comparisons</div>
        </div>
        <div class="stat-card regression">
            <div class="stat-value">{len(regressions)}</div>
            <div class="stat-label">Regressions</div>
        </div>
        <div class="stat-card improvement">
            <div class="stat-value">{len(improvements)}</div>
            <div class="stat-label">Improvements</div>
        </div>
        <div class="stat-card neutral">
            <div class="stat-value">{len(comparisons) - len(regressions) - len(improvements)}</div>
            <div class="stat-label">Neutral</div>
        </div>
    </div>
</div>
"""

        # Build regression details
        regression_html = ""
        if regressions:
            regression_html = '<div class="section"><h2>🔴 Regressions Detected</h2><table class="results-table">'
            regression_html += "<thead><tr><th>Benchmark</th><th>Metric</th><th>Current</th><th>Baseline</th><th>Change</th></tr></thead><tbody>"
            for comp in regressions:
                change_class = "regression" if comp["is_regression"] else ""
                regression_html += f"""
<tr class="{change_class}">
    <td>{comp['benchmark_name']}</td>
    <td>{comp['metric_name']}</td>
    <td>{comp['current_value']:.3f}</td>
    <td>{comp['baseline_value']:.3f}</td>
    <td>{comp['percent_change']:+.1f}%</td>
</tr>
"""
            regression_html += "</tbody></table></div>"

        # Build improvement details
        improvement_html = ""
        if improvements:
            improvement_html = '<div class="section"><h2>🟢 Performance Improvements</h2><table class="results-table">'
            improvement_html += "<thead><tr><th>Benchmark</th><th>Metric</th><th>Current</th><th>Baseline</th><th>Change</th></tr></thead><tbody>"
            for comp in improvements:
                improvement_html += f"""
<tr class="improvement">
    <td>{comp['benchmark_name']}</td>
    <td>{comp['metric_name']}</td>
    <td>{comp['current_value']:.3f}</td>
    <td>{comp['baseline_value']:.3f}</td>
    <td>{comp['percent_change']:+.1f}%</td>
</tr>
"""
            improvement_html += "</tbody></table></div>"

        # Build time series charts
        charts_html = self._build_charts_html(time_series_data)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Regression Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0A0D12;
            color: #E5E7EB;
            line-height: 1.6;
            padding: clamp(1rem, 3vw, 3rem);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: clamp(1.75rem, 4vw, 2.5rem);
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            color: #F9FAFB;
        }}

        .subtitle {{
            color: #9CA3AF;
            font-size: clamp(0.875rem, 2vw, 1rem);
            margin-bottom: 2rem;
        }}

        .summary {{
            background: #0F131C;
            border-radius: 999px;
            padding: clamp(1.5rem, 3vw, 2rem);
            margin-bottom: 2rem;
            border: 1px solid #1E2636;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .stat-card {{
            background: #161D2B;
            border-radius: 999px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #1E2636;
        }}

        .stat-value {{
            font-size: clamp(1.875rem, 4vw, 2.5rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #38BDF8;
        }}

        .stat-card.regression .stat-value {{ color: #EF4444; }}
        .stat-card.improvement .stat-value {{ color: #10B981; }}
        .stat-card.neutral .stat-value {{ color: #6B7280; }}

        .stat-label {{
            font-size: clamp(0.75rem, 1.5vw, 0.875rem);
            color: #9CA3AF;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .section {{
            background: #0F131C;
            border-radius: 16px;
            padding: clamp(1.5rem, 3vw, 2rem);
            margin-bottom: 2rem;
            border: 1px solid #1E2636;
        }}

        h2 {{
            font-size: clamp(1.25rem, 3vw, 1.5rem);
            font-weight: 600;
            margin-bottom: 1.5rem;
            letter-spacing: -0.01em;
        }}

        .results-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: clamp(0.875rem, 1.5vw, 0.9375rem);
        }}

        .results-table thead {{
            border-bottom: 1px solid #1E2636;
        }}

        .results-table th {{
            text-align: left;
            padding: 0.75rem 1rem;
            font-weight: 600;
            color: #9CA3AF;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        .results-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #161D2B;
        }}

        .results-table tr.regression td {{
            background: rgba(239, 68, 68, 0.1);
        }}

        .results-table tr.improvement td {{
            background: rgba(16, 185, 129, 0.1);
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 1.5rem;
        }}

        @media (max-width: 768px) {{
            .results-table {{
                font-size: 0.75rem;
            }}

            .results-table th,
            .results-table td {{
                padding: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Regression Report</h1>
        <div class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

        {summary_html}
        {regression_html}
        {improvement_html}
        {charts_html}
    </div>
</body>
</html>
"""

    def _build_charts_html(self, time_series_data: dict[str, list[dict[str, Any]]]) -> str:
        """Build chart sections for time series data."""
        if not time_series_data:
            return ""

        charts_html = '<div class="section"><h2>Performance Trends</h2>'

        for idx, (bench_name, series) in enumerate(time_series_data.items()):
            if len(series) < 2:
                continue

            labels = [p["commit"] for p in series]
            values = [p["value"] for p in series]

            charts_html += f"""
<div class="chart-container">
    <canvas id="chart_{idx}"></canvas>
</div>
<script>
    new Chart(document.getElementById('chart_{idx}'), {{
        type: 'line',
        data: {{
            labels: {json.dumps(labels)},
            datasets: [{{
                label: '{bench_name}',
                data: {json.dumps(values)},
                borderColor: '#38BDF8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                tension: 0.4,
                fill: true,
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    labels: {{
                        color: '#E5E7EB',
                        font: {{ size: 14 }}
                    }}
                }},
                title: {{
                    display: true,
                    text: '{bench_name}',
                    color: '#F9FAFB',
                    font: {{ size: 16, weight: '600' }}
                }}
            }},
            scales: {{
                y: {{
                    ticks: {{ color: '#9CA3AF' }},
                    grid: {{ color: '#1E2636' }}
                }},
                x: {{
                    ticks: {{ color: '#9CA3AF' }},
                    grid: {{ color: '#1E2636' }}
                }}
            }}
        }}
    }});
</script>
"""

        charts_html += '</div>'
        return charts_html

    def generate_markdown_report(
        self,
        comparisons: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Generate markdown report for CI/PR comments."""
        regressions = [c for c in comparisons if c["is_regression"]]
        improvements = [c for c in comparisons if c["is_improvement"]]

        md_lines = [
            "# Performance Regression Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- Total Comparisons: {len(comparisons)}",
            f"- 🔴 Regressions: {len(regressions)}",
            f"- 🟢 Improvements: {len(improvements)}",
            f"- ⚪ Neutral: {len(comparisons) - len(regressions) - len(improvements)}",
            "",
        ]

        if regressions:
            md_lines.extend([
                "## 🔴 Regressions Detected",
                "",
                "| Benchmark | Metric | Current | Baseline | Change |",
                "|-----------|--------|---------|----------|--------|",
            ])
            for comp in regressions:
                md_lines.append(
                    f"| {comp['benchmark_name']} | {comp['metric_name']} | "
                    f"{comp['current_value']:.3f} | {comp['baseline_value']:.3f} | "
                    f"**{comp['percent_change']:+.1f}%** |"
                )
            md_lines.append("")

        if improvements:
            md_lines.extend([
                "## 🟢 Performance Improvements",
                "",
                "| Benchmark | Metric | Current | Baseline | Change |",
                "|-----------|--------|---------|----------|--------|",
            ])
            for comp in improvements:
                md_lines.append(
                    f"| {comp['benchmark_name']} | {comp['metric_name']} | "
                    f"{comp['current_value']:.3f} | {comp['baseline_value']:.3f} | "
                    f"**{comp['percent_change']:+.1f}%** |"
                )
            md_lines.append("")

        output_path.write_text("\n".join(md_lines), encoding="utf-8")


def main():
    """CLI for generating regression reports."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_json",
        type=Path,
        help="Path to comparison results JSON (from compare.py --json-output)"
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Output HTML report path"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Output Markdown report path"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory with historical results"
    )
    args = parser.parse_args()

    if not args.comparison_json.exists():
        print(f"Error: {args.comparison_json} not found")
        return 1

    if not args.html and not args.markdown:
        print("Error: Specify at least one output format (--html or --markdown)")
        return 1

    # Load comparison data
    comparison_data = json.loads(args.comparison_json.read_text(encoding="utf-8"))
    comparisons = comparison_data.get("comparisons", [])

    # Generate reports
    tracker = BenchmarkTracker(results_dir=args.results_dir)
    reporter = RegressionReporter(tracker)

    if args.html:
        reporter.generate_html_report(comparisons, args.html)
        print(f"✓ HTML report written to {args.html}")

    if args.markdown:
        reporter.generate_markdown_report(comparisons, args.markdown)
        print(f"✓ Markdown report written to {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
