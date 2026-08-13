"""
Report generation module for creating HTML and PDF data quality reports.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json

from .profiler import ProfileResult
from .anomaly_detection import AnomalyResult


class ReportFormat(Enum):
    """Supported report formats."""

    HTML = "html"
    JSON = "json"


class ReportGenerator:
    """
    Generate comprehensive data quality reports in multiple formats.

    Creates HTML and JSON reports combining profiling and anomaly detection results.
    """

    def __init__(
        self,
        title: str = "Data Quality Report",
        include_charts: bool = True,
    ):
        """
        Initialize report generator.

        Args:
            title: Report title
            include_charts: Whether to include visual charts (HTML only)
        """
        self.title = title
        self.include_charts = include_charts

    def generate(
        self,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult] = None,
        format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate data quality report.

        Args:
            profile_result: Data profiling results
            anomaly_result: Optional anomaly detection results
            format: Report format (HTML or JSON)
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        if format == ReportFormat.HTML:
            content = self._generate_html(profile_result, anomaly_result)
        elif format == ReportFormat.JSON:
            content = self._generate_json(profile_result, anomaly_result)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def _generate_html(
        self,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult],
    ) -> str:
        """Generate HTML report."""
        html_parts = []

        # Header
        html_parts.append(self._html_header())

        # Summary section
        html_parts.append(self._html_summary(profile_result, anomaly_result))

        # Anomalies section
        if anomaly_result:
            html_parts.append(self._html_anomalies(anomaly_result))

        # Column profiles section
        html_parts.append(self._html_column_profiles(profile_result))

        # Coverage section
        if profile_result.coverage_matrix is not None:
            html_parts.append(self._html_coverage(profile_result))

        # Correlation section
        if profile_result.correlations is not None:
            html_parts.append(self._html_correlations(profile_result))

        # Time series section
        if profile_result.date_column:
            html_parts.append(self._html_time_series(profile_result))

        # Footer
        html_parts.append(self._html_footer())

        return "\n".join(html_parts)

    def _html_header(self) -> str:
        """Generate HTML header with CSS."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: #05070C;
            color: #E5E7EB;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.6;
            padding: clamp(1.5rem, 4vw, 3rem);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
            color: #F9FAFB;
        }}

        h2 {{
            font-size: clamp(1.5rem, 3vw, 2rem);
            font-weight: 600;
            letter-spacing: -0.01em;
            margin-top: 3rem;
            margin-bottom: 1.5rem;
            color: #F9FAFB;
            border-bottom: 2px solid #1E2636;
            padding-bottom: 0.5rem;
        }}

        h3 {{
            font-size: clamp(1.25rem, 2.5vw, 1.5rem);
            font-weight: 500;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #D1D5DB;
        }}

        .timestamp {{
            color: #9CA3AF;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: #0F131C;
            border: 1px solid #1E2636;
            border-radius: 12px;
            padding: 1.5rem;
        }}

        .stat-label {{
            color: #9CA3AF;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .stat-value {{
            font-size: clamp(1.75rem, 3vw, 2.5rem);
            font-weight: 600;
            color: #38BDF8;
        }}

        .stat-value.warning {{
            color: #E9A568;
        }}

        .stat-value.error {{
            color: #EF4444;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: #0F131C;
            border-radius: 12px;
            overflow: hidden;
            margin: 1rem 0;
        }}

        thead {{
            background: #161D2B;
        }}

        th {{
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #F9FAFB;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 0.875rem 1rem;
            border-top: 1px solid #1E2636;
            color: #D1D5DB;
        }}

        tr:hover {{
            background: #161D2B;
        }}

        .severity-high {{
            background: #7F1D1D;
            color: #FEE2E2;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        .severity-medium {{
            background: #78350F;
            color: #FED7AA;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        .severity-low {{
            background: #065F46;
            color: #A7F3D0;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #1E2636;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #38BDF8, #6EE7B7);
            border-radius: 999px;
            transition: width 0.3s ease;
        }}

        .numeric-stat {{
            font-family: "SF Mono", Monaco, "Courier New", monospace;
            font-size: 0.875rem;
            color: #D1D5DB;
        }}

        .alert {{
            background: #1E2636;
            border-left: 4px solid #38BDF8;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}

        .alert.warning {{
            border-left-color: #E9A568;
        }}

        .alert.error {{
            border-left-color: #EF4444;
        }}

        .column-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .column-card {{
            background: #0F131C;
            border: 1px solid #1E2636;
            border-radius: 12px;
            padding: 1.5rem;
        }}

        .column-name {{
            font-weight: 600;
            color: #F9FAFB;
            margin-bottom: 0.75rem;
            font-size: 1.1rem;
        }}

        .column-type {{
            color: #6EE7B7;
            font-size: 0.875rem;
            font-family: "SF Mono", Monaco, monospace;
            margin-bottom: 1rem;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #1E2636;
        }}

        .stat-row:last-child {{
            border-bottom: none;
        }}

        .stat-row-label {{
            color: #9CA3AF;
            font-size: 0.875rem;
        }}

        .stat-row-value {{
            color: #D1D5DB;
            font-weight: 500;
        }}

        footer {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #1E2636;
            color: #6B7280;
            font-size: 0.875rem;
            text-align: center;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{self.title}</h1>
    <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
"""

    def _html_summary(
        self,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult],
    ) -> str:
        """Generate summary section."""
        parts = ['<h2>Summary</h2>', '<div class="summary-grid">']

        # Basic stats
        parts.append(
            f"""
    <div class="stat-card">
        <div class="stat-label">Total Rows</div>
        <div class="stat-value">{profile_result.row_count:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Total Columns</div>
        <div class="stat-value">{profile_result.column_count}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Memory Usage</div>
        <div class="stat-value">{profile_result.memory_usage_mb:.1f} MB</div>
    </div>
"""
        )

        # Anomaly stats
        if anomaly_result:
            severity_class = (
                "error" if anomaly_result.severity_counts.get("high", 0) > 0
                else "warning" if anomaly_result.severity_counts.get("medium", 0) > 0
                else ""
            )

            parts.append(
                f"""
    <div class="stat-card">
        <div class="stat-label">Total Anomalies</div>
        <div class="stat-value {severity_class}">{anomaly_result.total_anomalies}</div>
    </div>
"""
            )

        parts.append("</div>")

        return "\n".join(parts)

    def _html_anomalies(self, anomaly_result: AnomalyResult) -> str:
        """Generate anomalies section."""
        parts = ['<h2>Anomalies Detected</h2>']

        if anomaly_result.total_anomalies == 0:
            parts.append('<div class="alert">No anomalies detected.</div>')
            return "\n".join(parts)

        # Severity breakdown
        parts.append('<div class="summary-grid">')
        for severity in ["high", "medium", "low"]:
            count = anomaly_result.severity_counts.get(severity, 0)
            parts.append(
                f"""
    <div class="stat-card">
        <div class="stat-label">{severity.capitalize()} Severity</div>
        <div class="stat-value">{count}</div>
    </div>
"""
            )
        parts.append("</div>")

        # Anomalies table
        parts.append(
            """
<h3>Details</h3>
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Column</th>
            <th>Severity</th>
            <th>Description</th>
            <th>Affected</th>
        </tr>
    </thead>
    <tbody>
"""
        )

        for anomaly in anomaly_result.anomalies[:100]:  # Limit to first 100
            affected = len(anomaly.affected_rows) if anomaly.affected_rows else "N/A"
            parts.append(
                f"""
        <tr>
            <td>{anomaly.anomaly_type.value}</td>
            <td><code>{anomaly.column}</code></td>
            <td><span class="severity-{anomaly.severity}">{anomaly.severity}</span></td>
            <td>{anomaly.description}</td>
            <td>{affected}</td>
        </tr>
"""
            )

        parts.append("    </tbody>\n</table>")

        return "\n".join(parts)

    def _html_column_profiles(self, profile_result: ProfileResult) -> str:
        """Generate column profiles section."""
        parts = ['<h2>Column Profiles</h2>', '<div class="column-grid">']

        for col_name, profile in profile_result.column_profiles.items():
            parts.append(f'<div class="column-card">')
            parts.append(f'<div class="column-name">{col_name}</div>')
            parts.append(f'<div class="column-type">{profile.dtype}</div>')

            # Basic stats
            parts.append('<div>')
            parts.append(
                f'<div class="stat-row">'
                f'<span class="stat-row-label">Count</span>'
                f'<span class="stat-row-value">{profile.count:,}</span></div>'
            )
            parts.append(
                f'<div class="stat-row">'
                f'<span class="stat-row-label">Missing</span>'
                f'<span class="stat-row-value">{profile.null_count:,} ({profile.null_percentage:.1f}%)</span></div>'
            )
            parts.append(
                f'<div class="stat-row">'
                f'<span class="stat-row-label">Unique</span>'
                f'<span class="stat-row-value">{profile.unique_count:,}</span></div>'
            )

            # Numeric stats
            if profile.mean is not None:
                parts.append(
                    f'<div class="stat-row">'
                    f'<span class="stat-row-label">Mean</span>'
                    f'<span class="stat-row-value numeric-stat">{profile.mean:.4f}</span></div>'
                )
                parts.append(
                    f'<div class="stat-row">'
                    f'<span class="stat-row-label">Std Dev</span>'
                    f'<span class="stat-row-value numeric-stat">{profile.std:.4f}</span></div>'
                )
                parts.append(
                    f'<div class="stat-row">'
                    f'<span class="stat-row-label">Min</span>'
                    f'<span class="stat-row-value numeric-stat">{profile.min:.4f}</span></div>'
                )
                parts.append(
                    f'<div class="stat-row">'
                    f'<span class="stat-row-label">Max</span>'
                    f'<span class="stat-row-value numeric-stat">{profile.max:.4f}</span></div>'
                )
                parts.append(
                    f'<div class="stat-row">'
                    f'<span class="stat-row-label">Median</span>'
                    f'<span class="stat-row-value numeric-stat">{profile.median:.4f}</span></div>'
                )

                if profile.inf_count > 0:
                    parts.append(
                        f'<div class="stat-row">'
                        f'<span class="stat-row-label">Infinite</span>'
                        f'<span class="stat-row-value" style="color: #EF4444;">{profile.inf_count}</span></div>'
                    )

            parts.append('</div>')
            parts.append('</div>')

        parts.append('</div>')

        return "\n".join(parts)

    def _html_coverage(self, profile_result: ProfileResult) -> str:
        """Generate coverage section."""
        parts = ['<h2>Data Coverage</h2>', '<table>', '<thead><tr><th>Column</th><th>Coverage</th><th>Visual</th></tr></thead>', '<tbody>']

        for _, row in profile_result.coverage_matrix.iterrows():
            col_name = row["column"]
            coverage = row["coverage_pct"]

            parts.append(
                f"""
        <tr>
            <td><code>{col_name}</code></td>
            <td>{coverage:.1f}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {coverage}%"></div>
                </div>
            </td>
        </tr>
"""
            )

        parts.append('</tbody></table>')

        return "\n".join(parts)

    def _html_correlations(self, profile_result: ProfileResult) -> str:
        """Generate correlations section."""
        parts = ['<h2>Correlations</h2>']

        corr_matrix = profile_result.correlations

        if corr_matrix is None or len(corr_matrix) == 0:
            parts.append('<div class="alert">No correlations computed.</div>')
            return "\n".join(parts)

        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        if high_corr:
            parts.append('<h3>High Correlations (|r| ≥ 0.7)</h3>')
            parts.append('<table><thead><tr><th>Column 1</th><th>Column 2</th><th>Correlation</th></tr></thead><tbody>')

            for col1, col2, corr_val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                parts.append(
                    f"""
        <tr>
            <td><code>{col1}</code></td>
            <td><code>{col2}</code></td>
            <td class="numeric-stat">{corr_val:.4f}</td>
        </tr>
"""
                )

            parts.append('</tbody></table>')
        else:
            parts.append('<div class="alert">No high correlations found.</div>')

        return "\n".join(parts)

    def _html_time_series(self, profile_result: ProfileResult) -> str:
        """Generate time series section."""
        parts = ['<h2>Time Series Information</h2>']

        if profile_result.date_column:
            parts.append(f'<p>Date column: <code>{profile_result.date_column}</code></p>')

        if profile_result.date_range:
            start, end = profile_result.date_range
            parts.append(f'<p>Date range: {start} to {end}</p>')

        if profile_result.date_gaps and len(profile_result.date_gaps) > 0:
            parts.append(f'<div class="alert warning">Detected {len(profile_result.date_gaps)} date gaps in the time series.</div>')

        if profile_result.entity_count:
            parts.append(f'<p>Number of entities: {profile_result.entity_count:,}</p>')

        return "\n".join(parts)

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <footer>
        Data Quality Report | Generated by QuantProjects Utils
    </footer>
</div>
</body>
</html>
"""

    def _generate_json(
        self,
        profile_result: ProfileResult,
        anomaly_result: Optional[AnomalyResult],
    ) -> str:
        """Generate JSON report."""
        report = {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "profile": profile_result.to_dict(),
        }

        if anomaly_result:
            report["anomalies"] = anomaly_result.to_dict()

        return json.dumps(report, indent=2, default=str)
