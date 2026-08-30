import type { ReactNode } from 'react';

import type { DashboardKpis, DashboardSummary, OperationalSignals, PipelineFunnel } from '../api/types';

/**
 * KpiCard — one Dashboard KPI (spec §26). The KPI VALUE IS A NUMBER THAT MAY BE
 * 0 (e.g. zero factors in production). Only a MISSING value ("—") means "no
 * data". "No data" is never rendered as 0 (spec §49).
 */
export interface KpiCardProps {
  label: string;
  value: number | string | null | undefined;
  hint?: string;
}

export function KpiCard({ label, value, hint }: KpiCardProps) {
  const has = value !== null && value !== undefined;
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{has ? String(value) : '—'}</div>
      {hint ? <div className="kpi-hint">{hint}</div> : null}
    </div>
  );
}

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : String(v);
}

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;
}

function renderKpis(kpis: DashboardKpis): ReactNode {
  return (
    <div className="kpi-grid" data-testid="kpi-grid">
      <KpiCard label="Total Factors" value={fmt(kpis.total_factors)} />
      <KpiCard label="New Today" value={fmt(kpis.new_today)} />
      <KpiCard label="New 7D" value={fmt(kpis.new_7d)} />
      <KpiCard label="Processing" value={fmt(kpis.processing)} />
      <KpiCard label="Validated" value={fmt(kpis.validated)} />
      <KpiCard label="Shadow" value={fmt(kpis.shadow)} />
      <KpiCard label="Production Eligible" value={fmt(kpis.production_eligible)} />
      <KpiCard label="Production" value={fmt(kpis.production)} />
      <KpiCard label="Degraded" value={fmt(kpis.degraded)} />
      <KpiCard label="Quarantined" value={fmt(kpis.quarantined)} />
    </div>
  );
}

function renderFunnel(funnel: PipelineFunnel): ReactNode {
  const rows: Array<{ label: string; value: number | null | undefined }> = [
    { label: 'Generated', value: funnel.generated },
    { label: 'Valid', value: funnel.valid },
    { label: 'Compiled', value: funnel.compiled },
    { label: 'Materialized', value: funnel.materialized },
    { label: 'QE Passed', value: funnel.qe_passed },
    { label: 'Novel', value: funnel.novel },
    { label: 'Clustered', value: funnel.clustered },
    { label: 'Library Candidate', value: funnel.library_candidate },
    { label: 'Promoted', value: funnel.promoted },
  ];
  return (
    <div className="funnel" data-testid="pipeline-funnel">
      {rows.map((r) => (
        <div className="funnel-row" key={r.label}>
          <span className="funnel-label">{r.label}</span>
          <span className="funnel-value">{fmt(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

function renderSignals(signals: OperationalSignals): ReactNode {
  const rows: Array<{ label: string; value: string }> = [
    { label: 'Job backlog', value: fmt(signals.job_backlog) },
    { label: 'Failure rate', value: pct(signals.failure_rate) },
    { label: 'Factor throughput', value: fmt(signals.factor_throughput) },
    { label: 'COS ingest lag', value: `${fmt(signals.cos_ingest_lag_seconds)} s` },
    { label: 'Cluster drift', value: fmt(signals.cluster_drift) },
    { label: 'Library version changes', value: fmt(signals.library_version_changes) },
    { label: 'Recent promotions', value: fmt(signals.recent_promotions) },
    { label: 'Health alerts', value: fmt(signals.health_alerts) },
  ];
  return (
    <div className="signals" data-testid="operational-signals">
      {rows.map((r) => (
        <div className="signal-row" key={r.label}>
          <span className="signal-label">{r.label}</span>
          <span className="signal-value">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/** §26 Dashboard: system operational posture — NOT a report catalog. */
export function DashboardSections({ summary }: { summary: DashboardSummary }) {
  return (
    <div className="dashboard-sections">
      <section aria-label="KPIs">
        <h2>System KPIs</h2>
        {renderKpis(summary.kpis)}
      </section>
      <section aria-label="Pipeline funnel">
        <h2>Pipeline Funnel</h2>
        {renderFunnel(summary.funnel)}
      </section>
      <section aria-label="Operational signals">
        <h2>Operational Signals</h2>
        {renderSignals(summary.signals)}
      </section>
    </div>
  );
}
