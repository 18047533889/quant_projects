/**
 * Overview page — §26 Dashboard (system operational posture, NOT a report
 * catalog). Renders KPIs / funnel / operational signals from the platform API.
 * Per §49 the loading / empty / permission-denied / partial-evidence / error /
 * stale states are handled by <StatePanel>.
 */
import { StatePanel } from '../components/StatePanel';
import { DashboardSections } from '../components/KpiCard';
import { PageChrome } from '../components/PageChrome';
import { useDashboard } from '../hooks';
import type { QueryViewState } from '../lib';
import { classifyState } from '../lib';

export function OverviewPage() {
  const dashboard = useDashboard();

  const frame = { data: dashboard.data, evidence: undefined };
  const state: QueryViewState = classifyState(
    dashboard.isLoading,
    dashboard.error,
    frame,
    dashboard.isFetching,
  );

  return (
    <div>
      <PageChrome
        title="Dashboard"
        description="System operational posture — KPIs, pipeline funnel, operational signals (spec §26)."
      />

      {state === 'loading' ? <StatePanel state="loading" resource="dashboard" /> : null}
      {state === 'empty' ? <StatePanel state="empty" resource="dashboard data" /> : null}
      {state === 'permission-denied' ? (
        <StatePanel
          state="permission-denied"
          resource="dashboard"
          requiredPermission="factor:read_summary"
        />
      ) : null}
      {state === 'partial-evidence' ? (
        <StatePanel state="partial-evidence" resource="dashboard metrics" evidence="NOT_COMPUTED" />
      ) : null}
      {state === 'error' ? (
        <StatePanel state="error" resource="dashboard" error={dashboard.error} />
      ) : null}
      {state === 'stale' ? <StatePanel state="stale" resource="dashboard" /> : null}

      {dashboard.data != null ? <DashboardSections summary={dashboard.data} /> : null}
    </div>
  );
}
