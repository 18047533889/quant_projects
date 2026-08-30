import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useAlerts } from '../hooks';
import { formatDate } from '../lib';

interface AlertRow {
  alert_id: string;
  severity: string;
  message: string;
  factor_id?: string | null;
  job_id?: string | null;
  created_at: string;
}

export function AlertsPage() {
  const alerts = useAlerts();
  const columns: ColumnDef<AlertRow, unknown>[] = [
    { header: 'Alert ID', accessorKey: 'alert_id' },
    { header: 'Severity', accessorKey: 'severity' },
    { header: 'Message', accessorKey: 'message' },
    { header: 'Factor', accessorKey: 'factor_id' },
    { header: 'Job', accessorKey: 'job_id' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome title="Alerts" description="Health alerts across the platform." />
      <ResourceListPage
        name="Alerts"
        resource="alerts"
        query={alerts}
        tableState={<Table columns={columns} data={(alerts.data as AlertRow[]) ?? []} />}
      />
    </div>
  );
}
