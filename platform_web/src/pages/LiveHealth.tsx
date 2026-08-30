import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useLiveHealth } from '../hooks';
import { formatDate } from '../lib';

interface LiveHealthRow {
  ref: string;
  factor_definition_id: string;
  name: string;
  lifecycle: string;
  health: string;
  pipeline_stage: string;
  last_evaluated_at?: string | null;
  evidence_status?: string;
}

export function LiveHealthPage() {
  const liveHealth = useLiveHealth();
  const columns: ColumnDef<LiveHealthRow, unknown>[] = [
    { header: 'Factor', accessorKey: 'name' },
    { header: 'Factor ID', accessorKey: 'factor_definition_id' },
    { header: 'Lifecycle', accessorKey: 'lifecycle' },
    { header: 'Health', accessorKey: 'health' },
    { header: 'Pipeline', accessorKey: 'pipeline_stage' },
    {
      header: 'Last Evaluated',
      accessorKey: 'last_evaluated_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    { header: 'Evidence', accessorKey: 'evidence_status' },
  ];

  return (
    <div>
      <PageChrome
        title="Live Health"
        description="Operational health of production factor assets (spec §9 HealthState)."
      />
      <ResourceListPage
        name="Live health"
        resource="live-health factors"
        query={liveHealth}
        tableState={<Table columns={columns} data={(liveHealth.data as LiveHealthRow[]) ?? []} />}
      />
    </div>
  );
}
