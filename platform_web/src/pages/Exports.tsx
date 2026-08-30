import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useExports } from '../hooks';
import { formatDate } from '../lib';

interface ExportRow {
  export_id: string;
  artifact_type: string;
  status: string;
  created_at?: string | null;
}

export function ExportsPage() {
  const exportsList = useExports();
  const columns: ColumnDef<ExportRow, unknown>[] = [
    { header: 'Export ID', accessorKey: 'export_id' },
    { header: 'Artifact Type', accessorKey: 'artifact_type' },
    { header: 'Status', accessorKey: 'status' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Exports"
        description="Report exports (spec §48)."
      />
      <ResourceListPage
        name="Exports"
        resource="exports"
        query={exportsList}
        tableState={<Table columns={columns} data={(exportsList.data as ExportRow[]) ?? []} />}
      />
    </div>
  );
}
