import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useStandards } from '../hooks';
import { formatDate } from '../lib';

interface StandardRow {
  standard_id: string;
  name: string;
  version: string;
  status: string;
  updated_at?: string | null;
}

export function StandardsPage() {
  const standards = useStandards();
  const columns: ColumnDef<StandardRow, unknown>[] = [
    { header: 'Standard ID', accessorKey: 'standard_id' },
    { header: 'Name', accessorKey: 'name' },
    { header: 'Version', accessorKey: 'version' },
    { header: 'Status', accessorKey: 'status' },
    {
      header: 'Updated',
      accessorKey: 'updated_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Standards"
        description="Platform standards / policies registry."
      />
      <ResourceListPage
        name="Standards"
        resource="standards"
        query={standards}
        requiredPermission="standards:read"
        tableState={<Table columns={columns} data={(standards.data as StandardRow[]) ?? []} />}
      />
    </div>
  );
}
