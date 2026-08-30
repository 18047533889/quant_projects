import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useDataSnapshots } from '../hooks';
import { formatDate } from '../lib';

interface DataSnapshotRow {
  snapshot_id: string;
  universe_id?: string;
  start_time?: string | null;
  end_time?: string | null;
  status: string;
  created_at?: string | null;
}

export function DataSnapshotsPage() {
  const snapshots = useDataSnapshots();
  const columns: ColumnDef<DataSnapshotRow, unknown>[] = [
    { header: 'Snapshot ID', accessorKey: 'snapshot_id' },
    { header: 'Universe', accessorKey: 'universe_id' },
    {
      header: 'Start',
      accessorKey: 'start_time',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    {
      header: 'End',
      accessorKey: 'end_time',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
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
        title="Data Snapshots"
        description="DataAccess snapshots (spec §38)."
      />
      <ResourceListPage
        name="Data snapshots"
        resource="data snapshots"
        query={snapshots}
        tableState={<Table columns={columns} data={(snapshots.data as DataSnapshotRow[]) ?? []} />}
      />
    </div>
  );
}
