import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useStreaming } from '../hooks';
import { formatDate } from '../lib';

interface StreamingRow {
  stream_name: string;
  status: string;
  last_updated_at?: string | null;
}

export function StreamingPage() {
  const streaming = useStreaming();
  const columns: ColumnDef<StreamingRow, unknown>[] = [
    { header: 'Stream', accessorKey: 'stream_name' },
    { header: 'Status', accessorKey: 'status' },
    {
      header: 'Last Updated',
      accessorKey: 'last_updated_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Streaming"
        description="Incremental updates — signal-available clock vs label-matured clock (spec §40)."
      />
      <ResourceListPage
        name="Streams"
        resource="streams"
        query={streaming}
        tableState={<Table columns={columns} data={(streaming.data as StreamingRow[]) ?? []} />}
      />
    </div>
  );
}
