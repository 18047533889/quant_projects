import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useOptimizers } from '../hooks';
import { formatDate } from '../lib';

interface OptimizerRow {
  optimizer_id: string;
  search_policy_ref: string;
  status: string;
  created_at?: string | null;
}

export function OptimizersPage() {
  const optimizers = useOptimizers();
  const columns: ColumnDef<OptimizerRow, unknown>[] = [
    { header: 'Optimizer ID', accessorKey: 'optimizer_id' },
    { header: 'Search Policy', accessorKey: 'search_policy_ref' },
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
        title="Optimizers"
        description="Treatment-search optimizers (spec §35)."
      />
      <ResourceListPage
        name="Optimizers"
        resource="optimizers"
        query={optimizers}
        tableState={<Table columns={columns} data={(optimizers.data as OptimizerRow[]) ?? []} />}
      />
    </div>
  );
}
