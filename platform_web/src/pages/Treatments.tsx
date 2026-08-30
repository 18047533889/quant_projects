import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useTreatments } from '../hooks';
import { formatDate } from '../lib';

interface TreatmentRow {
  ref: string;
  treatment_id: string;
  factor_definition_id: string;
  policy_hash: string;
  created_at?: string | null;
}

export function TreatmentsPage() {
  const treatments = useTreatments();
  const columns: ColumnDef<TreatmentRow, unknown>[] = [
    { header: 'Treatment ID', accessorKey: 'treatment_id' },
    { header: 'Factor', accessorKey: 'factor_definition_id' },
    { header: 'Policy Hash', accessorKey: 'policy_hash' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Treatments"
        description="Treatment search results (spec §8.4). TreatmentRef digests are carried from factor_assets."
      />
      <ResourceListPage
        name="Treatments"
        resource="treatments"
        query={treatments}
        tableState={<Table columns={columns} data={(treatments.data as TreatmentRow[]) ?? []} />}
      />
    </div>
  );
}
