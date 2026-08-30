import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useModels } from '../hooks';
import { formatDate } from '../lib';

interface ModelRow {
  model_id: string;
  model_version: string;
  model_architecture: string;
  feature_set_version_ref: string;
  label: string;
  data_snapshot_id: string;
  eval_metrics_uri?: string;
  created_at?: string | null;
}

export function ModelsPage() {
  const models = useModels();
  const columns: ColumnDef<ModelRow, unknown>[] = [
    { header: 'Model ID', accessorKey: 'model_id' },
    { header: 'Version', accessorKey: 'model_version' },
    { header: 'Architecture', accessorKey: 'model_architecture' },
    { header: 'Feature Set', accessorKey: 'feature_set_version_ref' },
    { header: 'Label', accessorKey: 'label' },
    { header: 'Data Snapshot', accessorKey: 'data_snapshot_id' },
    { header: 'Eval Metrics', accessorKey: 'eval_metrics_uri' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Models"
        description="Model registry (spec §2). Models are read-only / placeholder until the formal registry is wired."
      />
      <ResourceListPage
        name="Models"
        resource="models"
        query={models}
        tableState={<Table columns={columns} data={(models.data as ModelRow[]) ?? []} />}
      />
    </div>
  );
}
