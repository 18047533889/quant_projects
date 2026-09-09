import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useClusters } from '../hooks';
import { formatDate } from '../lib';
import { registryViewRow } from '../api/viewRows';

interface ClusterRow {
  ref: string;
  cluster_set_version_id: string;
  algorithm: string;
  backend?: string;
  seed?: number;
  resolution?: number | null;
  policy_hash?: string;
  created_at?: string | null;
}

export function ClustersPage() {
  const clusters = useClusters();
  const columns: ColumnDef<ClusterRow, unknown>[] = [
    { header: 'Cluster Set Version', accessorKey: 'cluster_set_version_id' },
    { header: 'Algorithm', accessorKey: 'algorithm' },
    { header: 'Backend', accessorKey: 'backend' },
    { header: 'Seed', accessorKey: 'seed' },
    { header: 'Resolution', accessorKey: 'resolution' },
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
        title="Clusters"
        description="Global clustering runs (ClusterSetVersion) and the layered cluster model (spec §14/§34/§35)."
      />
      <ResourceListPage
        name="Clusters"
        resource="clusters"
        query={clusters}
        tableState={<Table columns={columns} data={(clusters.data ?? []).map(registryViewRow)} />}
      />
    </div>
  );
}
