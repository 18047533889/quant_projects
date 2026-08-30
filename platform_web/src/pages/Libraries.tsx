import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useLibraries } from '../hooks';
import { formatDate } from '../lib';

interface LibraryRow {
  library_version_id: string;
  logical_library_id: string;
  cluster_set_version_id: string;
  members_count: number;
  policy_hash?: string;
  evidence_snapshot?: string;
  status: string;
  created_at?: string | null;
}

export function LibrariesPage() {
  const libraries = useLibraries();
  const columns: ColumnDef<LibraryRow, unknown>[] = [
    { header: 'Library Version', accessorKey: 'library_version_id' },
    { header: 'Logical Library', accessorKey: 'logical_library_id' },
    { header: 'Cluster Set Version', accessorKey: 'cluster_set_version_id' },
    { header: 'Members', accessorKey: 'members_count' },
    { header: 'Status', accessorKey: 'status' },
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
        title="Libraries"
        description="FactorLibraryVersion — governance asset sets with promotion/rollback (spec §15/§16)."
      />
      <ResourceListPage
        name="Libraries"
        resource="libraries"
        query={libraries}
        tableState={<Table columns={columns} data={(libraries.data as LibraryRow[]) ?? []} />}
      />
    </div>
  );
}
