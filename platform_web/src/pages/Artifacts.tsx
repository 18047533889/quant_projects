import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useArtifacts } from '../hooks';
import { formatDate } from '../lib';

interface ArtifactRow {
  artifact_id: string;
  artifact_type: string;
  schema_version: string;
  content_hash: string;
  storage_uri: string;
  size_bytes: number;
  created_at: string;
  producer_type: string;
  producer_version: string;
  security_classification?: string | null;
}

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MiB`;
}

export function ArtifactsPage() {
  const artifacts = useArtifacts();
  const columns: ColumnDef<ArtifactRow, unknown>[] = [
    { header: 'Artifact ID', accessorKey: 'artifact_id' },
    { header: 'Type', accessorKey: 'artifact_type' },
    { header: 'Schema', accessorKey: 'schema_version' },
    { header: 'Producer', accessorKey: 'producer_type' },
    { header: 'Producer Version', accessorKey: 'producer_version' },
    {
      header: 'Size',
      accessorKey: 'size_bytes',
      cell: (info) => fmtBytes(info.getValue() as number),
    },
    { header: 'Classification', accessorKey: 'security_classification' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Artifacts"
        description="Published artifact registry — immutable refs with content_hash (spec §7.2)."
      />
      <ResourceListPage
        name="Artifacts"
        resource="artifacts"
        query={artifacts}
        tableState={<Table columns={columns} data={(artifacts.data as ArtifactRow[]) ?? []} />}
      />
    </div>
  );
}
