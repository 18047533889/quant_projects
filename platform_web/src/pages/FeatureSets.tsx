import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useFeatureSets } from '../hooks';
import { formatDate } from '../lib';
import { featureSetViewRow } from '../api/viewRows';

interface FeatureSetRow {
  feature_set_id: string;
  version: string;
  consumer_profile?: string | null;
  source_library_versions?: string[];
  member_count: number;
  label_definition_ref?: string | null;
  schema_hash?: string;
  semantic_hash?: string;
  created_at?: string | null;
}

export function FeatureSetsPage() {
  const featureSets = useFeatureSets();
  const columns: ColumnDef<FeatureSetRow, unknown>[] = [
    { header: 'Feature Set ID', accessorKey: 'feature_set_id' },
    { header: 'Version', accessorKey: 'version' },
    { header: 'Consumer', accessorKey: 'consumer_profile' },
    { header: 'Members', accessorKey: 'member_count' },
    { header: 'Label', accessorKey: 'label_definition_ref' },
    { header: 'Schema Hash', accessorKey: 'schema_hash' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Feature Sets"
        description="Versioned feature sets with typed, ordered members (spec §17/§18)."
      />
      <ResourceListPage
        name="Feature sets"
        resource="feature sets"
        query={featureSets}
        tableState={<Table columns={columns} data={(featureSets.data ?? []).map(featureSetViewRow)} />}
      />
    </div>
  );
}
