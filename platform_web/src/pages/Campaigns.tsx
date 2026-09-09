import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useCampaigns } from '../hooks';
import { formatDate } from '../lib';
import { registryViewRow } from '../api/viewRows';

interface CampaignRow {
  ref: string;
  campaign_id: string;
  name: string;
  status: string;
  created_at?: string | null;
}

export function CampaignsPage() {
  const campaigns = useCampaigns();
  const columns: ColumnDef<CampaignRow, unknown>[] = [
    { header: 'Campaign ID', accessorKey: 'campaign_id' },
    { header: 'Name', accessorKey: 'name' },
    { header: 'Status', accessorKey: 'status' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome title="Campaigns" description="Factor-generation campaigns." />
      <ResourceListPage
        name="Campaigns"
        resource="campaigns"
        query={campaigns}
        tableState={<Table columns={columns} data={(campaigns.data ?? []).map(registryViewRow)} />}
      />
    </div>
  );
}
