import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useFactors } from '../hooks';
import { formatDate } from '../lib';

interface FactorRow {
  factor_definition_id: string;
  name: string;
  family: string;
  generator: string;
  campaign_id: string | null;
  lifecycle: string;
  health: string;
  pipeline_stage: string;
  grade: string | null;
  rank_ic: number | null;
  icir: number | null;
  coverage: number | null;
  turnover: number | null;
  selected_treatment_id: string | null;
  cluster_id: string | null;
  libraries: string[];
  created_at: string;
  last_evaluated_at: string | null;
  evidence_status: string;
}

function fmtNum(v: number | null): string {
  return v === null ? '—' : v.toFixed(4);
}

export function FactorsPage() {
  const factors = useFactors();
  const columns: ColumnDef<FactorRow, unknown>[] = [
    { header: 'Factor ID', accessorKey: 'factor_definition_id' },
    { header: 'Name', accessorKey: 'name' },
    { header: 'Family', accessorKey: 'family' },
    { header: 'Generator', accessorKey: 'generator' },
    { header: 'Campaign', accessorKey: 'campaign_id' },
    { header: 'Lifecycle', accessorKey: 'lifecycle' },
    { header: 'Health', accessorKey: 'health' },
    { header: 'Pipeline', accessorKey: 'pipeline_stage' },
    { header: 'Grade', accessorKey: 'grade' },
    {
      header: 'RankIC',
      accessorKey: 'rank_ic',
      cell: (info) => fmtNum(info.getValue() as number | null),
    },
    {
      header: 'ICIR',
      accessorKey: 'icir',
      cell: (info) => fmtNum(info.getValue() as number | null),
    },
    {
      header: 'Coverage',
      accessorKey: 'coverage',
      cell: (info) => fmtNum(info.getValue() as number | null),
    },
    { header: 'Treatment', accessorKey: 'selected_treatment_id' },
    { header: 'Cluster', accessorKey: 'cluster_id' },
    {
      header: 'Libraries',
      accessorKey: 'libraries',
      cell: (info) => (info.getValue() as string[]).join(', ') || '—',
    },
    { header: 'Created', accessorKey: 'created_at' },
    {
      header: 'Last Evaluated',
      accessorKey: 'last_evaluated_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    {
      header: 'Evidence',
      accessorKey: 'evidence_status',
      cell: (info) => String(info.getValue() ?? '—'),
    },
  ];

  const frame = {
    data: factors.data as FactorRow[] | null | undefined,
    evidence: (factors.data as FactorRow[] | undefined)?.[0]?.evidence_status,
  };

  return (
    <div>
      <PageChrome
        title="Factors"
        description="Factor catalog (spec §27). Columns mirror the platform factor-summary response."
      />
      <ResourceListPage
        name="Factors"
        resource="factors"
        query={factors}
        frame={frame}
        tableState={<Table columns={columns} data={(factors.data as FactorRow[]) ?? []} />}
      />
    </div>
  );
}
