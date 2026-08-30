import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useBacktests } from '../hooks';
import { formatDate } from '../lib';

interface BacktestRow {
  backtest_id: string;
  strategy_ref: string;
  artifact_type: string;
  artifact_id: string;
  content_hash: string;
  size_bytes: number;
  result_summary_uri: string;
  qe_reevaluation_status?: string;
  created_at?: string | null;
}

export function BacktestsPage() {
  const backtests = useBacktests();
  const columns: ColumnDef<BacktestRow, unknown>[] = [
    { header: 'Backtest ID', accessorKey: 'backtest_id' },
    { header: 'Strategy', accessorKey: 'strategy_ref' },
    { header: 'Artifact Type', accessorKey: 'artifact_type' },
    { header: 'Artifact ID', accessorKey: 'artifact_id' },
    { header: 'Content Hash', accessorKey: 'content_hash' },
    {
      header: 'Size',
      accessorKey: 'size_bytes',
      cell: (info) => fmtBytes(info.getValue() as number),
    },
    { header: 'Summary URI', accessorKey: 'result_summary_uri' },
    { header: 'QE Reeval', accessorKey: 'qe_reevaluation_status' },
    {
      header: 'Created',
      accessorKey: 'created_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
  ];

  return (
    <div>
      <PageChrome
        title="Backtests"
        description="Formal execution backtests (contract + future adapter, spec §41)."
      />
      <ResourceListPage
        name="Backtests"
        resource="backtests"
        query={backtests}
        tableState={<Table columns={columns} data={(backtests.data as BacktestRow[]) ?? []} />}
      />
    </div>
  );
}

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MiB`;
}
