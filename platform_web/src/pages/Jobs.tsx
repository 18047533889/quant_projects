import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useJobs } from '../hooks';
import { formatDate } from '../lib';

interface JobRow {
  ref: string;
  status: string;
  current_stage?: string | null;
  progress?: number;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  attempt_count?: number;
  error_class?: string | null;
}

function fmtProgress(p: number | undefined): string {
  if (p === undefined || p === null) return '—';
  return `${Math.round(p * 100)}%`;
}

export function JobsPage() {
  const jobs = useJobs();
  const columns: ColumnDef<JobRow, unknown>[] = [
    { header: 'Job', accessorKey: 'ref' },
    { header: 'Status', accessorKey: 'status' },
    { header: 'Stage', accessorKey: 'current_stage' },
    {
      header: 'Progress',
      accessorKey: 'progress',
      cell: (info) => fmtProgress(info.getValue() as number | undefined),
    },
    {
      header: 'Started',
      accessorKey: 'started_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    {
      header: 'Finished',
      accessorKey: 'finished_at',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    { header: 'Attempts', accessorKey: 'attempt_count' },
    { header: 'Error Class', accessorKey: 'error_class' },
  ];

  const frame = { data: jobs.data as JobRow[] | null | undefined, evidence: undefined };

  return (
    <div>
      <PageChrome
        title="Jobs"
        description="JobSpec / JobRecord / JobResult (spec §9, §42, §43)."
      />
      <ResourceListPage
        name="Jobs"
        resource="jobs"
        query={jobs}
        frame={frame}
        requiredPermission="job:read"
        tableState={<Table columns={columns} data={(jobs.data as JobRow[]) ?? []} />}
      />
    </div>
  );
}
