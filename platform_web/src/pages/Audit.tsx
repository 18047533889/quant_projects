import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useAudit } from '../hooks';
import { formatDate } from '../lib';

interface AuditLogRow {
  audit_id: string;
  actor_principal_id: string;
  action: string;
  resource: string;
  result: string;
  timestamp: string;
  request_id?: string | null;
}

export function AuditPage() {
  const audit = useAudit();
  const columns: ColumnDef<AuditLogRow, unknown>[] = [
    { header: 'Audit ID', accessorKey: 'audit_id' },
    { header: 'Actor', accessorKey: 'actor_principal_id' },
    { header: 'Action', accessorKey: 'action' },
    { header: 'Resource', accessorKey: 'resource' },
    { header: 'Result', accessorKey: 'result' },
    {
      header: 'Timestamp',
      accessorKey: 'timestamp',
      cell: (info) => formatDate(info.getValue() as string | null),
    },
    { header: 'Request ID', accessorKey: 'request_id' },
  ];

  return (
    <div>
      <PageChrome
        title="Audit Log"
        description="Audit trail of access decisions and governance actions (spec §23/§24.4)."
      />
      <ResourceListPage
        name="Audit log"
        resource="audit log entries"
        query={audit}
        requiredPermission="audit:read"
        tableState={<Table columns={columns} data={(audit.data as AuditLogRow[]) ?? []} />}
      />
    </div>
  );
}
