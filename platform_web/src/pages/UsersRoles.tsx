import type { ColumnDef } from '@tanstack/react-table';

import { PageChrome } from '../components/PageChrome';
import { ResourceListPage } from '../components/ResourceListPage';
import { Table } from '../components/Table';
import { useUsers } from '../hooks';

interface UserRow {
  principal_id: string;
  display_name: string;
  team: string;
  role: string;
  classification: string;
  principal_type: string;
}

export function UsersRolesPage() {
  const users = useUsers();
  const columns: ColumnDef<UserRow, unknown>[] = [
    { header: 'Principal ID', accessorKey: 'principal_id' },
    { header: 'Display Name', accessorKey: 'display_name' },
    { header: 'Team', accessorKey: 'team' },
    { header: 'Role', accessorKey: 'role' },
    { header: 'Classification', accessorKey: 'classification' },
    { header: 'Type', accessorKey: 'principal_type' },
  ];

  return (
    <div>
      <PageChrome
        title="Users & Roles"
        description="Principals, teams, roles and security classification (spec §24.1)."
      />
      <ResourceListPage
        name="Users"
        resource="users"
        query={users}
        requiredPermission="user:manage"
        tableState={<Table columns={columns} data={(users.data as UserRow[]) ?? []} />}
      />
    </div>
  );
}
