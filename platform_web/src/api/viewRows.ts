/** Pure projections from the declared API DTOs to table display fields. */
import type { RegistryRow } from './client';

export function registryViewRow<T extends object>({ entity, ...metadata }: RegistryRow<T>) {
  // Entity fields are authoritative; retain envelope ref/evidence alongside them.
  return { ...metadata, ...entity };
}

export function backtestViewRow<T extends { artifact: {
  artifact_type: string; artifact_id: string; content_hash: string; size_bytes: number;
} }>(row: T) {
  return { ...row.artifact, ...row };
}

export function featureSetViewRow<T extends { ordered_members: readonly unknown[] }>(row: T) {
  return { ...row, member_count: row.ordered_members.length };
}

export function libraryViewRow<T extends { members: readonly unknown[] }>(row: T) {
  return { ...row, members_count: row.members.length };
}

export function modelViewRow<T extends {
  label_definition: { label_name: string }; data_snapshot: { snapshot_id: string };
}>(row: T) {
  return { ...row, label: row.label_definition.label_name, data_snapshot_id: row.data_snapshot.snapshot_id };
}
