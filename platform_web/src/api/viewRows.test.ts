import { describe, expect, it } from 'vitest';
import { backtestViewRow, featureSetViewRow, libraryViewRow, modelViewRow, registryViewRow } from './viewRows';

describe('API table projections', () => {
  it('unwraps entity while retaining envelope identity and evidence', () => {
    const row = { ref: 'job-1', health: 'outer', evidence_status: 'pending',
      entity: { status: 'RUNNING', health: 'domain', created_at: null } };
    expect(registryViewRow(row)).toEqual({ ref: 'job-1', evidence_status: 'pending',
      status: 'RUNNING', health: 'domain', created_at: null });
    expect(row.entity.status).toBe('RUNNING');
  });
  it('projects nested artifact without replacing backtest creation time', () => {
    const row = { backtest_id: 'bt', created_at: 'backtest-time', artifact: {
      artifact_type: 'REPORT', artifact_id: 'a', content_hash: 'hash', size_bytes: 12,
      created_at: 'artifact-time' } };
    expect(backtestViewRow(row)).toMatchObject({ backtest_id: 'bt', artifact_id: 'a',
      size_bytes: 12, created_at: 'backtest-time' });
  });
  it('counts the declared member arrays, including legitimate empty lists', () => {
    expect(featureSetViewRow({ ordered_members: ['a', 'b'] }).member_count).toBe(2);
    expect(libraryViewRow({ members: [] }).members_count).toBe(0);
  });
  it('uses carried model label and snapshot fields', () => {
    expect(modelViewRow({ model_id: 'm', label_definition: { label_name: 'return' },
      data_snapshot: { snapshot_id: 's' } })).toMatchObject({ model_id: 'm', label: 'return', data_snapshot_id: 's' });
  });
});
