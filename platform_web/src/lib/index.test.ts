import { describe, expect, it } from 'vitest';

import {
  classifyState,
  emptyCopy,
  formatBytes,
  formatDate,
  formatCount,
  hasData,
  partialEvidenceCopy,
} from './index';

describe('hasData', () => {
  it('empty array is not data (→ empty state)', () => {
    expect(hasData([])).toBe(false);
    expect(hasData([1])).toBe(true);
    expect(hasData(null)).toBe(false);
    expect(hasData(undefined)).toBe(false);
    expect(hasData({})).toBe(true);
  });
});

describe('classifyState — six states (spec §49)', () => {
  it('loading when fetching and nothing rendered yet', () => {
    expect(classifyState(true, null, { data: undefined }, false)).toBe('loading');
  });

  it('empty when there is no data (never a zero table)', () => {
    expect(classifyState(false, null, { data: [] }, false)).toBe('empty');
    expect(classifyState(false, null, { data: null }, false)).toBe('empty');
  });

  it('permission-denied surfaces the 403', () => {
    const err = new Error('nope');
    (err as { code?: string }).code = 'PERMISSION_DENIED';
    expect(classifyState(false, err, { data: [] }, false)).toBe('permission-denied');
  });

  it('partial-evidence when the API says the label is not mature', () => {
    expect(
      classifyState(false, null, { data: [], evidence: 'LABEL_NOT_MATURE' }, false),
    ).toBe('partial-evidence');
    expect(classifyState(false, null, { data: [], evidence: 'NOT_COMPUTED' }, false)).toBe(
      'partial-evidence',
    );
  });

  it('error when the request fails', () => {
    expect(classifyState(false, new Error('boom'), { data: undefined }, false)).toBe('error');
  });

  it('stale when refetching with data already on screen', () => {
    expect(classifyState(true, null, { data: [1] }, true)).toBe('stale');
  });
});

describe('empty/partial copy (spec §49: never 0)', () => {
  it('says "no … data yet"', () => {
    expect(emptyCopy('factors')).toBe('No factors data yet.');
  });

  it('labels not mature → partial evidence copy', () => {
    expect(partialEvidenceCopy('Factors', 'LABEL_NOT_MATURE')).toContain('LABEL_NOT_MATURE');
  });
});

describe('formatters', () => {
  it('formatDate renders ISO or dash', () => {
    expect(formatDate('2026-08-26T00:00:00Z')).toMatch(/^2026-08-26T00:00:00/);
    expect(formatDate(null)).toBe('—');
    expect(formatDate('garbage')).toBe('—');
  });

  it('formatBytes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KiB');
    expect(formatBytes(null)).toBe('—');
  });

  it('formatCount never renders 0 for null', () => {
    expect(formatCount(3)).toBe('3');
    expect(formatCount(null)).toBe('—');
    expect(formatCount(0)).toBe('0');
  });
});
