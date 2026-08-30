import type { ReactNode } from 'react';

import type { DataFrame, QueryViewState } from '../lib';
import { classifyState, isArray } from '../lib';
import { StatePanel } from './StatePanel';

/**
 * ResourceListPage — shared scaffold for list pages.
 *
 * Drives a TanStack Query result through the six-state component and renders
 * `<tableState>` only when there is data. `name` is the resource noun used in
 * empty/partial copy ("factors", "jobs", …). `frame.evidence` lets the API tell
 * the page that evidence is not mature yet (spec §40).
 */
export interface ResourceListPageProps {
  name: string;
  /** The noun used in "No {name} data yet." copy. */
  resource: string;
  query: { isLoading: boolean; isFetching: boolean; error: unknown; data: unknown };
  frame?: DataFrame<unknown>;
  requiredPermission?: string;
  tableState: ReactNode;
}

export function ResourceListPage({
  name,
  resource,
  query,
  frame,
  requiredPermission,
  tableState,
}: ResourceListPageProps) {
  const effFrame: DataFrame<unknown> = frame ?? { data: query.data, evidence: undefined };
  const state: QueryViewState = classifyState(
    query.isLoading,
    query.error,
    effFrame,
    query.isFetching,
  );

  if (state === 'loading' || state === 'empty' || state === 'permission-denied' || state === 'error') {
    return (
      <section aria-label={name}>
        <StatePanel
          state={state}
          resource={resource}
          error={query.error}
          requiredPermission={requiredPermission}
        />
      </section>
    );
  }

  if (state === 'partial-evidence') {
    return (
      <section aria-label={name}>
        <StatePanel state="partial-evidence" resource={resource} evidence={effFrame.evidence} />
        {isArray(effFrame.data) ? tableState : null}
      </section>
    );
  }

  // Data present; stale banner above the table when a background refetch is in flight.
  return (
    <section aria-label={name}>
      {state === 'stale' ? <StatePanel state="stale" resource={resource} /> : null}
      {tableState}
    </section>
  );
}
