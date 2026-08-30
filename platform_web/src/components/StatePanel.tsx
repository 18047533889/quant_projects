import type { ReactNode } from 'react';

import type { QueryViewState } from '../lib';

/**
 * StatePanel — renders the SIX states required by spec §49.
 *
 *   loading           — spinner while the query fetches
 *   empty             — "no … data yet" (NEVER a zero-filled table)
 *   permission-denied — 403 surface: tell the user what permission they lack
 *   partial-evidence  — evidence pending (LABEL_NOT_MATURE / NOT_COMPUTED),
 *                       never zero-filled
 *   error             — failed request with the canonical error envelope
 *   stale             — background refetch with data already on screen
 *
 * `children` are rendered for the OK states; a `tableState`-driven page uses
 * <StatePanel> only for the non-ok six states.
 */

export interface StatePanelProps {
  state: QueryViewState;
  /** Resource noun for the empty/partial copy, e.g. "factors". */
  resource: string;
  /** The evidence status from the API, when the state is partial-evidence. */
  evidence?: string;
  /** Error to show when the state is error. */
  error?: unknown;
  /** Permission name the user lacks, for the permission-denied state. */
  requiredPermission?: string;
  /** Freshness label for the stale banner. */
  staleLabel?: string;
  /** Rendered for the "empty" state (defaults to a "no data yet" message). */
  emptyMessage?: ReactNode;
}

export function StatePanel({
  state,
  resource,
  evidence,
  error,
  requiredPermission,
  staleLabel,
  emptyMessage,
}: StatePanelProps) {
  switch (state) {
    case 'loading':
      return (
        <div className="state-panel state-panel--loading" data-state="loading" role="status">
          Loading {resource}…
        </div>
      );
    case 'empty':
      return (
        <div className="state-panel state-panel--empty" data-state="empty">
          {emptyMessage ?? `No ${resource} data yet.`}
        </div>
      );
    case 'permission-denied':
      return (
        <div className="state-panel state-panel--denied" data-state="permission-denied" role="alert">
          <strong>Permission denied.</strong>{' '}
          {requiredPermission ? (
            <>
              This view requires the <code>{requiredPermission}</code> permission.
            </>
          ) : (
            <>You do not have permission to view {resource}.</>
          )}
        </div>
      );
    case 'partial-evidence':
      return (
        <div className="state-panel state-panel--partial" data-state="partial-evidence">
          {resource} are not fully evaluated yet — evidence is {evidence ?? 'pending'}.
          Values are not zero-filled until labels mature.
        </div>
      );
    case 'error':
      return (
        <div className="state-panel state-panel--error" data-state="error" role="alert">
          Failed to load {resource}.{' '}
          {error instanceof Error ? error.message : String(error ?? 'Unknown error')}
        </div>
      );
    case 'stale':
      return (
        <div className="state-panel state-panel--stale" data-state="stale">
          Showing previously loaded {resource} while refreshing
          {staleLabel ? ` (last updated ${staleLabel})` : ''}…
        </div>
      );
  }
}
