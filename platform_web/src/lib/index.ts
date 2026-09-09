/**
 * Lib: tiny pure helpers (state → copy + formatters).
 *
 * Everything here is PURE — no domain logic. It maps the six render states and
 * formats API values. The "empty" -> "no data yet" mapping (spec §49: never
 * render "no data" as 0) lives here and is unit-tested.
 */

/** One of the six render states (spec §49). */
export type QueryViewState =
  | 'loading'
  | 'empty'
  | 'permission-denied'
  | 'partial-evidence'
  | 'error'
  | 'stale';

/** Data + evidence metadata needed to classify the state of a list view. */
export interface DataFrame<T> {
  data: T[] | T | null | undefined;
  /** Evidence availability for the payload (spec §40). */
  evidence?: string;
  /** When the data was last refreshed (for the stale banner). */
  fetchedAt?: number | null;
}

/** True when a non-null payload exists. */
export function hasData(data: unknown): boolean {
  if (Array.isArray(data)) return data.length > 0;
  return data !== null && data !== undefined;
}

/** True when the payload is a non-null array (list views). */
export function isArray<T>(value: T[] | T | null | undefined): value is T[] {
  return Array.isArray(value);
}

/**
 * Classify the six states from the query's raw signal. A missing/empty payload
 * is "empty" (never a zero-looking table) UNLESS the API marked evidence as
 * not-yet-computed, which is "partial-evidence" (spec §40: LABEL_NOT_MATURE is
 * never zero-filled).
 */
export function classifyState(
  isLoading: boolean,
  error: unknown,
  frame: DataFrame<unknown>,
  isFetching: boolean,
): QueryViewState {
  if (isLoading && !hasData(frame.data)) return 'loading';
  if (error) {
    if (error instanceof Error && (error as { code?: string }).code === 'PERMISSION_DENIED') {
      return 'permission-denied';
    }
    return 'error';
  }
  if (frame.evidence === 'LABEL_NOT_MATURE' || frame.evidence === 'NOT_COMPUTED') {
    return 'partial-evidence';
  }
  if (!hasData(frame.data)) return 'empty';
  if (isFetching) return 'stale';
  return 'loading';
}

/** "empty" must never be rendered as 0 — this is the copy for empty views. */
export function emptyCopy(resource: string): string {
  return `No ${resource} data yet.`;
}

/** "partial evidence" copy — evidence is pending, never zero-filled. */
export function partialEvidenceCopy(resource: string, evidence: string): string {
  return `${resource} are not fully evaluated yet — evidence is ${evidence}.`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toISOString();
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return String(value);
}
