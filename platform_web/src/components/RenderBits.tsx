import type { ReactNode } from 'react';

import type { ArtifactRef } from '../api/types';

/** Shared small render helpers: artifact + evidence cells. */

export function ArtifactTypeBadge({ type }: { type: string }) {
  return <span className={`badge badge--artifact badge--${type.toLowerCase()}`}>{type}</span>;
}

export function ArtifactRow({ artifact }: { artifact: ArtifactRef }) {
  return (
    <div className="artifact-cell">
      <div className="artifact-id">
        <code>{artifact.artifact_id}</code>
      </div>
      <div className="artifact-meta">
        <ArtifactTypeBadge type={artifact.artifact_type} />
        <span className="artifact-size">{fmtSize(artifact.size_bytes)}</span>
        <span className="artifact-date">{artifact.created_at}</span>
      </div>
    </div>
  );
}

export function EvidenceCell({ status }: { status?: string | null }) {
  if (!status) return <span className="evidence evidence--none">—</span>;
  return <span className={`evidence evidence--${status.toLowerCase()}`}>{status}</span>;
}

export function LifecycleBadge({ state }: { state: string }) {
  return <span className={`badge badge--lifecycle badge--${state.toLowerCase()}`}>{state}</span>;
}

export function HealthBadge({ state }: { state: string }) {
  return <span className={`badge badge--health badge--${state.toLowerCase()}`}>{state}</span>;
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export function EmptyNote({ children }: { children: ReactNode }) {
  return <span className="empty-note">{children}</span>;
}
