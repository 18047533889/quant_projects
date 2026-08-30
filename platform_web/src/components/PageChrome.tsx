import type { ReactNode } from 'react';

/**
 * PageChrome — consistent header + optional actions for every route page.
 * Pure presentational; no data, no logic.
 */
export function PageChrome({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-chrome">
      <h1>{title}</h1>
      {description ? <p className="page-description">{description}</p> : null}
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}
