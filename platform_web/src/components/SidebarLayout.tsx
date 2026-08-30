/**
 * SidebarLayout — spec §25 1-level nav: 6 groups. The nav shape is fixed;
 * individual pages live under /overview, /factors, /campaigns, /treatments,
 * /clusters, /libraries, /feature-sets, /models, /backtests, /optimizers,
 * /live-health, /streaming, /alerts, /jobs, /data-snapshots, /artifacts,
 * /standards, /exports, /audit, /users.
 */
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

const GROUPS: Array<{ label: string; items: Array<{ to: string; label: string }> }> = [
  { label: 'Overview', items: [{ to: '/overview', label: 'Dashboard' }] },
  {
    label: 'Research',
    items: [
      { to: '/factors', label: 'Factors' },
      { to: '/campaigns', label: 'Campaigns' },
      { to: '/treatments', label: 'Treatments' },
      { to: '/clusters', label: 'Clusters' },
      { to: '/libraries', label: 'Libraries' },
    ],
  },
  {
    label: 'Models',
    items: [
      { to: '/feature-sets', label: 'Feature Sets' },
      { to: '/models', label: 'Models' },
    ],
  },
  {
    label: 'Portfolio',
    items: [
      { to: '/backtests', label: 'Backtests' },
      { to: '/optimizers', label: 'Optimizers' },
    ],
  },
  {
    label: 'Production',
    items: [
      { to: '/live-health', label: 'Live Health' },
      { to: '/streaming', label: 'Streaming' },
      { to: '/alerts', label: 'Alerts' },
    ],
  },
  {
    label: 'Platform',
    items: [
      { to: '/jobs', label: 'Jobs' },
      { to: '/data-snapshots', label: 'Data Snapshots' },
      { to: '/artifacts', label: 'Artifacts' },
      { to: '/standards', label: 'Standards' },
      { to: '/exports', label: 'Exports' },
      { to: '/audit', label: 'Audit' },
      { to: '/users', label: 'Users & Roles' },
    ],
  },
];

export function SidebarLayout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">QRP Platform</div>
        <nav aria-label="Primary">
          {GROUPS.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-group-label">{group.label}</div>
              <ul>
                {group.items.map((item) => (
                  <li key={item.to}>
                    <NavLink
                      to={item.to}
                      className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}
                    >
                      {item.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
      <main className="main">
        <div className="main-content">{children}</div>
      </main>
    </div>
  );
}
