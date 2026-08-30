/**
 * App — routes per spec §25 (1-level nav) with the sidebar layout.
 *
 * Every route is a list page rendering ONLY what the platform API returns
 * (spec §49: no domain logic in the frontend).
 */
import { Navigate, Route, Routes } from 'react-router-dom';

import { SidebarLayout } from './components/SidebarLayout';
import { AlertsPage } from './pages/Alerts';
import { ArtifactsPage } from './pages/Artifacts';
import { AuditPage } from './pages/Audit';
import { BacktestsPage } from './pages/Backtests';
import { CampaignsPage } from './pages/Campaigns';
import { ClustersPage } from './pages/Clusters';
import { DataSnapshotsPage } from './pages/DataSnapshots';
import { ExportsPage } from './pages/Exports';
import { FactorsPage } from './pages/Factors';
import { FeatureSetsPage } from './pages/FeatureSets';
import { JobsPage } from './pages/Jobs';
import { LibrariesPage } from './pages/Libraries';
import { LiveHealthPage } from './pages/LiveHealth';
import { ModelsPage } from './pages/Models';
import { OptimizersPage } from './pages/Optimizers';
import { OverviewPage } from './pages/Overview';
import { StandardsPage } from './pages/Standards';
import { StreamingPage } from './pages/Streaming';
import { TreatmentsPage } from './pages/Treatments';
import { UsersRolesPage } from './pages/UsersRoles';

export default function App() {
  return (
    <SidebarLayout>
      <Routes>
        <Route path="/" element={<Navigate to="/overview" replace />} />
        <Route path="/overview" element={<OverviewPage />} />

        <Route path="/factors" element={<FactorsPage />} />
        <Route path="/campaigns" element={<CampaignsPage />} />
        <Route path="/treatments" element={<TreatmentsPage />} />
        <Route path="/clusters" element={<ClustersPage />} />
        <Route path="/libraries" element={<LibrariesPage />} />

        <Route path="/feature-sets" element={<FeatureSetsPage />} />
        <Route path="/models" element={<ModelsPage />} />

        <Route path="/backtests" element={<BacktestsPage />} />
        <Route path="/optimizers" element={<OptimizersPage />} />

        <Route path="/live-health" element={<LiveHealthPage />} />
        <Route path="/streaming" element={<StreamingPage />} />
        <Route path="/alerts" element={<AlertsPage />} />

        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/data-snapshots" element={<DataSnapshotsPage />} />
        <Route path="/artifacts" element={<ArtifactsPage />} />
        <Route path="/standards" element={<StandardsPage />} />
        <Route path="/exports" element={<ExportsPage />} />
        <Route path="/audit" element={<AuditPage />} />
        <Route path="/users" element={<UsersRolesPage />} />

        <Route path="*" element={<Navigate to="/overview" replace />} />
      </Routes>
    </SidebarLayout>
  );
}
