import { useQuery } from '@tanstack/react-query';

import { platformApi } from '../api/client';

/**
 * TanStack Query hooks wrapping the typed API client.
 *
 * `queryKey` derives from the resource path so a refetch of the same resource
 * shares one cache entry; `retry` respects the API's `retryable` flag (spec §50
 * error model) and the platform never auto-retries on 403/terminal errors.
 *
 * These hooks return the full query result, which the pages feed into the
 * <StatePanel> six-state component (loading / empty / permission-denied /
 * partial-evidence / error / stale).
 */

const STALE_TIME = 30_000;

export function useDashboard() {
  return useQuery({
    queryKey: ['dashboard'],
    queryFn: platformApi.dashboard,
    staleTime: STALE_TIME,
  });
}

export function useFactors() {
  return useQuery({
    queryKey: ['factors'],
    queryFn: () => platformApi.listFactors(),
    staleTime: STALE_TIME,
  });
}

export function useCandidates() {
  return useQuery({
    queryKey: ['candidates'],
    queryFn: platformApi.listCandidates,
    staleTime: STALE_TIME,
  });
}

export function useCampaigns() {
  return useQuery({
    queryKey: ['campaigns'],
    queryFn: platformApi.listCampaigns,
    staleTime: STALE_TIME,
  });
}

export function useTreatments() {
  return useQuery({
    queryKey: ['treatments'],
    queryFn: platformApi.listTreatments,
    staleTime: STALE_TIME,
  });
}

export function useClusters() {
  return useQuery({
    queryKey: ['clusters'],
    queryFn: platformApi.listClusters,
    staleTime: STALE_TIME,
  });
}

export function useLibraries() {
  return useQuery({
    queryKey: ['libraries'],
    queryFn: platformApi.listLibraries,
    staleTime: STALE_TIME,
  });
}

export function useFeatureSets() {
  return useQuery({
    queryKey: ['feature_sets'],
    queryFn: platformApi.listFeatureSets,
    staleTime: STALE_TIME,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: platformApi.listModels,
    staleTime: STALE_TIME,
  });
}

export function useBacktests() {
  return useQuery({
    queryKey: ['backtests'],
    queryFn: platformApi.listBacktests,
    staleTime: STALE_TIME,
  });
}

export function useOptimizers() {
  return useQuery({
    queryKey: ['optimizers'],
    queryFn: platformApi.listOptimizers,
    staleTime: STALE_TIME,
  });
}

export function useLiveHealth() {
  return useQuery({
    queryKey: ['live_health'],
    queryFn: platformApi.listLiveHealth,
    staleTime: STALE_TIME,
  });
}

export function useStreaming() {
  return useQuery({
    queryKey: ['streaming'],
    queryFn: platformApi.listStreaming,
    staleTime: STALE_TIME,
  });
}

export function useAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: platformApi.listAlerts,
    staleTime: STALE_TIME,
  });
}

export function useJobs() {
  return useQuery({
    queryKey: ['jobs'],
    queryFn: () => platformApi.listJobs(),
    staleTime: STALE_TIME,
  });
}

export function useDataSnapshots() {
  return useQuery({
    queryKey: ['data_snapshots'],
    queryFn: platformApi.listDataSnapshots,
    staleTime: STALE_TIME,
  });
}

export function useArtifacts() {
  return useQuery({
    queryKey: ['artifacts'],
    queryFn: platformApi.listArtifacts,
    staleTime: STALE_TIME,
  });
}

export function useStandards() {
  return useQuery({
    queryKey: ['standards'],
    queryFn: platformApi.listStandards,
    staleTime: STALE_TIME,
  });
}

export function useExports() {
  return useQuery({
    queryKey: ['exports'],
    queryFn: platformApi.listExports,
    staleTime: STALE_TIME,
  });
}

export function useAudit() {
  return useQuery({
    queryKey: ['audit'],
    queryFn: platformApi.listAudit,
    staleTime: STALE_TIME,
  });
}

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: platformApi.listUsers,
    staleTime: STALE_TIME,
  });
}

export function useEvents() {
  return useQuery({
    queryKey: ['events'],
    queryFn: platformApi.events,
    staleTime: STALE_TIME,
  });
}
