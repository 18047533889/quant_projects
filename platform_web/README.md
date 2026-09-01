# platform_web

Quant Research Platform web UI — React + TypeScript + Vite frontend
(QRP-P9, task #43). 20 pages mirroring the spec information architecture;
the dashboard renders **only** what the API returns — no computation in the frontend.

**Repo:** https://github.com/HKUST-QUANT-SOCIETY/platform_web (private)
**Backend contracts:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform (`app/contracts/*`)

## Stack

| Concern | Library |
|---|---|
| UI | React 18 |
| Language | TypeScript (strict) |
| Build | Vite 5 |
| Data fetching / cache | TanStack Query |
| Tables | TanStack Table |
| Routing | React Router 6 |
| Charts | ECharts (declared; not yet used) |
| Tests | Vitest |

## Pages (20)

Overview, Factors, Campaigns, Treatments, Clusters, Libraries, FeatureSets,
Models, Backtests, Optimizers, LiveHealth, Streaming, Alerts, Jobs, DataSnapshots,
Artifacts, Standards, Exports, Audit, UsersRoles.

## Scripts

```sh
npm run dev        # vite dev server
npm run typecheck  # tsc --noEmit
npm test           # vitest run
npm run build      # tsc --noEmit && vite build → dist/
```

## Build (CI)

This environment has **no node/npm** — build runs in GitHub Actions
(`.github/workflows/platform-web.yml`): setup-node + `npm ci` →
`typecheck` + `test` + `build`, uploads `dist/`. Generate `package-lock.json`
locally so the setup-node cache key works.

## Constraints

- Pure frontend scaffold: no domain logic, no Python, never recomputes metrics.
- Types in `src/api/types.ts` mirror `quant_platform/app/contracts/*`.
- "No data" is never rendered as 0 — missing values render as `—`, and a
  not-yet-mature label renders the `LABEL_NOT_MATURE` partial-evidence state.

## Related repos

- **quant_platform** — backend contracts / API (types source of truth)
