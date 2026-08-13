# Local Repository Snapshot Delta

Snapshot time (UTC): `2026-08-13T13:56:36Z`

## Scope and stability

This is a documentation-only delta for `/home/shw/quant_projects`, compared with [`LOCAL_REPO_SNAPSHOT.md`](LOCAL_REPO_SNAPSHOT.md), whose recorded repository state was bound to HEAD `ddb03749b7ff85e63b633770c43dfbbb7562af19`.

The server working tree was not cleaned, checked out, restored, stashed, pulled, or otherwise altered. The only deliberate write for this task is this report. Concurrent sessions advanced the shared server tree while this report was being measured. Accordingly, the state is **UNSTABLE**: the observations below are time-stamped samples, not a single atomic repository transaction.

Earlier function-level audits and inventories are evidence from the same evolving server tree, but they are not automatically certified against the current HEAD. Any current-HEAD certification requires a fresh audit and verification at the relevant commit/state.

## Repository identity delta

| Item | Earlier snapshot (`LOCAL_REPO_SNAPSHOT.md`) | Current observation |
|---|---|---|
| Snapshot binding | `ddb03749b7ff85e63b633770c43dfbbb7562af19` | N/A |
| HEAD | `ddb03749b7ff85e63b633770c43dfbbb7562af19` | `285289a7fd907713284784f53e55fe5a5e518d60` |
| Branch | `main` | `main` |
| `git log -1` | `ddb03749 feat(factor_engine): backend/runtime/modeling remediations and test updates.` | `285289a7 fix(q): keep intermediate region results resident` |
| Current commit date | Not separately recorded here | `2026-08-13T21:55:26+08:00` (`2026-08-13T13:55:26Z`) |
| Current commit author | Not separately recorded here | `Sun Haiwei <2946703196@qq.com>` |

The current HEAD is therefore 2,959 commits? No: the two identifiers are not numerically comparable; the reliable conclusion is that concurrent sessions advanced HEAD from the bound commit to a different commit.

## Working-tree counts

The first measurement and a later measurement disagreed, so both are retained.

| Count | Earlier snapshot observation | Later observation at `2026-08-13T13:56:36Z` |
|---|---:|---:|
| Tracked files (`git ls-files`) | 6,485 | 6,489 |
| Tracked paths with unstaged changes | 0 | 0 |
| Staged paths | 0 | 0 |
| Untracked status entries (`git status --porcelain=v1 -uall`) | 183 short-status entries | 390 entries |
| Individual untracked files (`git ls-files --others --exclude-standard`) | 374 | 390 |
| Status categories | 183 `??` | 390 `??` |

The earlier snapshot explained that short status grouped directory contents, hence its 183 status entries versus 374 individual untracked files. The later measurement used `-uall`; it reports 390 untracked entries and 390 individual untracked files. This increase, together with the HEAD change and tracked-file count change, confirms concurrent activity during the measurement window.

For the later observation, the status split is:

- Tracked modified paths: `0`
- Staged paths: `0`
- Untracked paths: `390`
- Status categories: `390 ??`

## Diff statistics

At both the first and later observations:

- `git diff --stat`: no output; no unstaged tracked diff.
- `git diff --cached --stat`: no output; no staged diff.

The untracked surface is not represented by either diff-stat command.

## Current DataAccess and FactorEngine versions

Read from the current source `pyproject.toml` files:

| Package | Source path | Current project version |
|---|---|---:|
| DataAccess (`data-access`) | `/home/shw/quant_projects/dataaccess/pyproject.toml` | `0.10.2` |
| FactorEngine (`factor-engine`) | `/home/shw/quant_projects/factor_engine/pyproject.toml` | `0.3.1` |

These are source project metadata versions, not necessarily the versions installed in the active interpreter. The earlier snapshot recorded installed `data-access` as `0.8.0` and `factor-engine` as not installed; this delta does not re-run installation or modify the environment.

## Interpretation

The previous snapshot remains a historical baseline bound to `ddb03749`. The current server tree has advanced to `285289a7fd907713284784f53e55fe5a5e518d60` on `main`, while the working tree continues to have no tracked or staged changes and a large, changing untracked surface. Concurrent sessions are the reason the repository identity and untracked counts differ. Earlier function audits, migration inventories, and other Wave 0 artifacts remain useful historical evidence from this evolving tree, but none should be described as current-HEAD certified without re-running the relevant checks against `285289a7` (or a subsequently observed stable HEAD).
