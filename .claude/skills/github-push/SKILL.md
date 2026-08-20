---
name: github-push
description: GitHub push helper for quant_projects and HKUST org repos
---

# GitHub Push Skill

## Repos

| Repo | Local path | Remote | Branch | Token |
|------|-----------|--------|--------|-------|
| quant_projects | `/home/shw/quant_projects` | `github.com/18047533889/quant_projects` | main | stored in git remote URL |
| factor_engine (HKUST org) | `/home/shw/quant_projects/factor_engine` | `github.com/HKUST-QUANT-SOCIETY/factor_engine` | main | stored in git remote URL |
| data_access (HKUST org) | `/home/shw/quant_projects/dataaccess` | `github.com/HKUST-QUANT-SOCIETY/data_access` | main | stored in git remote URL |

## Standard Push (quant_projects)

```bash
cd /home/shw/quant_projects
git add <paths>
git commit -m "description"
git push origin main
```

## HKUST Org Repos

factor_engine and dataaccess are subtrees or submodules within quant_projects. Push them separately:

```bash
cd /home/shw/quant_projects/factor_engine
git add <paths>
git commit -m "description"
git push origin main
```

```bash
cd /home/shw/quant_projects/dataaccess
git add <paths>
git commit -m "description"
git push origin main
```

## Diverged Branch

If push is rejected due to divergence:
1. Fetch remote changes
2. Merge or rebase onto remote main
3. Resolve any conflicts
4. Push

Never force-push unless explicitly asked.

## Token

Token `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` is embedded in the git remote URLs via `git remote set-url`. You do not need to specify it manually.
