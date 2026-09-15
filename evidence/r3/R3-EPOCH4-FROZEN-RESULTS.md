# Epoch 4 frozen backend regression evidence

Server-c main, HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970; uncommitted changes preserved. No commit or push.

Immediately after both runs completed and before releasing the snapshot metadata fix, the FE Python source hash was b7c3e91b156501f8702da7e9fa72d8773715e5e5aeada53cd01045bd5e884c1e, identical to R3-epoch4-source-before.json. All 104 files recorded in that manifest still had identical contents. This includes preserved peer edits and is not an authorship claim.

- Project environment: 1027 passed, 340 skipped, 35 warnings, 323.26s; command and returncode 0 in R07-epoch4-project-watchdog.json. Sampled subprocess-family peak RSS 784199680 bytes; no guard trigger.
- System Pandas 3 environment: 1019 passed, 348 skipped, 630 warnings, 318.43s; command and returncode 0 in R07-epoch4-pandas3-watchdog.json. Sampled subprocess-family peak RSS 603230208 bytes; no guard trigger.
- Exact per-node results and skip messages are retained in the respective XML/log files. Skips are not passes; test counts are not unique operator counts.

Sampling every 0.1s with a 2 GiB test safeguard is not a kernel memory cap, production 80% RSS acceptance, or 100k/GPU benchmark. The later runtime snapshot inference fix changes source identity; this epoch is evidence for the hash above, not a claim that all later source changes were exercised by these full runs.

Overall review remains PARTIAL. Approved-data default run, complete parameter-domain certification, exact negative-binding replan, and label/diagnostic workflow require additional work or authorized input as listed in R3-STATUS-AND-GATE-MAP-20260912.md.
