# R24 Snapshot Coherence Audit

R24-075..083:

- snapshot_now_only is caller INTENT, not proof: production requires source
  snapshot_valid_at / snapshot_created_at (R24-075..077).
- _window_is_historical is judged against the decision context / snapshot validity,
  never machine datetime.now() (R24-078).
- a composite child with no snapshot token is UNVERIFIABLE in production (R24-079/080).
- the manifest records EXACTLY the epoch verified for the read; a data=A/manifest=B
  drift is refused (retry) (R24-081..083).
