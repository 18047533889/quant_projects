# Prepared TRAIN rank: bounded stage benchmark

We tested 14 distinct U/inverted-U plans using FE average-tie percentile rank.
The inputs include ties, NaN and infinities. Each mode ran in a fresh process
at main `8b2a9ffa6b0b098f0d2665ab2fe9bb29f43fb910` with unchanged source.

| Rows | Public cache seconds | Prepared seconds | Public peak MiB | Prepared peak MiB |
|---:|---:|---:|---:|---:|
| 1,500,000 | 2.9105 | 1.8393 | 473.99 | 473.87 |
| 8,500,000 | 16.6532 | 6.3434 | 1021.17 | 989.13 |

Both modes called the FE rank executor once. The mutable public cache computed
14 fingerprints of the same frame; preparation computed one. Input identities,
per-plan output hashes, aggregate output hashes and FE rank binding matched.
Both modes preserved the input frame. At 8.5 million rows, preparation reduced
measured elapsed time by 61.9% in this single ordered pair.

Elapsed time includes frame construction, preparation, plan/context/index
validation and output hashing. We used synthetic full cross-sections, not COS
factor loading or the entire optimizer selection pipeline. These runs do not
measure out-of-sample improvement or prove a speedup for every optimizer method.
Raw receipts use `prepared_train_rank_{1500000,8500000}_{public,prepared}_20261001.json`.
The `available_ram_preflight_bytes` receipt field samples available RAM at the
end; the script also enforces a separate start-time RAM safety check.
