# R40 meaningful k-NN operator checks

Replaced stale one-input generated tests with valid target plus three distinct
feature panels for eight assets. No production implementation was changed.

Independent oracle builds average-rank normalized feature vectors, L2
distances, a kth-distance inclusive radius, peer means and Dirichlet energy.
Additional cases cover self exclusion, all boundary ties, feature/target
missingness, all-NaN features, retention, determinism and future-prefix
invariance. No same-day target availability constraint was relaxed.

Root verification: evidence/r40-dynamic-knn-root.log — **15 passed**.
These tests are not added to the historical paired-campaign coverage count.
