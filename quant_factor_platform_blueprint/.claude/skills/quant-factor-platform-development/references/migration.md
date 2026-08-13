# Migration Rules

Every legacy symbol must be REUSE_AFTER_TEST, REWRITE, REFERENCE_ONLY, CORPUS_ONLY, or DISCARD. Preserve useful semantics, not obsolete architecture. Do not keep production imports from legacy directories after migration. Server code is authoritative; historical GitHub paths are hints only.
