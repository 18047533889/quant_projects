# SMA FactorEngine execution identity

Research proposal deduplication for the value-repair trailing-SMA route binds the
runtime-selected factor_engine.backend.long_smoothing.lagged_mean callable,
the SMA selector adapter source, the selected callable source, and the complete
source file containing the selected FE callable. Hashing the containing module
also detects changes to its local rolling helper. The record includes Python,
NumPy, pandas, and Polars versions.

This identity is intentionally scoped, not a complete runtime-closure hash.
Imported FE modules used by rolling compilation, optional native kernels,
third-party transitive dependencies, and environment/build artifacts outside
those version strings are not individually hashed. The binding must not be
described as proving full runtime reproducibility or as a guarantee that every
dependency substitution changes the signature.
