# Tail saturation execution identity

Optimizer proposal deduplication for `TAIL_SATURATION` resolves the same FE
`winsorize` registration as the adapter (`pandas_numpy`, mode `any`). Its
execution signature includes the canonical name, catalog `semantic_version`
field, operator type, implementation and contract hashes, adapter source hash,
and Python/NumPy/pandas versions. FE currently records an empty string for
`winsorize.semantic_version`; that empty value is preserved and is not treated
as a release version. Missing registrations or incomplete identity metadata
fail closed, leaving affected proposals separate.

The shared FE identity builder also rejects a non-string `semantic_version`
(including `None`) as uncertifiable. It never invents a version fallback.

This change only defines when two proposals may share execution. It does not
change winsorization math, quantiles, or any TRAIN/VAL/TEST boundary. The
adapter continues to fit cross-sectional quantiles inside each date group.
