# Immutable ndarray freeze: single-copy A/B (2026-09-28)

The QE input/artifact freeze path previously built a C-contiguous `np.array(copy=True)`
and then converted that copy to immutable bytes. For an ndarray input,
`array.tobytes(order="C")` already makes the required immutable copy.
The new path serializes directly from the caller's ndarray view; list/tuple
inputs retain the existing construction path.

One fresh-process smoke comparison on server-c used an 8,388,608-element
float64 array (64 MiB). Both outputs were value-identical and ultimately owned
by immutable bytes:

| Path | Freeze elapsed | Process peak RSS |
| --- | ---: | ---: |
| Previous committed code | 0.0422 s | 360,076 KiB |
| Direct ndarray-to-bytes | 0.0334 s | 295,304 KiB |

These are single-run process measurements, not a route certification or a
guaranteed speedup under every load. The RSS includes interpreter imports and
the caller-owned input. The lower peak is consistent with removing the
intermediate 64 MiB ndarray. Tests cover C/F/strided layouts, source mutation
after freezing, immutable bytes ownership, object-array rejection, and the
existing production artifact isolation contracts. The full QE suite after
this change finished with 3129 passed and 26 skipped (35 warnings).
