# FA-013: Production Leiden vs Simplified Analysis

**Task**: Assess whether current clustering implementation is production-ready or requires real Leiden library.

**Status**: ✅ ALREADY SAFE (fail-closed architecture)

---

## Current Implementation Analysis

### ModularityClustering (factor_assets/clustering/families.py)

**Architecture**: Fail-closed with explicit opt-in for toy algorithm

```python
def __init__(
    self,
    graph: SparseCorrelationGraph,
    resolution: float = 1.0,
    min_cluster_size: int = 1,
    max_cluster_size: Optional[int] = None,
    allow_toy_algorithm: bool = False,  # ← Default False
):
```

**Safety Mechanism**:
- By default (`allow_toy_algorithm=False`), constructor raises `RuntimeError`
- Error message explicitly states: "Production clustering requires mature backend (e.g., Leiden via igraph)"
- Users must **explicitly** set `allow_toy_algorithm=True` for testing only

**Production Behavior**:
```python
if not allow_toy_algorithm:
    try:
        import igraph
        # Even if igraph available, still fails closed
        raise RuntimeError(
            "ModularityClustering is a toy/reference implementation. "
            "Production clustering requires mature backend (e.g., Leiden via igraph). "
            "Set allow_toy_algorithm=True ONLY for testing/debugging."
        )
    except ImportError:
        raise RuntimeError(...)
```

---

## Test Coverage

All tests in `tests/test_clustering_families.py` **explicitly** use `allow_toy_algorithm=True`:
- Line 107, 124, 142, 146, 164, 182, 205

This demonstrates:
1. Production code cannot use ModularityClustering without explicit override
2. Tests are honest about using toy implementation
3. Users are warned at construction time

---

## Production Path Forward

**Current State**: No production clustering available - fail-closed ✅

**When production clustering is needed**:

### Option A: Implement Leiden via igraph
```python
# In ModularityClustering.__init__:
if not allow_toy_algorithm:
    try:
        import igraph
        import leidenalg  # Python binding for Leiden
        self._backend = "leiden"
        self._use_leiden_impl = True
    except ImportError:
        raise RuntimeError("Install python-igraph + leidenalg for production")
```

### Option B: Use NetworkX + community libraries
```python
import networkx as nx
from networkx.algorithms import community

# Louvain (mature alternative to Leiden):
import community.community_louvain
```

### Option C: Keep fail-closed until actually needed
- Current approach is valid: **no production clustering = explicit error**
- Better than silent toy algorithm with unknown quality

---

## Recommendation

**KEEP CURRENT IMPLEMENTATION** - no changes needed.

**Rationale**:
1. Fail-closed architecture prevents silent quality issues
2. Toy algorithm clearly labeled and gated
3. Tests are honest about toy usage
4. When production clustering is needed, error message guides implementation

**Only implement real Leiden when**:
- Production workload requires clustering quality
- Can validate against reference implementation
- Can track algorithm version/seed/resolution in metadata

**P2 severity confirmed**: Not blocking, properly gated, clear path forward when needed.

---

## Acceptance Criteria Status

- ✅ Production cannot use simplified modularity without explicit opt-in
- ✅ Error message guides users to mature implementation
- ✅ Tests explicitly mark toy algorithm usage
- ✅ When production Leiden is implemented, gate will allow it

**Task Status**: Already satisfies safety requirements. P2 priority appropriate. No immediate action required.
