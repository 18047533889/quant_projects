# Documentation Implementation Checklist

**Based on**: DOCUMENTATION_AUDIT.md (2026-08-14)  
**Priority**: P0 tasks for production packages (dataaccess, factor_engine)

---

## Phase 1: dataaccess (Week 1) - 25 hours

### Day 1-2: Core Guides (11 hours)

- [ ] **docs/QUICKSTART.md** (2 hours)
  - Installation from pyproject.toml
  - Basic read example (5 lines)
  - Time range filtering
  - Column selection
  - First factor read
  - Next steps links

- [ ] **docs/API_REFERENCE.md** (4 hours)
  - Extract from docs/用户使用手册.md
  - Structure:
    - Core API (get_store, read, read_frame, read_arrow)
    - Factor API (read_factors, get_factor_catalog)
    - Advanced API (read_joined, sql_relation, plan)
    - Write API (write_arrow, compute_and_write, publish_from_staging)
    - Configuration (QueryBudget, environment variables)
  - Each entry: signature, parameters, returns, example

- [ ] **docs/TROUBLESHOOTING.md** (3 hours)
  - Common errors:
    - "Dataset not found in datasets.yaml"
    - "DuckDB: Connection failed"
    - "COS authentication failed"
    - "Memory limit exceeded"
    - "DeadlineExceeded during query"
  - Performance issues:
    - Slow queries (partition pruning)
    - Memory usage (streaming vs eager)
  - COS-specific:
    - Mirror vs remote mode selection
    - Credential configuration
    - PIT join issues
  - Getting help section

- [ ] **docs/FAQ.md** (2 hours)
  - 15-20 questions covering:
    - When to use read vs read_frame vs read_arrow?
    - How to configure COS remote mode?
    - What's the difference between mirror/remote/auto?
    - How do PIT joins work?
    - How to read multiple factors efficiently?
    - What's a ReadHandle vs ScanHandle?
    - How to set query timeout?
    - How to debug slow queries?
    - Can I use Polars instead of pandas?
    - How to write computed factors back?

### Day 3-4: Examples (6 hours)

- [ ] **Create examples/ directory**
- [ ] **examples/README.md** (30 min)
  - Overview of all examples
  - Prerequisites
  - How to run

- [ ] **examples/01_basic_read.py** (45 min)
  ```python
  # Simple DataFrame read with time range and columns
  from data_access import get_store
  store = get_store()
  df = store.read_frame(...)
  ```

- [ ] **examples/02_cos_remote.py** (1 hour)
  ```python
  # Configure and use COS remote mode
  import os
  os.environ["DATA_ACCESS_COS_READ_MODE"] = "remote"
  # ... credentials setup
  # Read from COS without local mirror
  ```

- [ ] **examples/03_pit_join.py** (1.5 hours)
  ```python
  # Point-in-time join between price and fundamentals
  store.read_joined(
      anchor="ashare_stock_daily",
      fields=["close", "pe_ratio"],
      joins=[...],
  )
  ```

- [ ] **examples/04_batch_factors.py** (1 hour)
  ```python
  # Read multiple factors in single query
  store.read_factors(
      factor_ids=["mom_20", "vol_60", "rsi_14"],
      time_range=("2024-01-01", "2024-12-31"),
      layout="wide",
  )
  ```

- [ ] **examples/05_http_client.py** (45 min)
  ```python
  # Remote HTTP client usage
  from data_access.service.client import DataAccessClient
  client = DataAccessClient(...)
  ```

- [ ] **examples/06_compute_and_write.py** (1 hour)
  ```python
  # SQL pipeline: read → transform → write
  store.compute_and_write(
      "SELECT ... FROM {{ashare_stock_daily}}",
      write_dataset="factor_lake_staging",
      factor_id="my_factor",
  )
  ```

- [ ] **examples/07_semantic_fields.py** (30 min)
  ```python
  # Semantic field resolution
  fields = store.resolve_fields(["close", "volume"])
  # DataRequest with semantic fields
  ```

### Day 5: Docstrings (8 hours)

Focus on public API (top 20 most-used functions):

- [ ] **store.py**: DataAccessStore class
  - [ ] get_store()
  - [ ] read()
  - [ ] read_frame()
  - [ ] read_arrow()
  - [ ] load_columns()
  - [ ] read_factors()
  - [ ] read_joined()
  - [ ] sql_relation()
  - [ ] compute_and_write()
  - [ ] write_arrow()
  - [ ] publish_from_staging()

- [ ] **read/read_handle.py**: ReadHandle class
  - [ ] to_pandas()
  - [ ] to_arrow()
  - [ ] to_polars()
  - [ ] to_lazy()
  - [ ] stream()

- [ ] **read/data_request.py**: DataRequest, ReadPlan
  - [ ] DataRequest.__init__()
  - [ ] ReadPlan.explain()
  - [ ] ReadPlan.execute()

- [ ] **cos_contract.py**: COS contracts
  - [ ] get_cos_contract()
  - [ ] validate_panel_request()

---

## Phase 2: factor_engine (Week 2) - 38 hours

### Day 1-2: Core Guides (15 hours)

- [ ] **docs/QUICKSTART.md** (3 hours)
  - Installation
  - First factor in 10 minutes:
    - Define simple momentum factor
    - Configure data source
    - Materialize to parquet
    - Verify output
  - Backend selection basics
  - Next steps

- [ ] **docs/API_REFERENCE.md** (8 hours - complex)
  - DSL API:
    - Factor definition (from api import Factor)
    - Operator catalog (cleaned_ops)
    - Data source configuration
    - Backend selection
  - Python API:
    - api.factor module
    - api.operator_registry
    - Materialization (run_many_parallel)
  - Configuration:
    - Backend types (pandas, polars, duckdb, auto)
    - Resource limits
    - Execution policies
  - Integration:
    - DataAccess integration
    - Mining integration
  - Each section: examples, parameters, returns

- [ ] **docs/TROUBLESHOOTING.md** (4 hours)
  - Operator errors:
    - "Operator X not found"
    - "Backend Y doesn't support operator Z"
    - "Parameter validation failed"
    - "Rolling window insufficient history"
  - Backend issues:
    - Auto selection not optimal
    - DuckDB SQL generation failed
    - Memory exhausted in pandas backend
  - Performance:
    - Slow materialization
    - High memory usage
    - Backend selection for speed
  - PIT timing:
    - Point-in-time violations
    - Look-ahead bias detection
  - Mining integration:
    - "Dataset not found"
    - Field resolution failed
  - Hard gate failures:
    - Explanation of each gate
    - How to diagnose failures

### Day 3: FAQ + Architecture (7 hours)

- [ ] **docs/FAQ.md** (3 hours)
  - 25-30 questions:
    - What backend should I use?
    - How to define a custom operator?
    - What's the difference between daily vs intraday?
    - How to use DataAccess with FactorEngine?
    - Can I mix operators from different families?
    - How to debug slow factor computation?
    - What's a hard gate and why did mine fail?
    - How to materialize multiple factors efficiently?
    - What's PIT-safe vs PIT-unsafe?
    - How to handle missing data?
    - Can I use custom data sources?
    - How to optimize memory usage?
    - What's the cost model?
    - How to profile factor execution?

- [ ] **docs/ARCHITECTURE.md** (4 hours)
  - Consolidate from scattered ADR files:
    - Overall system design
    - Operator architecture (families, kernels, backends)
    - DSL parsing and compilation
    - Multi-backend execution
    - Data flow (source → compute → materialize)
    - Integration points (DataAccess, mining)
    - Cost model design
    - Hard gates philosophy
  - Reference ADR files for details

### Day 4: Examples + Cleanup (8 hours)

- [ ] **Verify examples/** (4 hours)
  - Test each existing example runs
  - Fix broken examples
  - Add missing examples:
    - Backend comparison
    - Custom operator definition
    - Batch materialization
    - Mining integration

- [ ] **Reorganize docs/** (2 hours)
  - Create docs/archive/ or docs/audits/
  - Move R15-R43 reports to archive/
  - Keep only current user-facing docs in main docs/
  - Update docs/README.md as index

- [ ] **Create docs/DEPLOYMENT_QUICKSTART.md** (2 hours)
  - Production deployment checklist
  - Resource configuration
  - Monitoring setup
  - Common deployment issues

### Day 5: Critical Docstrings (8 hours)

Focus on public API:

- [ ] **api/__init__.py**: Public exports
- [ ] **api/factor.py**: Factor class
  - [ ] Factor definition
  - [ ] Factor.materialize()
- [ ] **api/operator_registry.py**: Operator discovery
  - [ ] get_operator()
  - [ ] list_operators()
- [ ] **api/cleaned_ops.py**: Core operators
  - Top 20 most-used operators
- [ ] **api/mining_integration.py**: DataAccess integration
  - [ ] default_ashare_pv_data_source_config()
- [ ] **core/**: Execution engine (if public)
- [ ] **planner/**: Query planning (if public)

---

## Phase 3: factor_layer (Week 3, Days 1-3) - 21 hours

- [ ] **Create docs/ directory**
- [ ] **docs/QUICKSTART.md** (2 hours)
  - Factor evaluation in 5 minutes
  - Basic IC calculation
  - Quantile analysis

- [ ] **docs/API_REFERENCE.md** (4 hours)
  - Evaluation metrics API
  - Factor registration API
  - Agent API (if applicable)

- [ ] **docs/ARCHITECTURE.md** (2 hours)
  - Layer design
  - Evaluation pipeline
  - Integration with other packages

- [ ] **docs/TROUBLESHOOTING.md** (1 hour)
  - Common evaluation errors
  - Performance issues

- [ ] **docs/FAQ.md** (1 hour)
  - 10-15 questions on evaluation

- [ ] **examples/** (4 hours)
  - Create 3-5 examples
  - IC calculation
  - Quantile analysis
  - Factor registration

- [ ] **Docstrings** (6 hours)
  - Public evaluation API
  - Factor registration functions
  - Top 50 most-used functions

- [ ] **Consider pyproject.toml** (1 hour)
  - Make package installable

---

## Phase 4: Minor Packages (Week 3, Days 4-5) - 13 hours

### factor_optimizer (7 hours)

- [ ] **docs/API_REFERENCE.md** (3 hours)
- [ ] **docs/TROUBLESHOOTING.md** (2 hours)
- [ ] **docs/FAQ.md** (2 hours)

### factor_preprocess (6 hours)

- [ ] **docs/API_REFERENCE.md** (3 hours)
- [ ] **docs/TROUBLESHOOTING.md** (2 hours)
- [ ] **docs/FAQ.md** (1 hour)

---

## Verification Checklist

After completing each package:

- [ ] All examples run without errors
- [ ] QUICKSTART takes new user <15 minutes
- [ ] API_REFERENCE has all public functions
- [ ] TROUBLESHOOTING covers top 10 errors seen in issues/slack
- [ ] FAQ answers top 20 questions
- [ ] Docstrings in all public API functions
- [ ] Cross-references between docs work
- [ ] No broken links

---

## Success Criteria

### Documentation Completeness
- [ ] dataaccess: 7/7 required docs (was 1/7)
- [ ] factor_engine: 7/7 required docs (was 2/7)
- [ ] factor_layer: 7/7 required docs (was 1/7)
- [ ] factor_optimizer: 7/7 required docs (was 4/7)
- [ ] factor_preprocess: 7/7 required docs (was 4/7)

### Docstring Coverage
- [ ] dataaccess: 65% → 85%
- [ ] factor_engine: 72% → 85%
- [ ] factor_layer: 49% → 75%
- [ ] factor_optimizer: 95% maintained

### Examples
- [ ] dataaccess: 0 → 7 runnable examples
- [ ] factor_engine: All examples verified working
- [ ] factor_layer: 0 → 5 examples

### User Experience
- [ ] New user can complete QUICKSTART in <15 minutes
- [ ] 80% of common questions answered in docs
- [ ] Clear navigation from README → detailed docs

---

## Tools and Templates

### Documentation Templates
Located in this document (see DOCUMENTATION_AUDIT.md for full templates):
- QUICKSTART.md template
- API_REFERENCE.md template
- TROUBLESHOOTING.md template
- FAQ.md template

### Docstring Template
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief one-line summary.
    
    Detailed description explaining what the function does,
    when to use it, and any important considerations.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
        
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

### Example Template
```python
"""
Title: Brief description

This example demonstrates:
- Feature 1
- Feature 2
- Feature 3

Prerequisites:
- Installed package
- Data available in X location

Expected output:
- Description of what you should see
"""

# Clear step-by-step code with comments
# ...

if __name__ == "__main__":
    # Make runnable as script
    pass
```

---

## Notes

- All paths relative to /home/shw/quant_projects
- Coordinate with team to avoid merge conflicts
- Test all examples before committing
- Update this checklist as you complete tasks
- Estimated total: ~97 hours over 3 weeks (split across team)

---

**Created**: 2026-08-14  
**Based on**: DOCUMENTATION_AUDIT.md comprehensive analysis  
**Status**: Ready for team review and task assignment