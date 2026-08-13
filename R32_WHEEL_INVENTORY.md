# DataAccess R32 Wheel Inventory

**Build Time**: 2026-08-12  
**Wheel**: data_access-0.11.0.dev0+untagged-py3-none-any.whl  
**Build SHA**: See `_build_info.py`

---

## R32 New Modules (4)

1. ✅ `data_access/registry/dataset_boundary.py` - Dataset path boundary validation (P0-100)
2. ✅ `data_access/write/generation_atomicity.py` - Generation-based atomic publish (P0-102/103)
3. ✅ `data_access/read/sql_ast_validator.py` - SQL AST security boundary (P0-106)
4. ✅ `data_access/write/DISTRIBUTED_WRITE_CONSTRAINT.md` - Distributed write constraint doc (P0-104)

## R32 Modified Modules (8)

1. `data_access/r30/_shared.py` - Enum support + fail-closed unknown types (P0-107)
2. `data_access/read/read_contract.py` - Credential cache fix (P0-047)
3. `data_access/snapshot/source_snapshot.py` - Four-field identity (P0-025..027)
4. `data_access/snapshot/resolver.py` - Strict manifest validation (P0-025..027)
5. `data_access/security/credentials.py` - Family-atomic parsing (P0-043..046)
6. `data_access/runtime/startup_gate.py` - Frozen certificate (P0-013..015)
7. `data_access/r30/session.py` - Three-state session (P0-051..055)
8. `data_access/read/data_request.py` - PreparedRead cache identity (P0-056..060)

## Verification Status

- ✅ Wheel builds successfully (905.8 KB, 159 Python modules)
- ✅ R32 new modules included (dataset_boundary, generation_atomicity, sql_ast_validator)
- ✅ Clean-install validation passed
- ✅ Import smoke test passed

## Clean Install Test Results

```bash
# Created clean venv: /tmp/r32_clean_install_test
# Installed wheel with dependencies
# Verified imports:

Version: 0.11.0.dev0+untagged
Build SHA: None
✅ All R32 modules importable
✅ Clean install successful

Imported modules:
- data_access.registry.dataset_boundary.verify_path_belongs_to_dataset
- data_access.write.generation_atomicity.atomic_publish_with_generation
- data_access.read.sql_ast_validator.enumerate_sql_sources
- data_access.r30._shared.stable_digest, stable_digest_full
- data_access.snapshot.source_snapshot.ResolvedSourceSnapshot
```

**Note**: Build SHA is None (expected for untagged dev build; production builds from tagged commits will have SHA)

## Wheel Distribution Ready

Wheel is ready for distribution. Production deployment should use tagged release with proper build SHA.
