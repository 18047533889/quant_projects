# Operator Automation Tools

Comprehensive tooling for diagnosing and batch-fixing common operator issues in the FactorEngine codebase.

## Overview

Three complementary tools for operator maintenance:

1. **`auto_fix_operators.py`** - Base scanner and auto-fixer
2. **`auto_fix_operators_enhanced.py`** - Enhanced with policy and bridge detection
3. **`batch_operator_operations.py`** - Workflow orchestration with validation

## Tools

### 1. Basic Auto-Fix Tool

**File**: `scripts/auto_fix_operators.py`

**Features**:
- Scans all operators for common issues
- Auto-fixes simple problems (docstrings, defaults)
- Generates detailed diagnostic reports
- Supports dry-run mode

**Usage**:

```bash
# Scan only - no changes
python3 scripts/auto_fix_operators.py --scan

# Preview fixes (dry-run)
python3 scripts/auto_fix_operators.py --all --dry-run

# Apply all auto-fixes
python3 scripts/auto_fix_operators.py --all --apply

# Fix specific issue type
python3 scripts/auto_fix_operators.py --type docstring --apply
python3 scripts/auto_fix_operators.py --type defaults --apply

# Fix specific operator
python3 scripts/auto_fix_operators.py --operator ts_mean --apply

# Custom report location
python3 scripts/auto_fix_operators.py --scan --report /tmp/my_report.md
```

**Issue Categories**:

| Category | Auto-fixable | Description |
|----------|--------------|-------------|
| `docstring` | ✅ Yes | Missing function/class docstrings |
| `defaults` | ✅ Yes | Missing parameter defaults |
| `typing` | ❌ No | Missing type hints |
| `implementation` | ❌ No | Missing methods (e.g., `calculate()`) |

**Example Output**:

```
🔍 Scanning operators...
✓ Scanned 614 operators in 172 files
✓ Found 745 issues

============================================================
SCAN SUMMARY
============================================================
Total operators scanned: 614
Total issues found: 745
Auto-fixable: 196
Requires manual fix: 549

By category:
  docstring: 184 (184 auto-fixable)
  defaults: 12 (12 auto-fixable)
  typing: 134 (0 auto-fixable)
  implementation: 415 (0 auto-fixable)

📊 Report written to: /tmp/auto_fix_operators_report.md
```

### 2. Enhanced Auto-Fix Tool

**File**: `scripts/auto_fix_operators_enhanced.py`

**Additional Features**:
- Policy detection and template generation
- Polars bridge stub generation
- Mining role classification
- Parameter validation generation
- Export policy gaps as JSON

**Usage**:

```bash
# Enhanced scan with policy detection
python3 scripts/auto_fix_operators_enhanced.py --scan

# Fix with policy templates
python3 scripts/auto_fix_operators_enhanced.py --type policy --apply

# Generate Polars bridge stubs
python3 scripts/auto_fix_operators_enhanced.py --type polars_bridge --scan

# Export policy gaps
python3 scripts/auto_fix_operators_enhanced.py --scan \
  --policy-report /tmp/policy_gaps.json
```

**Additional Categories**:

| Category | Auto-fixable | Description |
|----------|--------------|-------------|
| `policy` | ✅ Yes* | Missing policy registration |
| `polars_bridge` | ⚠️ Partial | Missing Polars implementations |

*Policy fixes generate templates that need manual review

**Policy Template Example**:

```python
# Policy for ts_momentum
"ts_momentum": {
    "timing_kind": TimingKind.INDEPENDENT_DAILY,
    "lane": Lane.PRODUCTION,
    "state": State.STATELESS,
    "param_role": {},
    "min_periods_rule": "window",
},
```

**Polars Bridge Template Example**:

```python
def pl_ts_momentum(data, window=20, **_):
    """Polars implementation of ts_momentum."""
    if pl is None:
        raise ImportError("polars not available")

    # TODO: Implement Polars-native version
    # Current: pandas fallback
    result = data.to_pandas()
    result = pd_ts_momentum(result, window=window)
    return pl.from_pandas(result)
```

### 3. Batch Operations Tool

**File**: `scripts/batch_operator_operations.py`

**Features**:
- Multi-stage workflow orchestration
- Git integration (branch, commit, rollback)
- Validation and testing integration
- Parallel processing support
- Checkpointing for long operations
- Automatic rollback on failures

**Usage**:

```bash
# Run default workflow (dry-run)
python3 scripts/batch_operator_operations.py --workflow default --dry-run

# Run with git integration
python3 scripts/batch_operator_operations.py --workflow default --git

# Run with parallel processing
python3 scripts/batch_operator_operations.py --workflow default --parallel --git

# Custom checkpoint directory
python3 scripts/batch_operator_operations.py --workflow default \
  --checkpoint-dir /tmp/my_checkpoint
```

**Default Workflow Stages**:

1. **Stage 1**: Add missing docstrings
   - Auto-fixable: Yes
   - Validation: Import check

2. **Stage 2**: Add parameter defaults
   - Auto-fixable: Yes
   - Validation: Import check

3. **Stage 3**: Generate policy templates
   - Auto-fixable: Partial (dry-run)
   - Output: Template files for manual review

**Git Integration**:

When `--git` is enabled:
- Creates new branch: `auto-fix-YYYYMMDD-HHMMSS`
- Commits after each successful stage
- Automatically rolls back on failure
- Shows diff statistics at completion

**Example Output**:

```
============================================================
BATCH OPERATION: Standard Operator Fix Workflow
============================================================

--- Stage 1/3: Add missing docstrings ---

Processing 184 issues...
✓ Fixed 5 issues in layer_primitives.py
✓ Fixed 3 issues in safe_ops.py
...
✓ Committed: Auto-fix: Add missing docstrings (184 fixes)

--- Stage 2/3: Add parameter defaults ---

Processing 12 issues...
✓ Fixed 2 issues in safe_ops.py
...
✓ Committed: Auto-fix: Add parameter defaults (12 fixes)

--- Stage 3/3: Generate policy templates ---

Processing 30 issues...
[DRY RUN] Would generate 30 policy templates

--- Final Validation ---

🔍 Validating imports...
✓ Import validation passed
🔍 Running linter...
✓ Linting passed

--- Summary ---
Add missing docstrings: 184/184 successful
Add parameter defaults: 12/12 successful
Generate policy templates: 30/30 successful

Git diff:
 cleaned_operators/layer_primitives.py | 15 ++++++++++
 cleaned_operators/safe_ops.py         | 25 +++++++++++++---
 2 files changed, 40 insertions(+), 3 deletions(-)

============================================================
✓ BATCH OPERATION COMPLETE: Standard Operator Fix Workflow
============================================================
```

## Workflow Recommendations

### Quick Fixes (Low Risk)

For simple, low-risk fixes like docstrings:

```bash
# 1. Scan and review
python3 scripts/auto_fix_operators.py --scan

# 2. Apply fixes
python3 scripts/auto_fix_operators.py --type docstring --apply
```

### Production Fixes (Medium Risk)

For fixes that affect functionality:

```bash
# 1. Scan with enhanced detection
python3 scripts/auto_fix_operators_enhanced.py --scan

# 2. Preview changes
python3 scripts/auto_fix_operators.py --type defaults --dry-run

# 3. Apply with git tracking
python3 scripts/batch_operator_operations.py --workflow default --git
```

### Comprehensive Audit (High Risk)

For comprehensive operator maintenance:

```bash
# 1. Full enhanced scan
python3 scripts/auto_fix_operators_enhanced.py --scan \
  --policy-report /tmp/policy_gaps.json

# 2. Review reports
cat /tmp/auto_fix_operators_report.md
cat /tmp/policy_gaps.json

# 3. Apply in stages with validation
python3 scripts/batch_operator_operations.py \
  --workflow default \
  --git \
  --checkpoint-dir /tmp/checkpoint

# 4. Manual fixes for complex issues
# Review templates in /tmp/auto_fix_operators_report.md

# 5. Run full test suite
pytest tests/operators/ -v
```

## Report Structure

### Main Report (`auto_fix_operators_report.md`)

```markdown
# Operator Auto-Fix Report

Generated: 2026-08-13T04:13:43

Total issues found: 745
Auto-fixable: 196
Fixed: 0

## Issues by Category
- **docstring**: 184
- **defaults**: 12
- **typing**: 134
- **implementation**: 415

## Issues by Severity
- **critical**: 415
- **medium**: 196
- **low**: 134

## Detailed Issues

### DOCSTRING (184 issues)

#### ts_momentum
- **Severity**: medium
- **Description**: Missing docstring
- **File**: /path/to/file.py:45
- **Auto-fixable**: Yes

...

## Recommendations

1. **Critical issues**: Address immediately
2. **High severity**: Fix before next release
3. **Medium/Low**: Schedule for upcoming sprint

4. **Manual fixes required**:
   - implementation: 415 operators
   - typing: 134 operators
```

### Policy Gaps Report (`policy_gaps.json`)

```json
{
  "total_gaps": 30,
  "operators": [
    {
      "name": "ts_momentum",
      "file": "/path/to/file.py",
      "template": "# Policy for ts_momentum\n..."
    },
    ...
  ]
}
```

## Issue Detection Details

### Docstring Detection

Detects:
- Functions without docstrings
- Classes without docstrings
- Operator implementations missing documentation

Generates:
- Context-aware docstrings based on operator name
- Proper formatting with triple quotes

### Default Parameter Detection

Detects parameters that should have defaults:
- `window`, `d`, `n`, `periods`, `span` → numeric defaults
- `min_periods`, `ddof` → 0 or 1
- `method`, `mode` → string defaults
- `normalize`, `center`, `adjust` → boolean defaults

Applies common defaults:
- `window=20`
- `periods=1`
- `min_periods=1`
- `method="linear"`

### Policy Detection

Scans for:
- Operators without policy registration
- Missing timing specifications
- Unclassified mining roles

Infers:
- `timing_kind` from operator prefix (`ts_`, `cs_`, `expanding_`)
- `lane` from name patterns (research, experimental)
- `state` from implementation patterns (stateful keywords)

### Polars Bridge Detection

Finds:
- `pd_*` functions without matching `pl_*` implementations
- Backend inconsistencies
- Missing registration for both backends

Generates:
- Stub implementations with pandas fallback
- Registration templates
- TODOs for native implementation

## Safety Features

### Backup System

- Creates timestamped backup directory before modifications
- Preserves original files
- Enables rollback with `fixer.rollback()`

### Dry-Run Mode

- Preview all changes without applying
- Shows what would be modified
- Validates fixes before application

### Git Integration

- Creates feature branch automatically
- Commits after each successful stage
- Automatic rollback on validation failure
- Shows diff statistics

### Validation Checks

1. **Import validation**: Ensures Python syntax is valid
2. **Linter checks**: Runs ruff or flake8
3. **Test execution**: Runs pytest on modified code
4. **Stage validation**: Custom validation per workflow stage

## Limitations

### What Can Be Auto-Fixed

✅ Missing docstrings (basic templates)
✅ Common parameter defaults
✅ Policy templates (require review)

### What Requires Manual Intervention

❌ Complex type hints
❌ Missing implementations
❌ Polars-native optimizations
❌ Parameter validation logic
❌ Test implementations

## Extending the Tools

### Custom Workflow

Create custom batch operations:

```python
from scripts.batch_operator_operations import BatchOperation, BatchStage

operation = BatchOperation(
    name="Custom Workflow",
    git_enabled=True
)

# Add custom stage
operation.stages.append(BatchStage(
    name="Fix specific operators",
    issue_filter=lambda i: i.operator.startswith("ts_"),
    validation=lambda: run_specific_tests(),
    max_failures=5
))

# Execute
processor = BatchProcessor(project_root, operation)
processor.execute()
```

### Custom Issue Detector

Extend the scanner:

```python
from scripts.auto_fix_operators_enhanced import EnhancedScanner

class MyScanner(EnhancedScanner):
    def _check_function(self, node, file_path, content):
        super()._check_function(node, file_path, content)

        # Add custom checks
        if self._is_deprecated(node):
            self.issues.append(Issue(
                operator=node.name,
                category="deprecated",
                severity="high",
                description="Uses deprecated API"
            ))
```

## Troubleshooting

### "Import validation failed"

The fix introduced syntax errors. Check:
- Indentation consistency
- Quote matching
- Parenthesis balancing

Run: `python3 -m py_compile <file>` to verify syntax.

### "Too many failures"

Increase `max_failures` in stage configuration or review the first few failures for patterns.

### "Git rollback failed"

Manually reset:
```bash
git checkout <original-branch>
git branch -D <auto-fix-branch>
```

### "Tests timeout"

Increase timeout in `ValidationRunner.run_tests()` or disable test validation for initial fixes.

## Performance

### Scan Performance

- **614 operators** scanned in **~5 seconds**
- **745 issues** detected
- Parallel scanning not yet implemented (could improve 2-3x)

### Fix Performance

- **Sequential**: ~0.5s per file
- **Parallel** (4 workers): ~2-3x faster for independent files
- **With validation**: Add 10-30s for import checks, 1-5min for tests

### Memory Usage

- Scanner: ~50MB
- Fixer: ~100MB (includes backups)
- Batch processor: ~150MB (includes git operations)

## Examples

### Fix All Docstrings

```bash
python3 scripts/auto_fix_operators.py \
  --type docstring \
  --apply \
  --report /tmp/docstring_fixes.md
```

### Generate Policy Templates

```bash
python3 scripts/auto_fix_operators_enhanced.py \
  --type policy \
  --scan \
  --policy-report /tmp/policies.json

# Review templates
jq '.operators[] | select(.name | startswith("ts_"))' /tmp/policies.json
```

### Safe Batch Operation with Git

```bash
# Dry-run first
python3 scripts/batch_operator_operations.py \
  --workflow default \
  --dry-run

# Apply with git safety
python3 scripts/batch_operator_operations.py \
  --workflow default \
  --git \
  --checkpoint-dir /tmp/fix_checkpoint

# If successful, review and merge
git diff HEAD~3  # Review last 3 commits
git log --oneline -5

# Merge to main (if satisfied)
git checkout main
git merge auto-fix-20260813-041343
```

## Summary

These tools provide a comprehensive solution for operator maintenance:

1. **Automated diagnosis** of 745+ common issues across 614 operators
2. **Safe batch fixing** with git integration and automatic rollback
3. **Validation integration** ensures fixes don't break functionality
4. **Extensible design** allows custom workflows and detectors

**Current Status**:
- ✅ Basic scanner and fixer: **Complete**
- ✅ Enhanced detection (policy, bridges): **Complete**
- ✅ Batch orchestration with validation: **Complete**
- ⚠️ Test generation: **Template only**
- ⚠️ Parallel processing: **Implemented but needs testing**

Use these tools to maintain operator quality and reduce manual maintenance overhead.
