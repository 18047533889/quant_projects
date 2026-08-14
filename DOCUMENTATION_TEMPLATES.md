# Documentation Templates

Templates for creating consistent documentation across all packages in quant_projects.

---

## Template 1: QUICKSTART.md

```markdown
# Quick Start Guide

Get started with [Package Name] in 10 minutes.

## Prerequisites

- Python 3.9 or higher
- [List required dependencies]
- [Any data or configuration needed]

## Installation

```bash
# From repository root
pip install -e "./[package_name][all]"

# Or specific extras
pip install -e "./[package_name][client]"
```

## Your First [Main Use Case] (5 minutes)

### Step 1: Import and Initialize

```python
from [package] import get_store  # or main entry point

store = get_store()
```

### Step 2: [Key Action 1]

```python
# Copy-pasteable example
result = store.do_something(
    param1="value1",
    param2="value2",
)
```

**Expected output**:
```
[Show what users should see]
```

### Step 3: [Key Action 2]

```python
# Next logical step
processed = result.transform()
```

### Step 4: Verify

```python
print(f"Success! Processed {len(processed)} items")
```

## What You Just Did

- ✓ [Achievement 1]
- ✓ [Achievement 2]
- ✓ [Achievement 3]

## Next Steps

**Learn More**:
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Architecture](ARCHITECTURE.md) - System design and concepts
- [Examples](../examples/) - More complex workflows

**Common Tasks**:
- [Link to common task 1 example]
- [Link to common task 2 example]
- [Link to common task 3 example]

**Troubleshooting**:
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Check [FAQ.md](FAQ.md) for frequently asked questions

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VAR_NAME` | `default` | What it controls |

### Config Files

Optional configuration in `config/settings.yaml`:

```yaml
key: value
```

## Getting Help

- 📖 [Full Documentation](README.md)
- 💬 Internal Slack: #channel-name
- 🐛 [GitHub Issues](link)
- 📧 Email: team@example.com
```

---

## Template 2: API_REFERENCE.md

```markdown
# API Reference

Complete API documentation for [Package Name].

## Table of Contents

- [Core API](#core-api)
- [Utilities](#utilities)
- [Configuration](#configuration)
- [Exceptions](#exceptions)

---

## Core API

### `function_name()`

**Purpose**: Brief one-line description.

**Signature**:
```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional_param: Type3 = default_value
) -> ReturnType:
```

**Parameters**:
- `param1` (Type1): Description of param1. 
  - Valid values: [range or enum]
  - Default: [if applicable]
- `param2` (Type2): Description of param2.
- `optional_param` (Type3, optional): Description. Default: `default_value`.

**Returns**:
- `ReturnType`: Description of return value.

**Raises**:
- `ExceptionType`: When this exception is raised.

**Example**:
```python
from package import function_name

result = function_name(
    param1="value1",
    param2=42,
)

print(result)
# Output: expected_output
```

**See Also**:
- [`related_function()`](#related-function) - Related functionality
- [Concept Guide](ARCHITECTURE.md#concept) - Background information

---

### `ClassName`

**Purpose**: Brief description of the class.

**Constructor**:
```python
def __init__(
    self,
    required_param: Type1,
    optional_param: Type2 = default
):
```

**Attributes**:
- `attribute_name` (Type): Description of public attribute.

**Methods**:

#### `method_name()`

**Purpose**: What this method does.

**Signature**:
```python
def method_name(self, param: Type) -> ReturnType:
```

**Parameters**: [As above]

**Returns**: [As above]

**Example**:
```python
obj = ClassName(required_param=value)
result = obj.method_name(param=value)
```

---

## Utilities

### Helper Functions

Brief overview of utility functions.

[Follow same format as Core API]

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VAR_NAME` | string | `"default"` | What it controls |
| `NUMERIC_VAR` | int | `100` | Numeric configuration |
| `BOOL_VAR` | bool | `false` | Boolean flag |

**Example**:
```bash
export VAR_NAME="custom_value"
export NUMERIC_VAR=200
```

### Configuration Objects

```python
from package import ConfigClass

config = ConfigClass(
    setting1=value1,
    setting2=value2,
)
```

---

## Exceptions

### `ExceptionName`

**Inherits from**: `BaseException`

**Raised when**: Description of when this exception occurs.

**Attributes**:
- `attribute`: Description

**Example**:
```python
try:
    result = function_that_might_fail()
except ExceptionName as e:
    print(f"Error: {e}")
    # Handle error
```

---

## Index

Quick reference for common tasks:

- **Task 1**: Use [`function_name()`](#function-name)
- **Task 2**: Use [`ClassName.method()`](#classname)
- **Task 3**: See [Examples](../examples/)

---

**Last Updated**: YYYY-MM-DD  
**Version**: X.Y.Z
```

---

## Template 3: TROUBLESHOOTING.md

```markdown
# Troubleshooting Guide

Common issues and solutions for [Package Name].

## Quick Diagnostics

**If you're experiencing issues, first check**:

1. ✓ Latest version installed: `pip show package-name`
2. ✓ Required dependencies: `pip install -e "./package[all]"`
3. ✓ Environment variables configured: `echo $VAR_NAME`
4. ✓ Required data/files present: `ls /path/to/data`

---

## Common Issues

### Issue: Error Message Appears

**Symptom**:
```
ExceptionType: Specific error message here
```

**Cause**: Explanation of why this error occurs.

**Solution**:

1. Check if [condition] is met:
   ```bash
   # Diagnostic command
   command --check
   ```

2. If issue persists, try:
   ```python
   # Fix code
   solution_code()
   ```

3. Verify fix:
   ```bash
   # Verification command
   ```

**Prevention**: How to avoid this issue in the future.

---

### Issue: Unexpected Behavior

**Symptom**: Description of what's happening vs. what should happen.

**Cause**: [Explanation]

**Solution**: [Step-by-step fix]

---

## Performance Issues

### Slow Queries / Operations

**Symptom**: Operation takes longer than expected.

**Diagnosis**:

1. Enable profiling:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Check resource usage:
   ```bash
   top -p $(pgrep -f python)
   ```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| Large dataset without filtering | Add time_range or column filters |
| Inefficient backend | Try different backend (see BACKEND_SELECTION_GUIDE.md) |
| Memory swapping | Reduce batch size or use streaming |

**Example Fix**:
```python
# Before (slow)
result = store.read("dataset")

# After (fast)
result = store.read(
    "dataset",
    time_range=("2024-01-01", "2024-12-31"),
    columns=["col1", "col2"]
)
```

### High Memory Usage

**Symptom**: Process using excessive memory or getting OOM killed.

**Solutions**:

1. **Use streaming**:
   ```python
   for batch in store.read(...).stream(batch_size=10000):
       process(batch)
   ```

2. **Reduce parallelism**:
   ```python
   # Reduce concurrent workers
   result = compute(max_workers=2)
   ```

3. **Use lazy evaluation**:
   ```python
   lazy_result = store.scan_polars(...)  # Doesn't load into memory
   ```

---

## Configuration Issues

### Environment Variable Not Recognized

**Symptom**: Configuration not being applied.

**Checklist**:
- [ ] Variable is exported: `export VAR=value` (not just `VAR=value`)
- [ ] No typos in variable name (case-sensitive)
- [ ] Process restarted after setting variable
- [ ] Variable visible in process: `python -c "import os; print(os.environ.get('VAR'))"`

---

## Integration Issues

### Integration with Other Package Failed

**Symptom**: Error when using [Package Name] with [Other Package].

**Common Causes**:
1. Version incompatibility
2. Missing configuration
3. Data format mismatch

**Solution**: [Specific steps]

---

## Error Reference

### Error Code: E001

**Message**: "Specific error text"

**Meaning**: Explanation

**Fix**: Solution steps

---

## Still Having Issues?

### Before Asking for Help

Collect this information:

1. **Version info**:
   ```bash
   pip show package-name
   python --version
   ```

2. **Error traceback**: Full error output (not just last line)

3. **Minimal reproducible example**:
   ```python
   # Simplest code that shows the issue
   ```

4. **Environment**: OS, Python version, how you installed

### Where to Get Help

- **Documentation**: Check [API Reference](API_REFERENCE.md) and [FAQ](FAQ.md)
- **Examples**: Look for similar usage in [examples/](../examples/)
- **Slack**: Internal team channel #channel-name
- **GitHub Issues**: [Repository link] (for bugs)
- **Email**: team@example.com

---

**Last Updated**: YYYY-MM-DD
```

---

## Template 4: FAQ.md

```markdown
# Frequently Asked Questions

Quick answers to common questions about [Package Name].

## Table of Contents

- [General](#general)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Integration](#integration)

---

## General

### What is [Package Name]?

Brief description of the package, its purpose, and main use cases.

### When should I use [Package Name] vs [Alternative]?

Comparison table or decision tree:

| Use [Package Name] when... | Use [Alternative] when... |
|---------------------------|---------------------------|
| Condition 1 | Condition 1 |
| Condition 2 | Condition 2 |

### Is [Package Name] production-ready?

Status statement with version information.

---

## Installation & Setup

### How do I install [Package Name]?

```bash
pip install -e "./package[all]"
```

See [Installation Guide](QUICKSTART.md#installation) for details.

### What are the system requirements?

- Python 3.9 or higher
- [Other requirements]

### Do I need to configure anything?

[Required vs optional configuration]

---

## Usage

### How do I do [Common Task 1]?

Brief answer with code example:

```python
from package import function
result = function(param="value")
```

See [Example 01](../examples/01_basic_usage.py) for complete code.

### What's the difference between FunctionA and FunctionB?

| Aspect | FunctionA | FunctionB |
|--------|-----------|-----------|
| Purpose | Use case A | Use case B |
| Performance | Fast | Slower but more features |
| When to use | Condition | Condition |

### Can I use [Feature X] with [Feature Y]?

Yes/No with explanation and example if applicable.

### What's the recommended way to do [Task]?

**Best practice**:
```python
# Recommended approach
```

**Why**: Explanation of benefits.

**Alternatives**:
```python
# Alternative approach (when to use)
```

---

## Performance

### How can I make [Operation] faster?

1. **Strategy 1**: Description and example
2. **Strategy 2**: Description and example
3. **Strategy 3**: Description and example

See [Performance Guide](PERFORMANCE.md) for detailed optimization.

### What's the expected performance for [Operation]?

**Typical benchmarks**:
- Small dataset (1K rows): ~100ms
- Medium dataset (100K rows): ~2s
- Large dataset (10M rows): ~30s

**Factors affecting performance**: [List]

### Why is [Operation] slow?

Common causes and solutions:

1. **Cause**: Explanation → **Fix**: Solution
2. **Cause**: Explanation → **Fix**: Solution

---

## Troubleshooting

### I'm getting error "[Error Message]". What does it mean?

**Cause**: Explanation

**Fix**: Step-by-step solution

See [Troubleshooting Guide](TROUBLESHOOTING.md#error-name) for details.

### Why isn't [Feature] working?

**Checklist**:
- [ ] Requirement 1 met
- [ ] Requirement 2 met
- [ ] Configuration correct

### How do I debug [Issue]?

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run operation
```

Then check logs for [specific patterns].

---

## Integration

### Does [Package Name] work with [Other Tool/Package]?

Yes/No with integration example or link to integration guide.

### How do I integrate with [System X]?

Brief overview with link to detailed guide or example.

### Can I use [Package Name] in [Environment] (e.g., Jupyter, production, Docker)?

Yes/No with specific considerations:

**Jupyter**: [Specific notes]
**Production**: [Specific notes]
**Docker**: [Specific notes]

---

## Advanced

### Can I extend [Package Name] with custom [Components]?

Yes, explanation of extension points:

```python
# Example of custom extension
```

### What's the architecture/design philosophy?

Brief overview with link to [ARCHITECTURE.md](ARCHITECTURE.md).

### Where is the source code?

[GitHub repository link]

---

## Not Finding Your Question?

- Check [API Reference](API_REFERENCE.md) for detailed documentation
- Look through [examples/](../examples/) for code samples
- See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
- Ask in Slack #channel-name
- Open a GitHub issue for bugs or feature requests

---

**Last Updated**: YYYY-MM-DD  
**Have a question to add?** Submit a PR or ask in Slack.
```

---

## Template 5: ARCHITECTURE.md

```markdown
# Architecture Guide

Design and implementation overview of [Package Name].

## Overview

High-level description of what the package does and how it fits into the larger system.

```
[ASCII diagram of system components]
```

## Core Concepts

### Concept 1

**Definition**: What it is.

**Purpose**: Why it exists.

**Implementation**: How it's implemented.

**Example**:
```python
# Example showing the concept
```

### Concept 2

[Same structure]

---

## System Components

### Component A

**Responsibility**: What this component does.

**Interface**:
```python
class ComponentA:
    def method1(self, param): ...
    def method2(self, param): ...
```

**Dependencies**:
- Component B (for X functionality)
- External library Y (for Z functionality)

**Implementation Notes**:
- Key design decision 1
- Key design decision 2

### Component B

[Same structure]

---

## Data Flow

### Typical Request Flow

```
1. User Code
   ↓
2. Public API (api/)
   ↓
3. Internal Processing (core/)
   ↓
4. Storage/External System
   ↓
5. Result Processing
   ↓
6. Return to User
```

**Detailed Steps**:

1. **User Code**: Description of what user does
2. **Public API**: What happens in API layer
3. **Internal Processing**: Core logic
4. **Storage**: Data access patterns
5. **Result Processing**: Output formatting
6. **Return**: Final result delivered

### Error Flow

How errors propagate through the system:

```
Error Source → Detection → Wrapping → Propagation → User
```

---

## Design Decisions

### Why [Decision 1]?

**Problem**: What problem we were solving.

**Options Considered**:
1. Option A: Pros/Cons
2. Option B: Pros/Cons
3. Option C (chosen): Pros/Cons

**Decision**: What we chose and why.

**Tradeoffs**: What we gave up.

**References**: ADR-001 (if applicable)

### Why [Decision 2]?

[Same structure]

---

## Extension Points

### How to Add [Custom Component]

**Interface to implement**:
```python
class CustomComponent(BaseClass):
    def required_method(self): ...
```

**Registration**:
```python
registry.register(CustomComponent)
```

**Example**: See [examples/custom_component.py](../examples/custom_component.py)

---

## Performance Characteristics

### Time Complexity

| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| Operation A | O(1) | O(n) | O(n²) |
| Operation B | O(log n) | O(n log n) | O(n log n) |

### Space Complexity

| Operation | Memory Usage |
|-----------|-------------|
| Operation A | O(1) |
| Operation B | O(n) |

### Scalability Limits

- Maximum X: Value (limited by Y)
- Maximum Y: Value (limited by Z)

---

## Security Considerations

### Authentication/Authorization

How security is handled.

### Data Validation

Input validation approach.

### Sensitive Data

How sensitive data is protected.

---

## Dependencies

### External Dependencies

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| library1 | ^1.2.0 | Purpose | MIT |
| library2 | ^2.0.0 | Purpose | Apache 2.0 |

### Internal Dependencies

Dependencies on other packages in the monorepo.

---

## Testing Strategy

### Unit Tests

Coverage of internal components.

### Integration Tests

Testing interaction between components.

### E2E Tests

End-to-end workflow testing.

---

## Future Directions

### Planned Improvements

1. Feature A (planned for vX.Y)
2. Feature B (under consideration)

### Known Limitations

1. Limitation 1: Why it exists, workaround
2. Limitation 2: Why it exists, workaround

---

## References

- [ADR-001: Decision Title](decisions/adr-001.md)
- [Related Package Architecture](../other_package/docs/ARCHITECTURE.md)
- [External Reference](https://example.com)

---

**Last Updated**: YYYY-MM-DD  
**Version**: X.Y.Z
```

---

## Template 6: Example Script

```python
"""
Example: Brief Title

This example demonstrates:
- Feature 1 being showcased
- Feature 2 being showcased  
- Feature 3 being showcased

Prerequisites:
- Package installed: pip install -e "./package[all]"
- [Any data or setup required]

Expected runtime: ~X seconds
Expected output: Description of what you'll see

Usage:
    python examples/XX_example_name.py
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# Package imports
from package import main_function, helper_function


def main():
    """Main example function with clear steps."""
    
    print("=" * 60)
    print("Example: Brief Title")
    print("=" * 60)
    print()
    
    # Step 1: Setup
    print("Step 1: Setup")
    print("-" * 40)
    
    # Clear code with comments explaining each step
    config = {
        "param1": "value1",
        "param2": 42,
    }
    print(f"✓ Configuration: {config}")
    print()
    
    # Step 2: Main operation
    print("Step 2: Perform main operation")
    print("-" * 40)
    
    result = main_function(**config)
    print(f"✓ Result: {result}")
    print()
    
    # Step 3: Process results
    print("Step 3: Process results")
    print("-" * 40)
    
    processed = helper_function(result)
    print(f"✓ Processed: {processed}")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Successfully completed example")
    print(f"✓ Processed {len(processed)} items")
    print()
    print("Next steps:")
    print("  - See examples/XX_next_example.py for more advanced usage")
    print("  - Read docs/API_REFERENCE.md for complete API")
    print("=" * 60)


def demo_variation():
    """Optional: Show variation of the main example."""
    print("\nVariation: Alternative approach")
    print("-" * 40)
    # Alternative code
    print("✓ Completed variation")


if __name__ == "__main__":
    # Make script runnable directly
    try:
        main()
        
        # Optional: Run variations
        # demo_variation()
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  - Check that package is installed")
        print("  - Verify prerequisites are met")
        print("  - See docs/TROUBLESHOOTING.md")
        sys.exit(1)
```

---

## Template 7: Package README.md

```markdown
# Package Name

> One-line description of what this package does

**Status**: Production / Beta / Alpha  
**Version**: X.Y.Z  
**Python**: 3.9+

## Quick Start

```bash
# Install
pip install -e "./package[all]"

# Use
from package import main_function
result = main_function(param="value")
```

See [Quick Start Guide](docs/QUICKSTART.md) for detailed tutorial.

## Features

- ✓ Feature 1: Brief description
- ✓ Feature 2: Brief description  
- ✓ Feature 3: Brief description
- ✓ Feature 4: Brief description

## Installation

### From Source

```bash
git clone https://github.com/org/repo.git
cd repo/package
pip install -e ".[all]"
```

### Extras

| Extra | Purpose | Install |
|-------|---------|---------|
| `[core]` | Core functionality (default) | `pip install -e "."` |
| `[client]` | HTTP client | `pip install -e ".[client]"` |
| `[service]` | API server | `pip install -e ".[service]"` |
| `[all]` | Everything | `pip install -e ".[all]"` |

## Usage Examples

### Example 1: Basic Usage

```python
from package import function_a

result = function_a(
    param1="value1",
    param2=42,
)
print(result)
```

### Example 2: Advanced Usage

```python
from package import function_b

with function_b.context() as ctx:
    result = ctx.process(data)
```

See [examples/](examples/) for more runnable code.

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICKSTART.md) | 10-minute tutorial |
| [API Reference](docs/API_REFERENCE.md) | Complete API docs |
| [Architecture](docs/ARCHITECTURE.md) | Design and concepts |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Examples](examples/) | Runnable code samples |

## Configuration

### Environment Variables

```bash
export VAR_NAME="value"
export NUMERIC_VAR=100
```

### Config File

Optional `config/settings.yaml`:

```yaml
key: value
```

See [Configuration Guide](docs/QUICKSTART.md#configuration) for details.

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linters
ruff check .
mypy .
```

### Project Structure

```
package/
├── __init__.py          # Public API
├── core/                # Core functionality
├── utils/               # Utilities
├── docs/                # Documentation
├── examples/            # Example scripts
├── tests/               # Test suite
└── pyproject.toml       # Package config
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_module.py::test_function

# With coverage
pytest tests/ --cov=package --cov-report=html
```

## Performance

Expected performance for typical workloads:

- Small dataset (1K items): ~100ms
- Medium dataset (100K items): ~2s
- Large dataset (10M items): ~30s

See [Performance Guide](docs/PERFORMANCE.md) for optimization tips.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

[License type] - See [LICENSE](../LICENSE) for details.

## Support

- **Documentation**: Start with [Quick Start](docs/QUICKSTART.md)
- **Examples**: Check [examples/](examples/) directory
- **Issues**: File bugs or feature requests on GitHub
- **Slack**: Internal team channel #channel-name
- **Email**: team@example.com

## Related Packages

- [Related Package 1](../package1/) - Brief description
- [Related Package 2](../package2/) - Brief description

## Citation

If you use this package in research:

```bibtex
@software{package_name,
  title={Package Name},
  author={Team Name},
  year={2024},
  url={https://github.com/org/repo}
}
```

---

**Maintained by**: Team Name  
**Last Updated**: YYYY-MM-DD
```

---

## Usage Guidelines

### When to Use Each Template

1. **QUICKSTART.md**: Every package, first doc users read
2. **API_REFERENCE.md**: All public APIs needing systematic documentation
3. **TROUBLESHOOTING.md**: Packages with common errors or setup complexity
4. **FAQ.md**: Packages with >5 recurring questions
5. **ARCHITECTURE.md**: Complex packages, helps onboarding and design discussions
6. **Example scripts**: Every package, 5-7 progressive examples
7. **README.md**: Every package, gateway to all other docs

### Customization Tips

- Replace `[Package Name]`, `[placeholders]` with actual values
- Keep QUICKSTART under 10 minutes reading time
- Use real code examples that users can copy-paste
- Link between documents extensively
- Update "Last Updated" date when modifying

### Quality Checklist

Before committing documentation:

- [ ] All code examples tested and working
- [ ] No broken links between documents
- [ ] Consistent terminology throughout
- [ ] Prerequisites clearly stated
- [ ] Next steps guide readers forward
- [ ] Troubleshooting covers top 5 errors
- [ ] FAQ answers top 15 questions
- [ ] Examples are self-contained and runnable

---

**Template Version**: 1.0  
**Created**: 2026-08-14  
**For**: quant_projects documentation standardization