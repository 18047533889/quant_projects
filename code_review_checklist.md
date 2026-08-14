# Code Review Checklist

## 1. Code Quality Fundamentals

### Style & Formatting
- [ ] Code follows PEP 8 style guidelines
- [ ] Consistent indentation (4 spaces)
- [ ] Line length ≤ 120 characters
- [ ] No trailing whitespace
- [ ] Proper import ordering (stdlib, third-party, local)
- [ ] No unused imports

### Naming Conventions
- [ ] Variables use `snake_case`
- [ ] Constants use `UPPER_SNAKE_CASE`
- [ ] Classes use `PascalCase`
- [ ] Functions use descriptive, action-oriented names
- [ ] No single-letter variables (except loop counters)

### Documentation
- [ ] Module-level docstring present
- [ ] Public functions have docstrings with parameters and return types
- [ ] Complex logic has inline comments
- [ ] Type hints added for function signatures

## 2. Code Structure

### Function Quality
- [ ] Functions do one thing well (Single Responsibility)
- [ ] Function length ≤ 50 lines (guideline)
- [ ] Cyclomatic complexity ≤ 10
- [ ] Maximum 5 parameters per function
- [ ] No deep nesting (max 3-4 levels)

### Class Design
- [ ] Classes have clear, single responsibility
- [ ] No god objects (≤ 10 public methods)
- [ ] Proper use of inheritance vs composition
- [ ] Instance attributes defined in `__init__`

### File Organization
- [ ] File length ≤ 500 lines (guideline)
- [ ] Related functionality grouped together
- [ ] Clear separation of concerns
- [ ] No circular dependencies

## 3. Error Handling

### Exception Management
- [ ] Specific exceptions caught (not bare `except:`)
- [ ] No silently swallowed exceptions
- [ ] Proper exception context preserved
- [ ] Resource cleanup in `finally` or context managers
- [ ] No broad `Exception` catches without re-raise

### Validation
- [ ] Input parameters validated
- [ ] Boundary conditions checked
- [ ] Null/None checks where needed
- [ ] Type validation for dynamic inputs

## 4. Performance & Efficiency

### Algorithm Efficiency
- [ ] Appropriate data structures used
- [ ] No unnecessary copies of large data
- [ ] Efficient iteration patterns
- [ ] Database queries optimized (minimize N+1)

### Memory Management
- [ ] Large datasets handled in chunks/streaming
- [ ] Resources explicitly closed
- [ ] No obvious memory leaks
- [ ] Generator used for large sequences

## 5. Testing

### Test Coverage
- [ ] Unit tests for new functions
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Integration tests for complex flows
- [ ] Test names clearly describe what they test

### Test Quality
- [ ] Tests are independent and isolated
- [ ] No flaky tests
- [ ] Meaningful assertions
- [ ] Setup and teardown properly handled

## 6. Security & Safety

### Data Safety
- [ ] No hardcoded credentials or secrets
- [ ] Sensitive data properly masked in logs
- [ ] SQL injection prevented (parameterized queries)
- [ ] Path traversal vulnerabilities checked
- [ ] Input sanitization for user data

### Concurrency Safety
- [ ] Thread-safe where needed
- [ ] No race conditions
- [ ] Proper use of locks/synchronization
- [ ] Deadlock potential minimized

## 7. Maintainability

### Code Clarity
- [ ] Logic is easy to follow
- [ ] No clever tricks without comments
- [ ] Magic numbers replaced with named constants
- [ ] Business logic separated from infrastructure

### Technical Debt
- [ ] No TODO/FIXME without issue tracking
- [ ] Deprecated APIs not used
- [ ] No copy-pasted code (DRY principle)
- [ ] Dependencies up to date and necessary

## 8. Domain-Specific (Quantitative Finance)

### Data Integrity
- [ ] Point-in-time correctness maintained
- [ ] No look-ahead bias in calculations
- [ ] Proper handling of missing/NaN values
- [ ] Time zones handled correctly
- [ ] Market data alignment verified

### Mathematical Correctness
- [ ] Numerical stability considered
- [ ] Division by zero protected
- [ ] Floating-point comparison done safely
- [ ] Statistical calculations verified
- [ ] Edge cases (zero variance, etc.) handled

### Performance Requirements
- [ ] Vectorized operations where possible (NumPy/Pandas)
- [ ] Database queries batched appropriately
- [ ] Caching strategy appropriate
- [ ] Memory usage reasonable for large panels

## 9. Git & Version Control

### Commit Quality
- [ ] Commits are atomic and focused
- [ ] Commit messages clear and descriptive
- [ ] No merge artifacts or debug code
- [ ] Branch up to date with main

### Changes Review
- [ ] Only relevant files modified
- [ ] No unintended file deletions
- [ ] Configuration changes documented
- [ ] Migration scripts included if needed

## 10. Documentation

### User-Facing Documentation
- [ ] README updated if API changed
- [ ] Examples updated for new features
- [ ] Breaking changes clearly documented
- [ ] Migration guide provided if needed

### Developer Documentation
- [ ] Architecture decisions documented
- [ ] Complex algorithms explained
- [ ] Dependencies and setup documented
- [ ] Performance characteristics noted

---

## Priority Levels

### P0 (Must Fix Before Merge)
- Security vulnerabilities
- Data corruption risks
- Critical bugs
- Breaking changes without migration

### P1 (Should Fix Before Merge)
- Code quality issues (complexity, readability)
- Missing error handling
- Performance problems
- Insufficient test coverage

### P2 (Can Be Addressed Later)
- Style inconsistencies
- Minor refactoring opportunities
- Documentation improvements
- Non-critical technical debt

---

## Common Issues to Watch For

### Anti-Patterns
- God classes/functions
- Shotgun surgery (changes scattered across many files)
- Tight coupling
- Circular dependencies
- Mutable default arguments
- Global state

### Code Smells
- Long parameter lists
- Duplicate code
- Feature envy (method uses another class more than its own)
- Data clumps (same group of parameters everywhere)
- Inappropriate intimacy (classes too dependent on each other's internals)

### Python-Specific
- Modifying list while iterating
- Using mutable default arguments
- Not using context managers for resources
- String concatenation in loops (use join)
- Not leveraging standard library

---

## Reviewer Tips

1. **Start broad, then narrow**: Architecture → Design → Implementation → Style
2. **Be specific**: Point to exact lines, suggest concrete improvements
3. **Explain why**: Don't just say "change this", explain the rationale
4. **Positive feedback**: Call out good patterns and clever solutions
5. **Pick your battles**: Not every nit needs to be fixed immediately
6. **Test locally**: For complex changes, check out the branch and run it
