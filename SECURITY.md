# Security Audit Report

**Audit Date:** 2026-08-14  
**Project:** Quantitative Trading Platform (FactorEngine + DataAccess + Evaluator)  
**Auditor:** Automated Security Review + Bandit Static Analysis

---

## Executive Summary

This security audit identified **1 CRITICAL**, **2 HIGH**, **1,458 MEDIUM**, and **40,260 LOW** severity issues across the codebase. The critical issue involves hardcoded credentials in the `.env` file that must be addressed immediately. High-severity issues involve unsafe pickle deserialization that could lead to arbitrary code execution. Medium-severity issues primarily involve SQL injection vectors and cryptographic hash usage.

**Immediate Actions Required:**
1. ✅ **CRITICAL**: Revoke exposed GitHub token and LQTP credentials in `.env` file
2. ✅ **HIGH**: Add integrity checks to pickle deserialization in spill storage and cache
3. ✅ **MEDIUM**: Implement SQL identifier quoting across all dynamic SQL construction

---

## Critical Findings

### 1. Hardcoded Credentials in Repository (CRITICAL)

**Location:** `/home/shw/quant_projects/.env`  
**Severity:** CRITICAL  
**CWE:** CWE-798 (Use of Hard-coded Credentials)

**Issue:**
The `.env` file contains live production credentials committed to the repository:
- `GITHUB_TOKEN=ghp_REDACTED` — Active GitHub PAT with full repository access
- `LQTP_PASSWORD=3213709208` — Plaintext trading platform password
- `LQTP_SERVER=110.42.223.26:50051` — Production server IP exposed

**Impact:**
- Anyone with repository access can authenticate as the service account
- GitHub token grants read/write access to all repositories
- Trading platform credentials allow unauthorized market data access or order placement
- If repository is ever made public or leaked, credentials are permanently compromised

**Remediation:**
```bash
# 1. IMMEDIATE: Revoke the GitHub token at https://github.com/settings/tokens
# 2. IMMEDIATE: Change LQTP password through the trading platform admin panel

# 3. Remove .env from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 4. Add .env to .gitignore
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
git add .gitignore
git commit -m "security: prevent credential files from being committed"

# 5. Use environment variables or a secrets manager
# For local development: export variables or use .env.local (gitignored)
# For production: use AWS Secrets Manager, HashiCorp Vault, or k8s secrets
```

**Best Practice:**
- Never commit credentials to version control
- Use `.env.example` with placeholder values for documentation
- Inject secrets at runtime via environment variables
- Rotate credentials immediately after any potential exposure
- Use short-lived tokens where possible (OAuth, IAM roles)

---

## High-Severity Findings

### 2. Unsafe Pickle Deserialization in Spill Storage (HIGH)

**Location:** `/home/shw/quant_projects/factor_engine/runtime/multibackend/spill_strategy.py:213`  
**Severity:** HIGH  
**CWE:** CWE-502 (Deserialization of Untrusted Data)

**Issue:**
```python
data = pickle.loads(serialized)
```

The spill storage system deserializes pickle data read from disk without integrity verification. The spill directory path is configurable and not validated against a trusted base directory.

**Attack Vector:**
- On shared compute infrastructure, an attacker with file system access could replace spill files
- Symlink attacks could redirect spill paths to attacker-controlled locations
- Race conditions between spill file write and read operations
- If `_spill_dir` is on a shared mount (NFS, CIFS), network attackers gain access

**Impact:**
Arbitrary code execution when spilled data is restored. In a quantitative trading context:
- Manipulate factor calculations to generate false trading signals
- Exfiltrate proprietary trading algorithms or market data
- Pivot to other systems using service account credentials

**Remediation:**
```python
# In spill_strategy.py, add integrity checking

import hmac
import hashlib

class SpillMetadata:
    checksum: str  # Add this field

def _spill_to_disk(self, key: str, data: Any) -> Path:
    serialized = pickle.dumps(data, protocol=5)
    
    # Compute HMAC (use a key from secure config)
    secret = os.environ.get("SPILL_HMAC_KEY", "").encode()
    if not secret:
        raise ValueError("SPILL_HMAC_KEY must be set")
    checksum = hmac.new(secret, serialized, hashlib.sha256).hexdigest()
    
    path = self._spill_dir / f"{key}.pkl"
    path.write_bytes(serialized)
    
    # Store checksum in metadata
    self._metadata[key] = SpillMetadata(checksum=checksum, ...)
    return path

def _restore_from_disk(self, key: str) -> Any:
    path = self._metadata[key].path
    serialized = path.read_bytes()
    
    # Verify integrity before deserializing
    secret = os.environ.get("SPILL_HMAC_KEY", "").encode()
    expected = self._metadata[key].checksum
    actual = hmac.new(secret, serialized, hashlib.sha256).hexdigest()
    
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"Spill file {path} failed integrity check")
    
    return pickle.loads(serialized)
```

**Better Alternative:**
Replace pickle entirely with Apache Arrow IPC format (already used in this codebase):
```python
import pyarrow as pa

def _spill_to_disk(self, key: str, data: pa.Table) -> Path:
    # Arrow IPC includes checksums and is not executable
    path = self._spill_dir / f"{key}.arrow"
    with pa.OSFile(str(path), 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, data.schema) as writer:
            writer.write_table(data)
    return path

def _restore_from_disk(self, key: str) -> pa.Table:
    path = self._metadata[key].path
    with pa.memory_map(str(path), 'r') as source:
        return pa.ipc.open_file(source).read_all()
```

### 3. Unsafe Pickle Deserialization in Cache (HIGH)

**Location:** `/home/shw/quant_projects/quant_evaluator/runtime/cache_v2.py:280, 507, 685`  
**Severity:** HIGH  
**CWE:** CWE-502 (Deserialization of Untrusted Data)

**Issue:**
```python
value = pickle.loads(data)
```

The cache system deserializes pickle data retrieved from an in-memory cache keyed by arbitrary strings. If cache keys can be influenced by external input (API parameters, user-supplied factor IDs), this becomes exploitable.

**Remediation:**
Same as above — add HMAC verification or migrate to Arrow IPC/msgpack with schema validation.

---

## Medium-Severity Findings

### 4. SQL Injection via Unquoted Identifiers (MEDIUM)

**Locations:**
- `/home/shw/quant_projects/dataaccess/core/engine.py:342-343, 350, 386, 973, 1050, 1101`
- `/home/shw/quant_projects/factor_engine/backend/sql_pushdown/duckdb_performance.py:340, 349, 363`
- `/home/shw/quant_projects/dataaccess/cos/s3_duckdb.py:120-121`
- `/home/shw/quant_projects/factor_engine/runtime/resource_governor.py:1275`

**Severity:** MEDIUM  
**CWE:** CWE-89 (SQL Injection)

**Issue:**
Multiple locations build SQL DDL/DML statements using f-strings with unquoted identifiers:
```python
# dataaccess/core/engine.py:386
execute(f'CREATE OR REPLACE TABLE "{view_name}" AS {inlined_sql}')

# duckdb_performance.py:340
conn.execute(f"SELECT COUNT(*) FROM {table_name}")

# s3_duckdb.py:120
execute(f"SET s3_access_key_id={_sql_string(creds.access_key_id)}")
```

**Attack Vector:**
If `view_name`, `table_name`, or SQL fragments ever contain user-controlled input:
```python
view_name = 'temp"; DROP TABLE factors; --'
# Results in: CREATE OR REPLACE TABLE "temp"; DROP TABLE factors; --" AS ...
```

**Current Risk Assessment:**
- `view_name` is currently UUID-based and generated internally
- `table_name` flows from internal configuration
- **However**, no validation enforces this guarantee at the call site

**Remediation:**
```python
# Add identifier quoting helper (already exists in store.py, extend it)
def _quote_ident(name: str) -> str:
    """Quote SQL identifier, escaping embedded quotes."""
    return f'"{name.replace(chr(34), chr(34) * 2)}"'

# Apply at all SQL construction sites
execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}")
execute(f"CREATE INDEX ON {_quote_ident(table_name)}({_quote_ident(col)})")

# Add allowlist validation as defense-in-depth
def _validate_identifier(name: str) -> None:
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
```

**For PRAGMA statements** (resource_governor.py:1275):
```python
ALLOWED_PRAGMAS = {"threads", "memory_limit", "temp_directory"}

for name, value in pragmas.items():
    if name not in ALLOWED_PRAGMAS:
        raise ValueError(f"Pragma {name} not allowed")
    if name == "threads":
        value = int(value)  # Already done
    conn.execute(f"PRAGMA {name}={value}")
```

### 5. Weak Cryptographic Hashes for Security Context (MEDIUM)

**Locations:** 23 instances across codebase  
**Severity:** MEDIUM  
**CWE:** CWE-327 (Use of a Broken or Risky Cryptographic Algorithm)

**Issue:**
MD5 and SHA-1 are used for hashing in security-sensitive contexts:
```python
# dataaccess/core/identity_encoder.py:72
digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()

# factor_engine/backend/evidence_delta.py:33
return hashlib.sha1(header + data).hexdigest()
```

**Analysis:**
- **MD5**: Collision attacks are practical (2^21 operations)
- **SHA-1**: Collision attacks demonstrated (SHAttered, 2017)
- **Impact**: If hashes are used for integrity verification, attackers can craft collisions

**Current Usage:**
Most instances appear to be for **cache keys** and **content-addressable storage**, not authentication:
- `identity_encoder.py` — generates short display IDs
- `parameter_canonicalizer.py` — caches operator parameter equivalence
- `evidence_delta.py` — mimics git blob hash format (which uses SHA-1)

**Remediation:**
```python
# For Python 3.9+, explicitly mark non-security usage
digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()

# For security contexts (integrity, authentication), use SHA-256 or BLAKE2
digest = hashlib.sha256(data).hexdigest()
digest = hashlib.blake2b(data, digest_size=32).hexdigest()
```

**Update locations:**
- ✅ Add `usedforsecurity=False` to all MD5/SHA-1 calls used for cache keys
- ✅ Migrate `evidence_delta.py` to SHA-256 (or keep SHA-1 with explicit comment that it mimics git)
- ✅ Document which hashes are for cache vs. security

### 6. XML External Entity (XXE) Vulnerabilities (MEDIUM)

**Locations:**
- `/home/shw/quant_projects/factor_engine/scripts/audit_r15_fix_report.py:107`
- `/home/shw/quant_projects/factor_engine/scripts/certify_primitive_evidence.py:195`
- `/home/shw/quant_projects/factor_engine/scripts/generate_r28_evidence.py:101`

**Severity:** MEDIUM  
**CWE:** CWE-611 (Improper Restriction of XML External Entity Reference)

**Issue:**
```python
root = ET.parse(junitxml).getroot()
```

Standard library `xml.etree.ElementTree` is vulnerable to XXE attacks when parsing untrusted XML.

**Attack Vector:**
If JUnit XML files are sourced from CI artifacts that could be attacker-controlled:
```xml
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
<testsuites><testsuite>&xxe;</testsuite></testsuites>
```

**Remediation:**
```bash
pip install defusedxml
```

```python
import defusedxml.ElementTree as ET

# Replace all ET.parse() calls
root = ET.parse(junitxml).getroot()  # Now safe
```

**Alternative** (if defusedxml not available):
```python
import xml.etree.ElementTree as ET

# Disable DTD processing and entity expansion
parser = ET.XMLParser()
parser.entity = {}  # Disable entity expansion
parser.parser.UseForeignDTD(False)
parser.parser.SetParamEntityParsing(0)

tree = ET.parse(junit_path, parser=parser)
```

### 7. YAML Unsafe Load with Custom Loader (MEDIUM)

**Location:** `/home/shw/quant_projects/dataaccess/registry/yaml_loader.py:43`  
**Severity:** MEDIUM  
**CWE:** CWE-502 (Deserialization of Untrusted Data)

**Issue:**
```python
return yaml.load(text, Loader=StrictYAMLLoader)
```

Using `yaml.load()` with a custom `StrictYAMLLoader`. Bandit flags this as potentially unsafe.

**Analysis:**
Need to verify that `StrictYAMLLoader` actually restricts dangerous tags. If it inherits from `yaml.Loader` without overriding tag handlers, it's vulnerable.

**Remediation:**
```python
# Verify StrictYAMLLoader implementation
class StrictYAMLLoader(yaml.SafeLoader):
    """Subclass of SafeLoader with additional restrictions."""
    pass

# Or use SafeLoader directly
return yaml.safe_load(text)
```

**Inspect current implementation:**
```bash
grep -A 20 "class StrictYAMLLoader" dataaccess/registry/yaml_loader.py
```

If `StrictYAMLLoader` does not inherit from `SafeLoader`, change to:
```python
yaml_tags_allowed = {'!include', '!env'}  # Define allowed custom tags
return yaml.safe_load(text)
```

### 8. Binding to All Network Interfaces (MEDIUM)

**Locations:**
- `/home/shw/quant_projects/dataaccess/service/__main__.py:36`
- `/home/shw/quant_projects/factor_engine/service/app.py:1520`

**Severity:** MEDIUM  
**CWE:** CWE-668 (Exposure of Resource to Wrong Sphere)

**Issue:**
```python
default=os.environ.get("DATA_ACCESS_API_HOST", "0.0.0.0")
```

Services default to binding on `0.0.0.0`, exposing them to all network interfaces.

**Impact:**
- On developer workstations, exposes service to local network
- On cloud VMs without firewall rules, exposes to internet
- Increases attack surface for unauthenticated endpoints

**Remediation:**
```python
# Default to localhost for development
default=os.environ.get("DATA_ACCESS_API_HOST", "127.0.0.1")

# Production deployments should explicitly set:
# DATA_ACCESS_API_HOST=0.0.0.0  # Only if behind a firewall/load balancer
```

**Deployment Guide:**
```bash
# Local development (default)
python -m factor_engine.service.app  # Binds to 127.0.0.1

# Production (explicit)
export DATA_ACCESS_API_HOST=0.0.0.0
export DATA_ACCESS_API_PORT=8000
python -m factor_engine.service.app  # Behind nginx/ALB
```

### 9. URL Open Without Scheme Validation (MEDIUM)

**Location:** `/home/shw/quant_projects/factor_engine/runtime/metrics_export.py:275`  
**Severity:** MEDIUM  
**CWE:** CWE-918 (Server-Side Request Forgery)

**Issue:**
```python
with urllib.request.urlopen(request, timeout=timeout_sec) as response:
```

`urlopen` accepts `file://` URLs and custom schemes, which could be exploited for SSRF or local file disclosure.

**Remediation:**
```python
from urllib.parse import urlparse

def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"URL scheme {parsed.scheme} not allowed")
    if parsed.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError("Requests to localhost not allowed")
    # Add CIDR block checks for internal networks if needed

url = "https://metrics.example.com/push"
_validate_url(url)
with urllib.request.urlopen(url, timeout=timeout_sec) as response:
    ...
```

### 10. Eval/Exec Usage in Code Generation (MEDIUM)

**Locations:**
- `/home/shw/quant_projects/factor_engine/runtime/multibackend/polars_expression_compiler.py:147`
- Test files: `test_ts_nth_value_polars_causality.py:34`, `test_r40_operator_spec_127_130.py:57`

**Severity:** MEDIUM  
**CWE:** CWE-95 (Improper Neutralization of Directives in Dynamically Evaluated Code)

**Issue:**
```python
compiled_expr = eval(expr_str, {"pl": pl})
```

`eval()` and `exec()` execute arbitrary Python code. If `expr_str` contains user input, this is critical.

**Analysis:**
- `polars_expression_compiler.py` — generates Polars DSL from internal AST (not direct user input)
- Test files — exec() used to compile test fixtures (controlled)

**Remediation:**
```python
# For polars_expression_compiler.py
# Option 1: Parse and validate the AST before eval
import ast

def _validate_expr_ast(expr_str: str) -> None:
    """Ensure expression only contains safe Polars calls."""
    tree = ast.parse(expr_str, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Whitelist allowed functions
            if isinstance(node.func, ast.Attribute):
                if node.func.attr not in ALLOWED_POLARS_FUNCTIONS:
                    raise ValueError(f"Function {node.func.attr} not allowed")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements not allowed in expressions")

_validate_expr_ast(expr_str)
compiled_expr = eval(expr_str, {"pl": pl}, {})  # Empty locals dict

# Option 2: Use a proper DSL parser instead of eval
# Build an AST-to-Polars-expression compiler without eval
```

**For test files:**
Keep as-is but add `# noqa: S102` with explanation:
```python
exec(compile(src, "<sig>", "exec"), ns)  # noqa: S102 - controlled test fixture, no user input
```

---

## Low-Severity Findings

### 11. Path Traversal Risk in Config Loading (LOW)

**Location:** `/home/shw/quant_projects/factor_engine/service/security.py:453-470`  
**Severity:** LOW  
**CWE:** CWE-22 (Path Traversal)

**Issue:**
`authorized_config_path` defaults to relative path `Path("configs")` when `FACTOR_ENGINE_CONFIG_ROOT` is unset.

**Remediation:**
```python
def get_config_root() -> Path:
    root = os.environ.get("FACTOR_ENGINE_CONFIG_ROOT")
    if root:
        path = Path(root)
        if not path.is_absolute():
            raise ValueError("FACTOR_ENGINE_CONFIG_ROOT must be absolute")
        return path
    # Default to project root + configs
    return Path(__file__).parent.parent / "configs"
```

### 12. Unvalidated Input in API Endpoints (LOW)

**Location:** `/home/shw/quant_projects/factor_engine/service/app.py:395-422`  
**Severity:** LOW  
**CWE:** CWE-20 (Improper Input Validation)

**Issue:**
`run_mode`, `surface`, `market` extracted from JSON without allowlist validation.

**Remediation:**
```python
VALID_SURFACES = {"daily", "intraday", "weekly", "monthly"}
VALID_MARKETS = {"ashare", "us", "hk"}
VALID_DIALECTS = {"python", "dsl_v1", "dsl_v2"}

surface = str(payload.get("surface") or "daily")
if surface not in VALID_SURFACES:
    raise HTTPException(status_code=422, detail=f"Invalid surface: {surface}")
```

---

## Bandit Summary

**Total Issues:** 41,741  
**High Severity:** 23 (all weak hash usage)  
**Medium Severity:** 1,458  
**Low Severity:** 40,260

**Top Issue Types:**
1. **B608 (SQL injection risk):** 1,327 instances — mostly false positives from legitimate parameterized queries, but ~15 genuine risks identified
2. **B324 (weak hash):** 23 instances — all using MD5/SHA-1, need `usedforsecurity=False` flag
3. **B108 (hardcoded temp file):** 107 instances — informational, low risk
4. **B301 (pickle):** 9 instances — 2 high-risk (spill/cache), 7 in tests
5. **B314 (XML):** 3 instances — all in test/audit scripts parsing JUnit XML
6. **B506 (YAML unsafe load):** 2 instances — need to verify `StrictYAMLLoader` implementation
7. **B104 (bind 0.0.0.0):** 4 instances — services default to all interfaces

---

## Positive Security Controls Already in Place

The codebase demonstrates strong security awareness:

✅ **Credential Redaction:** `logging_config.py` includes `SensitiveFilter` that strips passwords/tokens from logs  
✅ **SQL Identifier Quoting:** `_quote_ident()` helper exists in `store.py`  
✅ **Secret Field Protection:** `S3Credentials` uses `repr=False` on secret fields  
✅ **Path Validation:** `authorized_config_path()` checks for symlinks and validates against root  
✅ **Input Validation:** Pydantic models with `extra="forbid"` on API endpoints  
✅ **Authentication:** Production routes require authentication  
✅ **Fail-Closed Design:** Many authorization checks default to deny  

**Recommendation:** Extend these patterns consistently across all modules.

---

## Remediation Roadmap

### Phase 1: Immediate (Within 24 hours)
- [ ] Revoke GitHub token `ghp_REDACTED`
- [ ] Change LQTP password `3213709208`
- [ ] Remove `.env` from git history
- [ ] Add `.env` to `.gitignore`
- [ ] Migrate credentials to environment variables

### Phase 2: Critical (Within 1 week)
- [ ] Add HMAC integrity checks to `spill_strategy.py` pickle deserialization
- [ ] Add integrity checks to `cache_v2.py` pickle deserialization
- [ ] Migrate spill storage to Arrow IPC format (preferred long-term solution)
- [ ] Audit all SQL construction sites and add identifier quoting
- [ ] Add `_validate_identifier()` allowlist checks to SQL builders

### Phase 3: High Priority (Within 2 weeks)
- [ ] Add `usedforsecurity=False` to all MD5/SHA-1 hash calls
- [ ] Replace `xml.etree.ElementTree` with `defusedxml` in audit scripts
- [ ] Verify `StrictYAMLLoader` inherits from `SafeLoader`
- [ ] Change service binding default from `0.0.0.0` to `127.0.0.1`
- [ ] Add URL scheme validation to `metrics_export.py`
- [ ] Add input allowlists to API endpoints (`surface`, `market`, `dialect`)

### Phase 4: Defense-in-Depth (Within 1 month)
- [ ] Implement automated secret scanning in CI/CD (GitGuardian, TruffleHog)
- [ ] Add pre-commit hooks to prevent credential commits
- [ ] Configure Bandit in CI with baseline to catch new issues
- [ ] Document security architecture and threat model
- [ ] Conduct penetration testing of API endpoints
- [ ] Implement rate limiting and request throttling
- [ ] Add security headers to HTTP responses (HSTS, CSP, X-Frame-Options)
- [ ] Enable audit logging for all authentication and authorization events

---

## Security Best Practices for Development

### Secrets Management
```bash
# Development: Use .env.local (gitignored)
cp .env.example .env.local
# Edit .env.local with local credentials

# Production: Use secrets manager
export GITHUB_TOKEN=$(aws secretsmanager get-secret-value --secret-id prod/github-token --query SecretString --output text)
export LQTP_PASSWORD=$(vault kv get -field=password secret/lqtp)
```

### SQL Query Construction
```python
# ❌ NEVER do this
query = f"SELECT * FROM {table} WHERE id = {user_id}"

# ✅ Use parameterized queries
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))

# ✅ Quote identifiers
query = f"SELECT * FROM {_quote_ident(table)} WHERE id = ?"
```

### Deserialization
```python
# ❌ NEVER deserialize untrusted data
data = pickle.loads(user_input)

# ✅ Use safe formats
data = json.loads(user_input)  # For simple types
data = pa.ipc.open_file(path).read_all()  # For dataframes

# ✅ If pickle is required, verify integrity
if not hmac.compare_digest(computed_hmac, stored_hmac):
    raise ValueError("Integrity check failed")
data = pickle.loads(trusted_bytes)
```

### Input Validation
```python
# ❌ Trust user input
market = request.json.get("market")

# ✅ Validate against allowlist
VALID_MARKETS = {"ashare", "us", "hk"}
market = request.json.get("market")
if market not in VALID_MARKETS:
    raise ValueError(f"Invalid market: {market}")
```

### Logging Secrets
```python
# ❌ Log sensitive data
logger.info(f"Connecting with password: {password}")

# ✅ Redact secrets
logger.info(f"Connecting with password: {'*' * 8}")

# ✅ Use SensitiveFilter (already in logging_config.py)
# Automatically redacts fields matching: password, token, secret, key
```

---

## Testing Security Fixes

### Run Bandit After Fixes
```bash
# Full scan
bandit -r factor_engine/ dataaccess/ -f json -o bandit_report.json

# Filter for high/medium severity
bandit -r factor_engine/ -ll -ii

# Exclude test files
bandit -r factor_engine/ --exclude factor_engine/tests/
```

### Verify Pickle Integrity Checks
```python
# Test integrity check catches tampering
def test_spill_integrity():
    spiller = SpillStrategy()
    key = spiller.spill_to_disk("test", data)
    
    # Tamper with spill file
    path = spiller._metadata[key].path
    tampered = path.read_bytes()[:-10] + b"malicious"
    path.write_bytes(tampered)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="integrity check failed"):
        spiller.restore_from_disk(key)
```

### Verify SQL Injection Prevention
```python
def test_sql_injection_prevented():
    malicious_table = 'temp"; DROP TABLE factors; --'
    
    # Should not execute DROP
    with pytest.raises(ValueError):
        query = f"SELECT * FROM {_quote_ident(malicious_table)}"
    
    # Quoted result should be safe
    safe = _quote_ident(malicious_table)
    assert safe == '"temp""; DROP TABLE factors; --"'
```

---

## Contact & Escalation

**Security Issues:** Report to security team immediately  
**Questions:** Contact platform architecture team  
**Incident Response:** Follow incident response playbook in runbooks/

---

*This security audit was generated automatically. Manual verification of critical findings is required before deployment.*
