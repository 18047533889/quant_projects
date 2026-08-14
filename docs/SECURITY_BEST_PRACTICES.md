# Security Best Practices

**Version:** 1.0  
**Last Updated:** 2026-08-14  
**Applies To:** FactorEngine, DataAccess, QuantEvaluator, and all platform components

---

## Table of Contents

1. [Secrets Management](#secrets-management)
2. [SQL Injection Prevention](#sql-injection-prevention)
3. [Deserialization Security](#deserialization-security)
4. [Input Validation](#input-validation)
5. [Cryptographic Standards](#cryptographic-standards)
6. [API Security](#api-security)
7. [Path Traversal Prevention](#path-traversal-prevention)
8. [Logging Security](#logging-security)
9. [Dependency Security](#dependency-security)
10. [Code Review Checklist](#code-review-checklist)

---

## Secrets Management

### Never Commit Secrets to Git

**❌ Bad:**
```python
# config.py
GITHUB_TOKEN = "ghp_abc123xyz"
API_KEY = "sk_live_1234567890"
DATABASE_PASSWORD = "mypassword123"
```

**✅ Good:**
```python
# config.py
import os

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
API_KEY = os.environ["API_KEY"]
DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
```

### Use .env Files Correctly

**Project Structure:**
```
quant_projects/
├── .env.example      # ✅ Committed - contains placeholders
├── .env.local        # ✅ Gitignored - contains real secrets
├── .env              # ❌ Should be gitignored
└── .gitignore        # Must include .env and .env.local
```

**.env.example:**
```bash
# GitHub token for API access
GITHUB_TOKEN=your_github_token_here

# LQTP trading platform credentials
LQTP_USERNAME=your_username
LQTP_PASSWORD=your_password
LQTP_SERVER=server_ip:port
```

**.gitignore:**
```
.env
.env.local
.env.*.local
*.secret
*.key
credentials.json
```

### Production Secrets

**Option 1: AWS Secrets Manager**
```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

credentials = get_secret('prod/trading-platform')
password = credentials['password']
```

**Option 2: HashiCorp Vault**
```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.auth.approle.login(role_id=ROLE_ID, secret_id=SECRET_ID)

secret = client.secrets.kv.v2.read_secret_version(
    path='trading/credentials',
    mount_point='secret'
)
password = secret['data']['data']['password']
```

**Option 3: Kubernetes Secrets**
```yaml
# k8s-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-credentials
type: Opaque
data:
  password: <base64-encoded-password>
```

```python
# In pod, read from mounted volume
with open('/var/secrets/password', 'r') as f:
    password = f.read().strip()
```

### Rotating Secrets

**Immediate Actions After Exposure:**
1. Revoke the exposed credential immediately
2. Generate a new credential with different value
3. Update all services using the credential
4. Audit access logs for unauthorized usage
5. Remove from git history if committed:

```bash
# Remove secret from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (coordinate with team first!)
git push origin --force --all
git push origin --force --tags
```

---

## SQL Injection Prevention

### Parameterized Queries

**❌ Bad:**
```python
# String concatenation - VULNERABLE
user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)

# f-string with table name - VULNERABLE
table = request.args.get('table')
query = f"SELECT * FROM {table} WHERE active = 1"
cursor.execute(query)
```

**✅ Good:**
```python
# Parameterized query - SAFE
user_id = request.args.get('id')
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))

# Allowlist for table names - SAFE
ALLOWED_TABLES = {'users', 'orders', 'products'}
table = request.args.get('table')
if table not in ALLOWED_TABLES:
    raise ValueError(f"Invalid table: {table}")
query = f"SELECT * FROM {table} WHERE active = 1"
cursor.execute(query)
```

### SQL Identifier Quoting

**Use the helper function:**
```python
def _quote_ident(name: str) -> str:
    """
    Quote SQL identifier, escaping embedded quotes.
    
    Examples:
        _quote_ident('users') -> '"users"'
        _quote_ident('my"table') -> '"my""table"'
    """
    return f'"{name.replace(chr(34), chr(34) * 2)}"'

# Safe dynamic table/column names
table = _quote_ident(user_table_name)
column = _quote_ident(user_column_name)
query = f"SELECT {column} FROM {table} WHERE id = ?"
cursor.execute(query, (user_id,))
```

### DuckDB-Specific Patterns

**❌ Bad:**
```python
conn.execute(f"SELECT * FROM {table_name}")
conn.execute(f"CREATE INDEX ON {table}({col})")
conn.execute(f"PRAGMA {pragma_name}={value}")
```

**✅ Good:**
```python
# Quote identifiers
conn.execute(f"SELECT * FROM {_quote_ident(table_name)}")
conn.execute(f"CREATE INDEX ON {_quote_ident(table)}({_quote_ident(col)})")

# Allowlist for PRAGMA
ALLOWED_PRAGMAS = {'threads', 'memory_limit', 'temp_directory'}
if pragma_name not in ALLOWED_PRAGMAS:
    raise ValueError(f"Pragma {pragma_name} not allowed")
conn.execute(f"PRAGMA {pragma_name}={int(value)}")
```

### Validation Helper

```python
import re

def validate_sql_identifier(name: str) -> None:
    """
    Validate that a string is a safe SQL identifier.
    
    Allows: letters, digits, underscores
    Must start with letter or underscore
    Max length: 64 characters
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")

# Use before constructing SQL
table_name = request.args.get('table')
validate_sql_identifier(table_name)
query = f"SELECT * FROM {_quote_ident(table_name)}"
```

---

## Deserialization Security

### Pickle Security

**❌ Bad:**
```python
import pickle

# Loading untrusted data - DANGEROUS
with open('user_upload.pkl', 'rb') as f:
    data = pickle.load(f)  # Arbitrary code execution!

# Network data - DANGEROUS
data = pickle.loads(socket.recv(1024))
```

**✅ Good:**
```python
# Use safe formats for untrusted data
import json

with open('user_upload.json', 'r') as f:
    data = json.load(f)  # Safe - no code execution

# For dataframes, use Arrow
import pyarrow as pa

with pa.memory_map('data.arrow', 'r') as source:
    table = pa.ipc.open_file(source).read_all()
```

### Pickle with Integrity Checks

**If pickle is required (internal use only):**
```python
import pickle
import hmac
import hashlib
import os

# Get secret from environment
PICKLE_HMAC_KEY = os.environ['PICKLE_HMAC_KEY'].encode()

def secure_pickle_dump(obj: Any, path: Path) -> str:
    """Pickle object with HMAC integrity check."""
    serialized = pickle.dumps(obj, protocol=5)
    
    # Compute HMAC
    mac = hmac.new(PICKLE_HMAC_KEY, serialized, hashlib.sha256).hexdigest()
    
    # Write data
    path.write_bytes(serialized)
    
    return mac

def secure_pickle_load(path: Path, expected_mac: str) -> Any:
    """Load pickle with integrity verification."""
    serialized = path.read_bytes()
    
    # Verify HMAC
    actual_mac = hmac.new(PICKLE_HMAC_KEY, serialized, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_mac, actual_mac):
        raise ValueError(f"Pickle integrity check failed for {path}")
    
    return pickle.loads(serialized)

# Usage
mac = secure_pickle_dump(my_data, Path('cache.pkl'))
# Store mac in metadata
metadata['checksum'] = mac

# Later
loaded = secure_pickle_load(Path('cache.pkl'), metadata['checksum'])
```

### YAML Security

**❌ Bad:**
```python
import yaml

# Unsafe loader - can instantiate arbitrary objects
config = yaml.load(file, Loader=yaml.Loader)
config = yaml.unsafe_load(file)
```

**✅ Good:**
```python
import yaml

# Safe loader - only basic Python objects
config = yaml.safe_load(file)

# Or use custom loader that inherits from SafeLoader
class StrictYAMLLoader(yaml.SafeLoader):
    """Custom loader with additional restrictions."""
    pass

config = yaml.load(file, Loader=StrictYAMLLoader)
```

### JSON is Usually Better

```python
# JSON doesn't execute code - always safe
import json

config = json.load(file)
data = json.loads(api_response)

# For complex types, use Pydantic
from pydantic import BaseModel

class Config(BaseModel):
    timeout: int
    max_retries: int
    api_key: str

config = Config.model_validate_json(json_string)
```

---

## Input Validation

### API Endpoints

**❌ Bad:**
```python
@app.post("/analyze")
def analyze_factor(request: Request):
    surface = request.json.get('surface')
    market = request.json.get('market')
    
    # No validation - accepts any value
    result = run_analysis(surface=surface, market=market)
    return result
```

**✅ Good:**
```python
from pydantic import BaseModel, validator
from typing import Literal

class AnalysisRequest(BaseModel):
    surface: Literal['daily', 'intraday', 'weekly', 'monthly']
    market: Literal['ashare', 'us', 'hk']
    lookback: int
    
    @validator('lookback')
    def validate_lookback(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('lookback must be between 1 and 1000')
        return v

@app.post("/analyze")
def analyze_factor(request: AnalysisRequest):
    # Input automatically validated by Pydantic
    result = run_analysis(
        surface=request.surface,
        market=request.market,
        lookback=request.lookback
    )
    return result
```

### Allowlists vs Denylists

**❌ Bad (Denylist):**
```python
# Trying to block bad values - incomplete
FORBIDDEN_SURFACES = ['../../../etc', 'admin', 'root']

if surface not in FORBIDDEN_SURFACES:
    load_surface_config(surface)  # Still vulnerable!
```

**✅ Good (Allowlist):**
```python
# Only allow known-good values
ALLOWED_SURFACES = {'daily', 'intraday', 'weekly', 'monthly'}

if surface not in ALLOWED_SURFACES:
    raise ValueError(f"Invalid surface: {surface}")

load_surface_config(surface)  # Safe
```

### Numeric Range Validation

```python
from pydantic import BaseModel, Field

class FactorParams(BaseModel):
    window: int = Field(ge=1, le=252, description="Window size in days")
    alpha: float = Field(ge=0.0, le=1.0, description="Decay factor")
    min_periods: int = Field(ge=1, le=100, description="Minimum observations")
    
    class Config:
        extra = 'forbid'  # Reject unknown fields

# Usage
params = FactorParams(window=20, alpha=0.05, min_periods=10)  # OK
params = FactorParams(window=999, alpha=0.05, min_periods=10)  # ValidationError
```

### String Pattern Validation

```python
import re
from pydantic import validator

class FactorId(BaseModel):
    id: str
    
    @validator('id')
    def validate_id(cls, v):
        # Only alphanumeric, underscores, hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid factor ID format')
        if len(v) > 64:
            raise ValueError('Factor ID too long')
        return v

# Usage
factor = FactorId(id='momentum_20d')  # OK
factor = FactorId(id='../../etc/passwd')  # ValidationError
```

---

## Cryptographic Standards

### Hash Functions

**For Security (integrity, authentication):**
```python
import hashlib

# ✅ Use SHA-256 or stronger
digest = hashlib.sha256(data).hexdigest()
digest = hashlib.sha512(data).hexdigest()
digest = hashlib.blake2b(data, digest_size=32).hexdigest()
```

**For Cache Keys (non-security):**
```python
# ✅ Explicitly mark as non-security
# Python 3.9+
digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()

# Add comment explaining usage
# This hash is used only for cache key generation, not integrity verification
cache_key = hashlib.md5(params_json.encode(), usedforsecurity=False).hexdigest()
```

### HMAC for Message Authentication

```python
import hmac
import hashlib
import os

# Secret key from environment
SECRET_KEY = os.environ['HMAC_SECRET_KEY'].encode()

def sign_message(message: bytes) -> str:
    """Generate HMAC-SHA256 signature."""
    return hmac.new(SECRET_KEY, message, hashlib.sha256).hexdigest()

def verify_signature(message: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 signature using constant-time comparison."""
    expected = sign_message(message)
    return hmac.compare_digest(expected, signature)

# Usage
message = b"transfer $1000 to account 12345"
signature = sign_message(message)

# Later, verify
if verify_signature(message, received_signature):
    process_transaction(message)
else:
    raise ValueError("Invalid signature")
```

### Random Number Generation

**❌ Bad:**
```python
import random

# Predictable - NOT for security
token = ''.join(random.choices('abcdefg0123456789', k=32))
```

**✅ Good:**
```python
import secrets

# Cryptographically secure
token = secrets.token_hex(32)  # 64-character hex string
token = secrets.token_urlsafe(32)  # URL-safe string
random_int = secrets.randbelow(100)  # Random int in [0, 100)
```

---

## API Security

### Authentication

```python
from fastapi import Header, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify JWT token and return claims."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            os.environ['JWT_SECRET_KEY'],
            algorithms=['HS256']
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/factors")
def list_factors(user: dict = Depends(verify_token)):
    # user contains decoded JWT claims
    return get_factors_for_user(user['user_id'])
```

### Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("10/minute")
def analyze_factor(request: Request, params: AnalysisRequest):
    return run_analysis(params)
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
        "https://dashboard.example.com"
    ],  # ❌ Never use ["*"] in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Security Headers

```python
from fastapi import Response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response
```

---

## Path Traversal Prevention

### File Access Validation

**❌ Bad:**
```python
@app.get("/config/{filename}")
def get_config(filename: str):
    # Path traversal vulnerability
    path = Path(f"configs/{filename}")
    return path.read_text()

# Attack: GET /config/../../etc/passwd
```

**✅ Good:**
```python
from pathlib import Path

CONFIG_ROOT = Path("/app/configs").resolve()

@app.get("/config/{filename}")
def get_config(filename: str):
    # Validate filename doesn't contain path separators
    if '/' in filename or '\\' in filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Resolve full path and check it's under CONFIG_ROOT
    requested = (CONFIG_ROOT / filename).resolve()
    
    if not requested.is_relative_to(CONFIG_ROOT):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Not found")
    
    return requested.read_text()
```

### Symlink Protection

```python
def safe_path(base: Path, relative: str) -> Path:
    """
    Safely resolve a path relative to base, preventing symlink escapes.
    
    Args:
        base: Base directory (must be absolute)
        relative: Relative path from user input
        
    Returns:
        Resolved path if safe
        
    Raises:
        ValueError: If path escapes base or contains symlinks
    """
    if not base.is_absolute():
        raise ValueError("Base path must be absolute")
    
    # Check for obvious path traversal
    if '..' in Path(relative).parts:
        raise ValueError("Path contains '..' component")
    
    # Resolve path
    requested = (base / relative).resolve()
    
    # Check symlinks in the path
    current = requested
    while current != base:
        if current.is_symlink():
            raise ValueError(f"Symlink detected: {current}")
        current = current.parent
    
    # Check it's under base
    if not requested.is_relative_to(base):
        raise ValueError("Path escapes base directory")
    
    return requested
```

---

## Logging Security

### Redacting Secrets

```python
import logging
import re

class SensitiveFilter(logging.Filter):
    """Filter that redacts sensitive information from log messages."""
    
    SENSITIVE_PATTERNS = [
        (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.I), 'password'),
        (re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.I), 'token'),
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.I), 'api_key'),
        (re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.I), 'secret'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        
        for pattern, name in self.SENSITIVE_PATTERNS:
            message = pattern.sub(f'{name}=***REDACTED***', message)
        
        record.msg = message
        record.args = ()
        return True

# Configure logging
logger = logging.getLogger('factor_engine')
logger.addFilter(SensitiveFilter())

# Usage
logger.info(f"Connecting with password={password}")
# Logged as: "Connecting with password=***REDACTED***"
```

### Structured Logging

```python
import structlog

# Configure structured logging with redaction
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_sensitive_fields,  # Custom processor
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Log with structured data
logger.info(
    "user_login",
    user_id=user.id,
    username=user.username,
    # ❌ Never log passwords/tokens
    ip_address=request.remote_addr
)
```

### What NOT to Log

**❌ Never log:**
- Passwords, tokens, API keys
- Credit card numbers, SSNs
- Private keys, certificates
- Full SQL queries with parameter values
- Raw user input before validation
- Session IDs, cookies

**✅ Safe to log:**
- User IDs (not usernames if PII concern)
- Request IDs, trace IDs
- Error types and stack traces (without sensitive data in variables)
- Performance metrics
- Business events (factor calculation, trade execution)

---

## Dependency Security

### Keep Dependencies Updated

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade package_name
```

### Pin Dependencies

**requirements.txt:**
```
# ✅ Pin exact versions for reproducibility
pandas==2.0.3
numpy==1.24.3
pyarrow==12.0.1

# ❌ Don't use unpinned versions in production
# pandas>=2.0
# numpy
```

### Audit New Dependencies

Before adding a new dependency:
1. Check GitHub stars and maintenance activity
2. Review open security issues
3. Check download statistics on PyPI
4. Scan with `safety check` or Snyk
5. Review the license

```bash
# Check package info
pip show package_name

# View dependencies
pipdeptree -p package_name
```

### Private Package Security

```python
# ❌ Bad - hardcoded token in requirements.txt
git+https://ghp_token@github.com/company/private-repo.git

# ✅ Good - use credential helper
git+https://${GITHUB_TOKEN}@github.com/company/private-repo.git
```

---

## Code Review Checklist

### Security Review Questions

When reviewing code, ask:

**Secrets:**
- [ ] Are there any hardcoded credentials?
- [ ] Are secrets loaded from environment variables or secrets manager?
- [ ] Is the `.env` file in `.gitignore`?

**SQL Injection:**
- [ ] Are SQL queries parameterized?
- [ ] Are table/column names from user input quoted?
- [ ] Is there an allowlist for dynamic identifiers?

**Deserialization:**
- [ ] Is `pickle.load()` used on untrusted data?
- [ ] Is there integrity verification for pickle files?
- [ ] Can safer formats (JSON, Arrow) be used instead?

**Input Validation:**
- [ ] Are API inputs validated with Pydantic or similar?
- [ ] Are there allowlists for enum-like parameters?
- [ ] Are numeric inputs range-checked?

**Path Traversal:**
- [ ] Are file paths validated against a base directory?
- [ ] Are symlinks checked and rejected?
- [ ] Are path separators and `..` blocked?

**Logging:**
- [ ] Are passwords/tokens excluded from logs?
- [ ] Is the `SensitiveFilter` applied to loggers?
- [ ] Are log levels appropriate (no DEBUG in production)?

**Cryptography:**
- [ ] Is SHA-256+ used for security contexts?
- [ ] Is `usedforsecurity=False` set for cache key hashes?
- [ ] Is `secrets` module used for token generation?

**Dependencies:**
- [ ] Are all dependencies pinned to specific versions?
- [ ] Have new dependencies been security-scanned?
- [ ] Are dependencies from trusted sources?

---

## Automated Security Checks

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-ll', '--recursive', '--exclude', 'tests']
        
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### CI/CD Pipeline

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r . -ll -f json -o bandit-report.json
          
      - name: Run Safety
        run: |
          pip install safety
          safety check --json
          
      - name: Upload Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: bandit-report.json
```

---

## Incident Response

### If a Secret is Exposed

1. **Immediate Revocation:**
   - Revoke the exposed credential immediately
   - Generate a new credential with a different value

2. **Impact Assessment:**
   - Check access logs for unauthorized usage
   - Identify all services using the credential
   - Determine if data was accessed or modified

3. **Remediation:**
   - Update all services with new credential
   - Remove secret from git history if committed
   - Implement additional monitoring

4. **Prevention:**
   - Add pre-commit hooks to prevent future commits
   - Document incident and lessons learned
   - Update security training materials

### Reporting Security Issues

**Internal:** security@company.com  
**External Researchers:** security-reports@company.com

**Include in report:**
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation

---

## Training Resources

- **OWASP Top 10:** https://owasp.org/www-project-top-ten/
- **Python Security:** https://python.readthedocs.io/en/stable/library/security_warnings.html
- **Bandit Documentation:** https://bandit.readthedocs.io/
- **CWE Database:** https://cwe.mitre.org/

---

*Last updated: 2026-08-14*  
*Maintained by: Platform Security Team*
