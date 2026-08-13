# ResourceBroker Audit Findings - 2026-08-14

## Critical Issues Found

### Issue 1: Duplicate task_id Admission (CRITICAL)
**Location**: `resource_broker.py:713-736` (`try_reserve`)

**Problem**: 
- Line 732 directly assigns `self._running[tid] = task` without checking if `tid` already exists
- If same `task_id` is reserved twice:
  - CPU/IO tokens are acquired twice (lines 729, 731)
  - Only one entry in `self._running` (dict overwrites)
  - On single release: only releases tokens once → **token leak**
  - Memory accounting corrupted: `sum(self._running.values())` undercounts actual allocations

**Exploit scenario**:
```python
broker.try_reserve(task, task_id="job123")  # Acquires 4 CPU tokens
broker.try_reserve(task, task_id="job123")  # Acquires 4 MORE tokens, overwrites _running
# Now: 8 tokens acquired, _running shows 1 task, tokens leak on single release
```

**Impact**: Token exhaustion, scheduler deadlock when all tokens leak

---

### Issue 2: Negative Token Accounting in _release_locked (HIGH)
**Location**: `resource_broker.py:738-742`

**Problem**:
```python
def _release_locked(self, task: TaskResourceContract, *, task_id: str) -> None:
    self._running.pop(task_id, None)  # Returns None if not present
    self._cpu.release(task.cpu_tokens)  # Always releases
    self._io.release(task.io_tokens)
```

- `_running.pop(task_id, None)` silently succeeds even if `task_id` not present
- Tokens are released **regardless** of whether task was in `_running`
- Repeated calls drive token counters negative: `self.in_use = max(0, self.in_use - tokens)` masks the bug but corrupts accounting

**Exploit scenario**:
```python
broker._release_locked(task, task_id="ghost")  # Not in _running
# CPU/IO tokens still released -> accounting drift
```

**Impact**: Token counters become meaningless, admission decisions wrong

---

### Issue 3: reserve() Backward Compat TOCTOU Race (MEDIUM)
**Location**: `resource_broker.py:744-758`

**Problem**:
```python
def reserve(self, task: TaskResourceContract, *, task_id: str = "") -> bool:
    tid = str(task_id or id(task))
    with self._lock:
        if self._running.get(tid) is not None:  # Check
            return True
    lease = self.try_reserve(task, task_id=tid)  # Use (lock released)
```

- Lines 750-752: Check if `tid` in `_running` **inside lock**
- Lock released before line 753 `try_reserve` call
- Another thread can `try_reserve(tid)` between check and use
- Result: duplicate admission via different code paths

**Impact**: Same as Issue 1 when hit concurrently

---

### Issue 4: release() Lease Lookup Fallthrough Bypasses Idempotence (MEDIUM)
**Location**: `resource_broker.py:760-769`

**Problem**:
```python
def release(self, task: TaskResourceContract, *, task_id: str = "") -> None:
    tid = str(task_id or id(task))
    with self._lock:
        lease = self._leases.pop(tid, None)
    if lease is not None:
        lease.release()
        return
    # No lease: directly call _release_locked
    self._release_locked(task, task_id=tid)  # NOT idempotent-checked
```

- Line 769: Falls through to `_release_locked` if no lease found
- `_release_locked` has no idempotence guard (see Issue 2)
- Caller that uses `try_reserve` directly (returns lease, not in `_leases`) then calls `release()`:
  - First call: lease not in `_leases`, calls `_release_locked`, succeeds
  - Second call: lease still not in `_leases`, calls `_release_locked` again → **double release**

**Impact**: Negative token accounting on repeated `release()` calls

---

## Test Coverage Gaps

### Missing Test: Duplicate task_id rejection
No test in `test_resource_broker.py` that attempts to reserve same `task_id` twice

### Missing Test: Idempotent release verification
No test that calls `release()` twice on same task_id and verifies tokens only released once

### Missing Test: Token accounting exactness
No test that sums `_cpu.in_use` / `_io.in_use` across reserve/release cycles

### Missing Test: Concurrent reserve/release on same task_id
No threading test for TOCTOU race in `reserve()`

---

## Secondary Issues

### Issue 5: ReservationLease.release() Lock Ordering (LOW)
**Location**: `resource_broker.py:429-435`

- Lease holds its own `_lock` (line 431-434)
- Then calls `broker._release_locked` which acquires broker's `_lock` (line 738 in caller)
- Potential deadlock if broker method ever calls lease method while holding broker lock
- Currently safe but fragile

### Issue 6: can_admit() Non-Atomic with try_reserve() (MEDIUM)
**Location**: `resource_broker.py:681-711` and `713-736`

- `can_admit()` is documented as "pure function" (R31-P0-009)
- BUT: It reads `_running` to compute `in_use` (line 695)
- Between `can_admit()` True and `try_reserve()` call, another thread can change `_running`
- Result: `can_admit()` can return True but subsequent `try_reserve()` fail (acceptable)
- OR: `can_admit()` False but `try_reserve()` would succeed (missed opportunity)
- Design is correct (caller should use `try_reserve` atomically), but doc could clarify

---

## Recommendations

### Fix 1: Add duplicate task_id check in try_reserve
```python
def try_reserve(self, task, task_id=''):
    tid = str(task_id or id(task))
    with self._lock:
        if tid in self._running:
            # Already reserved: reject to prevent token leak
            return None
        if not self.can_admit(task):
            return None
        # ... rest of acquisition
```

### Fix 2: Make _release_locked verify task was reserved
```python
def _release_locked(self, task, *, task_id: str) -> None:
    was_running = self._running.pop(task_id, None)
    if was_running is None:
        return  # Not reserved, don't release tokens
    self._cpu.release(task.cpu_tokens)
    self._io.release(task.io_tokens)
```

### Fix 3: Atomic reserve() check-and-use
```python
def reserve(self, task, *, task_id: str = "") -> bool:
    tid = str(task_id or id(task))
    with self._lock:
        if tid in self._running:
            return True
        # Keep lock held through try_reserve core logic
        if tid in self._running:  # Double-check
            return True
        lease = self.try_reserve(task, task_id=tid)
        # ... rest
```
OR: Simplify by always using try_reserve path (preferred)

### Fix 4: Add idempotence flag to lease
```python
class ReservationLease:
    def release(self):
        with self._lock:
            if self._released:
                return  # Already released
            self._released = True
            self._broker._release_locked_unchecked(self._task, task_id=self._task_id)
```

### Fix 5: Add comprehensive regression tests
- `test_duplicate_task_id_rejected()`
- `test_idempotent_release()`
- `test_token_accounting_exact()`
- `test_concurrent_reserve_release_same_id()`
