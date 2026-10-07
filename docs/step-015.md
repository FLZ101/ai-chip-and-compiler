# Step 015: Runtime and Driver

This document describes the host-side runtime API.

Step 015 adds:

- `runtime/buffer.py`
- `runtime/driver.py`
- `runtime/profiler.py`

---

## 1. Buffer

File: [../runtime/buffer.py](../runtime/buffer.py)

`Buffer` represents a contiguous FP32 allocation in simulated global memory.

Features:

- shape and byte-count metadata,
- `write(array)` host-to-device copy,
- `read()` device-to-host copy.

---

## 2. Driver

File: [../runtime/driver.py](../runtime/driver.py)

`Driver` owns a `Device` and provides:

- aligned buffer allocation,
- host-to-device and device-to-host copies,
- ISA kernel launches,
- synchronization,
- runtime profiling records.

Example:

```python
from runtime import Driver

driver = Driver(global_size=1 << 20, sm_count=4)
a = driver.allocate((8,), name="a")
b = driver.allocate((8,), name="b")
out = driver.allocate((8,), name="out")
driver.copy_to_device(a, ...)
driver.copy_to_device(b, ...)
driver.launch(isa_text, grid_dim=(1,), block_dim=(8,), profile=True)
result = driver.copy_from_device(out)
```

The current simulator executes synchronously, so `sync()` is a no-op that
preserves the future asynchronous host API.

---

## 3. Profiler

File: [../runtime/profiler.py](../runtime/profiler.py)

`RuntimeProfiler` collects `LaunchRecord` objects:

- cycles,
- optional `ProfileCounters` snapshot.

`summary()` reports launch count, total cycles, and per-launch records.

---

## 4. Files changed

- [../runtime/__init__.py](../runtime/__init__.py)
- [../runtime/buffer.py](../runtime/buffer.py)
- [../runtime/driver.py](../runtime/driver.py)
- [../runtime/profiler.py](../runtime/profiler.py)
- [../tests/unit/test_runtime.py](../tests/unit/test_runtime.py)

---

## 5. Tests

New tests cover:

- multiple deterministic launches,
- host-to-device and device-to-host copies,
- no state corruption between launches,
- profiler summary,
- unknown-buffer rejection.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
