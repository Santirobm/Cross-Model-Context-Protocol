# CMCP Release Validation (Windows)

**Date:** 2026-11-20  
**Branch:** `exp/gate2-llama-ablation`  
**OS / Platform:** Windows (`win32`)  
**Virtualenv Interpreter:** `C:\Users\sgi.lm\Desktop\Cross-Model-Context-Protocol\.venv\Scripts\python.exe` (`Python 3.13.5`, `pytest 9.2.0`)

---

## 1. Executed Commands & Observed Results

### 1.1 CTest (`build`)
- **Command:**
  ```powershell
  $env:PATH = "C:\msys64\clang64\bin;" + $env:PATH
  ctest --test-dir build --output-on-failure
  ```
- **Result:** `1/1` passed (`test_cmcp_core`, `0.01 sec`, total `0.04 sec`, exit code `0`).

### 1.2 CTest (`build-san`)
- **Command:**
  ```powershell
  $env:PATH = "C:\msys64\clang64\bin;" + $env:PATH
  ctest --test-dir build-san --output-on-failure
  ```
- **Result:** `1/1` passed (`test_cmcp_core`, `0.13 sec`, total `0.16 sec`, exit code `0`).

### 1.3 Pytest (`.venv\Scripts\pytest.exe`)
- **Command:**
  ```powershell
  .venv\Scripts\pytest.exe tests/ -v
  ```
- **Interpreter:** `C:\Users\sgi.lm\Desktop\Cross-Model-Context-Protocol\.venv\Scripts\python.exe` (`Python 3.13.5`)
- **Result:** `22 passed in 2.00s` (0 failed, 0 skipped, exit code `0`):
  - `tests/test_gate1_abd.py`: 3 passed
  - `tests/test_protocol_and_adapters.py`: 7 passed
  - `tests/test_serialize_api.py`: 6 passed
  - `tests/test_state_store_consistency.py`: 6 passed
