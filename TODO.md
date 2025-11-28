# ndsolver Modernization Plan

**Original**: PhD research code (Erdmann 2006, Goodman 2011) for n-dimensional Stokes flow in periodic porous media.

**Goal**: Get running on modern Python with scipy solvers, then expand to full feature set.

---

## Completed: Package Reorganization

```
ndsolver/                 # repo root
├── ndsolver/             # main package
│   ├── __init__.py
│   ├── core.py           # Solver class
│   ├── cli.py            # command-line interface
│   ├── hdf5.py           # HDF5 I/O
│   ├── nd_domain.py      # geometry generation
│   └── symbolic/         # equation assembly
│       ├── __init__.py
│       ├── Equation.py
│       ├── eq_solver.py
│       ├── ndim_eq.py
│       └── ndimed.py
├── tests/                # test suite
│   ├── test_validate.py
│   ├── test_solver.py    # pytest unit tests
│   └── test_domains.py   # procedural domain generation
├── scripts/              # utility scripts
├── pyproject.toml        # modern packaging
├── requirements.txt
└── README.md
```

---

## Target Environment

**Python Version**: **3.11** (recommended)
- Stable, well-supported, good performance
- Avoid 3.12+ for now (some scientific packages still catching up)
- 3.10 is fine as fallback

**Key Dependencies** (modern versions):
- numpy >= 1.24
- scipy >= 1.11
- tables (PyTables) >= 3.8
- Pillow >= 10.0 (replaces scipy.misc.imread)
- pytest >= 7.0 (for testing)

---

## pyenv Setup Instructions

```bash
# Install pyenv if not present
curl https://pyenv.run | bash

# Add to ~/.bashrc or ~/.zshrc:
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart shell, then:
pyenv install 3.11.9
cd /home/meawoppl/repos/ndsolver
pyenv local 3.11.9

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (after we create requirements.txt)
pip install -r requirements.txt
```

---

## Phase 1: Core Python 3 Compatibility (scipy-only path)

### 1.1 Create Project Infrastructure
- [x] Create `requirements.txt` with pinned versions
- [x] Create `pyproject.toml` for modern packaging
- [x] Add `.python-version` file for pyenv
- [x] Reorganize into proper package structure
- [x] Update internal imports to relative/absolute

### 1.2 Fix Print Statements (core solver path done)
Files fixed:
- [x] `ndsolver/core.py` - Replaced with logging
- [x] `ndsolver/symbolic/eq_solver.py`
- [x] `ndsolver/symbolic/ndim_eq.py`
- [x] `ndsolver/symbolic/ndimed.py`
- [x] `ndsolver/symbolic/Equation.py`

Files remaining (secondary):
- [x] `ndsolver/nd_domain.py` - Replaced with logging
- [x] `ndsolver/hdf5.py`
- [x] `ndsolver/cli.py`
- [x] `tests/test_validate.py`
- [x] `scripts/benchmark.py`
- [x] `scripts/` other files - Converted to f-strings

### 1.3 Fix Dictionary Iteration
- [x] `ndsolver/symbolic/Equation.py` - `.iteritems()` → `.items()`
- [x] `ndsolver/symbolic/eq_solver.py`
- [x] `ndsolver/core.py`

### 1.3a Fix Dict Modification During Iteration
- [x] `ndsolver/symbolic/Equation.py` - `list(self.keys())` before del

### 1.3b Add Logging
- [x] `ndsolver/__init__.py` - Package logger setup
- [x] `ndsolver/core.py` - Replaced dbprint with logging module
- [x] Removed `printing` parameter from Solver constructor
- [x] Removed `dbcallback` parameter from Solver constructor

### 1.4 Replace Deprecated scipy.misc.imread
- [x] `ndsolver/cli.py` - Replaced with Pillow

### 1.5 Update PyTables API
Files with `openFile()` → `open_file()`:
- [x] `ndsolver/hdf5.py`
- [x] `ndsolver/core.py`
- [x] `ndsolver/cli.py`
- [x] `ndsolver/nd_domain.py`
- [x] `scripts/qqueue.py`
- [x] `scripts/summarize.py`
- [x] `scripts/sv-approximations.py`
- [x] `tests/test_validate.py`

### 1.6 Fix Integer Division
- [x] Audit all `/` operators for integer division - All already use floats (e.g., `/2.`)
- [x] No changes needed - code already handles division correctly

### 1.7 Fix Indentation (tabs vs spaces)
- [x] Fixed 5 tab characters in core.py (lines 550, 903, 911, 921, 924)
- [x] Converted all tabs to 4 spaces

### 1.8 Modernize String Formatting
- [x] Replace `%` formatting with f-strings
- [x] Example: `"Value: %s" % x` → `f"Value: {x}"`
- [x] Example: `"(%i, %i)" % (x, y)` → `f"({x}, {y})"`

---

## Phase 2: Get Basic Solver Running

### 2.1 Minimal Import Test
```bash
python -c "from ndsolver import Solver"
```
- [x] Fix imports until this works

### 2.2 Create Simple Test Case
- [x] Create `tests/test_solver.py` with pytest tests
- [x] Small grid (5x5, 8x8) test cases
- [x] Single obstacle tests
- [x] Run with `spsolve` and `splu` methods

### 2.3 Validate Against Known Solution
- [x] Use `test_validate.py` as reference (ported to Python 3)
- [x] Port key validation tests to pytest (`test_solver.py`)
- [x] Verified correct_P, correct_u, correct_v solutions match

---

## Phase 3: Full scipy Solver Suite

### 3.1 Verify All scipy Solvers Work
- [x] `spsolve` - Direct solve (default) - verified in tests
- [x] `splu` - LU factorization - verified in tests
- [ ] `bicgstab` - Iterative BiCGSTAB
- [x] Removed all `trilinos` code paths from core.py (~500 lines)
- [x] Removed `ruge`/pyamg code paths

### 3.2 Update Sparse Matrix Construction
- [x] Verify `lil_matrix` → `csr_matrix` conversions work
- [x] Check for any deprecated sparse APIs

### 3.3 Test Convergence Loop
- [x] Verify Biot number acceleration works
- [x] Test pressure correction iteration
- [x] Check divergence calculation

---

## Phase 4: HDF5 I/O Restoration

### 4.1 Fix PyTables Schema
- [x] Update table descriptors for PyTables 3.x
- [x] Test CArray storage for large domains
- [x] Verify simulation save/load cycle (test_validate.py tests pass)

### 4.2 Test Big Mode (Memory Mapping)
- [ ] Create large test domain (64³ or 128³)
- [ ] Verify `.mem` file creation
- [ ] Test memory-mapped array access

---

## Phase 5: Extended Features

### 5.1 N-Dimensional Support
- [x] Test 2D domains (test_solver.py, test_validate.py)
- [x] Test 3D domains (test_validate.py::test_3d)
- [ ] Document 4D limitations/status

### 5.2 Domain Generation
- [x] Created `tests/test_domains.py` with procedural generation
- [x] Test random geometry generation (make_random_circles_domain, make_porous_media_domain)
- [x] Fix any numpy random API changes (uses np.random.default_rng)

### 5.3 CLI Restoration
- [x] Migrated from optparse to argparse
- [x] Fixed all print statements
- [x] Replaced scipy.misc.imread with Pillow
- [ ] Test batch processing mode
- [ ] Test all command-line flags

### 5.4 Results Aggregation
- [ ] Fix `summarize.py`
- [ ] Test permeability calculation
- [ ] Verify statistical output

---

## Phase 6: Code Quality (Optional but Recommended)

### 6.1 Testing Infrastructure
- [x] Set up pytest (pyproject.toml configured)
- [x] Created `tests/test_solver.py` with 15 tests
- [x] Add CI with GitHub Actions (.github/workflows/ci.yml)

### 6.2 Code Cleanup
- [x] Remove wildcard imports (`from scipy import *`) - Fixed in eq_solver.py
- [ ] Add type hints to core functions
- [ ] Add docstrings to public API

### 6.3 Documentation
- [ ] Write proper README.md
- [ ] Add usage examples
- [ ] Document solver parameters

---

## Files by Priority

**Critical Path** (must fix for scipy-only):
1. `ndsolver/core.py` - Main solver
2. `ndsolver/symbolic/Equation.py` - Equation representation
3. `ndsolver/symbolic/ndim_eq.py` - DOF generation
4. `ndsolver/symbolic/ndimed.py` - Grid utilities
5. `ndsolver/symbolic/eq_solver.py` - Symbolic solving

**Secondary** (needed for full functionality):
6. `ndsolver/hdf5.py` - File I/O
7. `ndsolver/nd_domain.py` - Geometry
8. `ndsolver/cli.py` - Command line

**Tertiary** (nice to have):
9. `tests/test_validate.py` - Testing
10. Remaining utility files in `scripts/`

---

## Known Solver Methods

| Method | Backend | Status | Notes |
|--------|---------|--------|-------|
| `spsolve` | scipy.sparse.linalg | **Working** | Direct solver, default |
| `splu` | scipy.sparse.linalg | **Working** | LU factorization, faster |
| `nobi` | scipy | Target | No Biot acceleration |
| `bicgstab` | scipy | Target | Iterative solver |
| `trilinos` | PyTrilinos | **Removed** | MPI parallel code deleted |
| `ruge` | pyamg | **Removed** | pyamg code deleted |

---

## Quick Start After Setup

```bash
# After completing Phase 1-2:
cd /home/meawoppl/repos/ndsolver
source .venv/bin/activate

# Test import
python -c "from ndsolver import Solver; print('Success!')"

# Run simple test
python -c "
import numpy as np
from ndsolver import Solver

# Create small 2D domain with obstacle
domain = np.zeros((16, 16), dtype=np.int8)
domain[6:10, 6:10] = 1  # solid block

# Solve with pressure drop in x-direction
s = Solver(domain, (1, 0), sol_method='spsolve')
s.converge()
print('Done!')
"

# Run tests
python -m pytest tests/test_solver.py -v
```

---

## Estimated Complexity

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1 | Medium | Low - mechanical changes |
| Phase 2 | Low | Medium - may uncover hidden issues |
| Phase 3 | Low | Low - well-defined scipy API |
| Phase 4 | Medium | Medium - PyTables changes |
| Phase 5 | Medium | Medium - edge cases |
| Phase 6 | High | Low - optional polish |

---

## Notes

- The core algorithm (pressure-velocity coupling with Biot acceleration) is solid
- Staggered grid FD with periodic BC is well-implemented
- Main work is mechanical Python 2→3 porting
- scipy sparse solver API is stable - should "just work" once syntax fixed
- Defer Trilinos/MPI until scipy path is solid
