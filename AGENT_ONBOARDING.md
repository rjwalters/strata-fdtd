# Agent Onboarding: Strata FDTD

## Repository Overview

**Strata FDTD** is a high-performance Python package for 3D acoustic simulation using the Finite-Difference Time-Domain (FDTD) method. This repository was recently migrated from the private `ml-audio-codecs` repository to create a focused, public toolkit for FDTD acoustic simulation.

**Migration Date**: December 14, 2024
**Current Branch**: `migration/initial-fdtd-code` (ready for review/merge to main)
**Migration Commit**: `d201a4f`

## Repository Purpose

This package is designed for:
- **Loudspeaker cabinet design** - Simulate acoustic behavior of speaker enclosures
- **Metamaterial development** - Design and test acoustic metamaterials
- **General acoustic modeling** - Wave propagation, resonators, etc.

## Package Structure

```
strata-fdtd/
├── src/strata_fdtd/           # Main Python package
│   ├── core/                  # Core FDTD solver (FDTDSolver, grids, sources, probes)
│   ├── boundaries/            # Boundary conditions (PML, rigid walls, radiation impedance)
│   ├── materials/             # ADE-based acoustic materials (porous, solids, resonators)
│   ├── geometry/              # SDF primitives, CSG operations, loudspeakers, resonators
│   ├── manufacturing/         # Lamination (Slice/Stack), constraints, CAD export (DXF/STL)
│   ├── io/                    # HDF5 I/O for simulation results
│   ├── analysis/              # Post-processing (A/C weighting, SPL calculations)
│   ├── cli/                   # Command-line interface (fdtd-compute)
│   └── _kernels/              # C++ native kernels with OpenMP (10-75x speedup)
│       ├── CMakeLists.txt     # CMake build configuration
│       ├── kernels.cpp        # pybind11 bindings
│       ├── include/           # C++ headers
│       └── src/               # C++ implementation (fdtd_step, boundaries, pml, ade, microphones)
├── tests/                     # 27 test files (15.8K lines)
├── docs/                      # 7 comprehensive guides
├── examples/                  # Example scripts (basic_pulse, helmholtz_resonator, etc.)
├── benchmarks/                # Performance benchmarks (profile_fdtd, benchmark_ade, etc.)
├── viz/                       # Tauri desktop visualization app (23.8K lines TypeScript)
├── pyproject.toml             # Modern Python packaging with scikit-build-core
├── README.md                  # Comprehensive public README
├── CHANGELOG.md               # Version history and migration notes
├── LICENSE                    # MIT License
└── migrate.py                 # Migration automation script (for reference)
```

## Key Technologies

**Python Stack:**
- **Core**: NumPy, SciPy, H5py
- **Optional**: PyTorch (GPU acceleration)
- **Geometry**: ezdxf (DXF export), numpy-stl (STL export)
- **CLI**: Click, Rich (terminal UI)

**C++ Stack:**
- **Build**: scikit-build-core + CMake
- **Bindings**: pybind11
- **Parallelization**: OpenMP
- **Optimization**: SIMD (NEON on ARM64, SSE/AVX on x86_64)

**Visualization:**
- **Desktop**: Tauri (Rust + TypeScript/React)
- **3D**: Three.js
- **Plots**: Plotly.js
- **HDF5**: h5wasm (browser-based HDF5 reading)

## Migration Context

### What Changed
- **Package name**: `metamaterial` → `strata_fdtd`
- **C++ extension**: `_fdtd_kernels` → `_kernels`
- **Structure**: Flat package → Modular subpackages
- **Build system**: setuptools → scikit-build-core
- **All imports updated**: ~2,015 files processed

### What Stayed The Same
- **Core FDTD algorithms** - No changes to solver logic
- **Native kernels** - Same C++ implementation, just renamed
- **Test suite** - All 27 test files migrated
- **Documentation** - All 7 guides migrated
- **Visualization** - Complete Tauri app included

### What's NOT Included (Stayed in ml-audio-codecs)
- **Codec specifications** (Shamrock, Infinity, Foundations)
- **ML codec implementations** (`src/codec/`)
- **Document tooling** (critical-review.py, generate-revision.py)
- **Loom orchestration** (kept simple for public repo)

## Common Tasks

### 1. Development Setup

```bash
# Clone and install with native kernels
git clone https://github.com/rjwalters/strata-fdtd.git
cd strata-fdtd

# macOS
brew install cmake libomp
pip install -e ".[dev,native]"

# Linux
sudo apt install cmake g++ libomp-dev
pip install -e ".[dev,native]"

# Run tests
pytest tests/ -v
```

### 2. Running Simulations

```bash
# Via Python
python examples/basic_pulse.py

# Via CLI
fdtd-compute --script examples/basic_pulse.py --output results.h5

# With specific backend
fdtd-compute --script sim.py --backend native --output results.h5
```

### 3. Building C++ Extension

```bash
# Rebuild native kernels
pip install -e ".[native]" --force-reinstall --no-deps

# Check if native kernels are available
python -c "from strata_fdtd import has_native_kernels; print(has_native_kernels())"
```

### 4. Running Benchmarks

```bash
# Profile FDTD solver
python benchmarks/profile_fdtd.py --size 100 --steps 100

# Benchmark ADE materials
python benchmarks/benchmark_ade.py

# Quick benchmark
python benchmarks/benchmark_ade.py --quick
```

### 5. Desktop Visualization App

```bash
cd viz
pnpm install
pnpm tauri dev  # Development mode
pnpm tauri build  # Production build
```

## Important Files to Know

### Core Implementation
- **`src/strata_fdtd/core/solver.py`** (3,447 lines) - Main FDTDSolver class
- **`src/strata_fdtd/core/solver_gpu.py`** (2,273 lines) - GPU-accelerated solver
- **`src/strata_fdtd/boundaries/_boundaries.py`** (819 lines) - PML, rigid walls
- **`src/strata_fdtd/materials/library.py`** - Pre-defined material constants

### Build Configuration
- **`pyproject.toml`** - Python packaging, dependencies, optional extras
- **`src/strata_fdtd/_kernels/CMakeLists.txt`** - C++ build configuration
- **`src/strata_fdtd/_kernels/kernels.cpp`** - pybind11 bindings

### Documentation
- **`README.md`** - Public-facing documentation
- **`CHANGELOG.md`** - Version history
- **`docs/getting-started.md`** - Installation and first simulation
- **`docs/api-reference.md`** - Complete API documentation

## Known Issues / TODO

### Immediate Priorities
1. **Merge migration branch** to main
2. **Test installation** on different platforms (macOS, Linux, Windows)
3. **Validate native kernel builds** on different architectures
4. **Run full test suite** with dependencies installed

### Documentation Needs
5. **Update docs/** for new package structure (some imports may be outdated)
6. **Add API reference** for subpackages (currently monolithic)
7. **Create CONTRIBUTING.md** for external contributors

### CI/CD Setup
8. **Add GitHub Actions workflows** (.github/workflows/)
   - Multi-platform testing (Ubuntu, macOS, Windows)
   - Native extension building
   - Linting and type checking (ruff, black, mypy)
   - Benchmark tracking
9. **Set up PyPI release workflow**

### Future Enhancements
10. **Documentation website** (GitHub Pages or Read the Docs)
11. **PyPI package release** (currently not published)
12. **Conda package** for easier installation
13. **Web-based viewer** (complement to desktop app)

## Testing Strategy

### Test Categories
- **Core solver tests**: `tests/test_fdtd.py`, `tests/test_fdtd_gpu.py`
- **Material tests**: `tests/test_materials.py`
- **Geometry tests**: `tests/test_sdf*.py`, `tests/test_geometry.py`
- **Integration tests**: `tests/test_fdtd_benchmarks.py`
- **Native kernel tests**: `tests/test_native_extension.py`, `tests/test_microphone_native.py`

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/test_fdtd.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip GPU tests (if no GPU available)
pytest tests/ -v -m "not gpu"

# Run benchmarks
pytest tests/test_fdtd_benchmarks.py -v --benchmark
```

## Performance Characteristics

### Native Kernel Speedup
| Grid Size | Python (M cells/s) | Native C++ (M cells/s) | Speedup |
|-----------|-------------------|----------------------|---------|
| 50³       | 28                | 72                   | 2.6x    |
| 100³      | 32                | 531                  | 16.4x   |
| 200³      | 33                | 2,469                | 75x     |

*Benchmarked on Apple M1 Max with 28 threads*

### Memory Usage
- Base FDTD: ~200 MB for 100³ grid
- With PML boundaries: +20%
- With materials (ADE): +30%
- GPU backend: Requires grid to fit in VRAM

## Development Guidelines

### Code Style
- **Line length**: 100 characters
- **Formatter**: Black (configured in pyproject.toml)
- **Linter**: Ruff (configured in pyproject.toml)
- **Type hints**: mypy (optional but encouraged)

### Import Conventions
```python
# Preferred imports
from strata_fdtd import FDTDSolver, GaussianPulse, PML
from strata_fdtd.materials import AIR_20C, FIBERGLASS_48
from strata_fdtd.geometry import Box, Sphere, Union
from strata_fdtd.manufacturing import export_dxf, export_stl
from strata_fdtd.io import HDF5ResultWriter, HDF5ResultReader

# Also valid (submodule imports)
from strata_fdtd.core import FDTDSolver
from strata_fdtd.boundaries import PML, RigidBoundary
```

### Commit Messages
- Use conventional commits format
- Include "🤖 Generated with Claude Code" footer for AI-assisted changes
- Reference issue numbers when applicable

## Getting Help

### Documentation
- **README.md** - Overview and quick start
- **docs/getting-started.md** - Detailed installation
- **docs/api-reference.md** - Complete API documentation
- **docs/troubleshooting.md** - Common issues

### Code References
- Check existing tests for usage examples
- Review `examples/` directory for complete examples
- Look at benchmarks for performance-critical code

### Migration History
- Migration plan: `/Users/rwalters/.claude/plans/wise-beaming-pascal.md`
- Original repository: https://github.com/rjwalters/ml-audio-codecs (private)
- Migration script: `migrate.py` (for reference)

## Quick Reference

### Import Cheat Sheet
| Module | Key Exports |
|--------|-------------|
| `strata_fdtd` | FDTDSolver, GPUFDTDSolver, GaussianPulse, PML, UniformGrid, NonuniformGrid |
| `strata_fdtd.materials` | AIR_20C, FIBERGLASS_48, Material library |
| `strata_fdtd.geometry` | Box, Sphere, Union, Intersection, Difference |
| `strata_fdtd.manufacturing` | Slice, Stack, export_dxf, export_stl |
| `strata_fdtd.io` | HDF5ResultWriter, HDF5ResultReader |
| `strata_fdtd.analysis` | apply_weighting, calculate_spl |

### File Count Summary
- **Python files**: 30 modules (18.7K lines)
- **C++ files**: 11 files (3.3K lines)
- **TypeScript files**: ~80 files (23.8K lines)
- **Test files**: 27 files (15.8K lines)
- **Total**: ~66K lines of code

## Agent Workflow Suggestions

When working in this repository:

1. **Start by reading** README.md and relevant docs
2. **Check test files** to understand expected behavior
3. **Review migration commit** (d201a4f) to understand what changed
4. **Run tests** before making changes
5. **Use existing patterns** - consistency is important for public code
6. **Update docs** when adding features
7. **Add tests** for new functionality
8. **Run benchmarks** if changing performance-critical code

## Success Criteria

The repository is ready for public release when:
- ✅ Code migrated and reorganized
- ✅ All imports updated
- ✅ Build system modernized
- ✅ Documentation comprehensive
- ⏳ Tests pass on all platforms
- ⏳ Native kernels build successfully
- ⏳ CI/CD workflows set up
- ⏳ Package published to PyPI
- ⏳ Documentation website live

---

**Status**: Migration complete, ready for validation and public release preparation.

**Last Updated**: December 14, 2024
