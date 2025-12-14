# Agent Starting Prompt for Strata FDTD

## Quick Context

You are working in **Strata FDTD**, a Python package for 3D acoustic simulation using FDTD (Finite-Difference Time-Domain) methods. This repository was just migrated from a private `ml-audio-codecs` repository on December 14, 2024.

**Current Branch**: `migration/initial-fdtd-code`
**Migration Commit**: `d201a4f`

## What This Repository Does

Strata FDTD simulates acoustic wave propagation for:
- Loudspeaker cabinet design
- Acoustic metamaterial development
- General acoustic modeling

**Key Features:**
- High-performance FDTD solver with C++ native kernels (10-75x speedup)
- GPU acceleration via PyTorch
- Complete geometry modeling (SDF, CSG operations)
- Manufacturing tools (lamination, DXF/STL export)
- Desktop visualization app (Tauri + React + Three.js)

## Package Structure

```
src/strata_fdtd/
├── core/              # FDTDSolver, grids, sources, probes
├── boundaries/        # PML, rigid walls, radiation impedance
├── materials/         # Frequency-dependent acoustic materials
├── geometry/          # SDF primitives, CSG, loudspeakers, resonators
├── manufacturing/     # Lamination, constraints, CAD export
├── io/                # HDF5 I/O
├── analysis/          # A/C weighting, SPL calculations
├── cli/               # fdtd-compute command-line tool
└── _kernels/          # C++ OpenMP kernels (18.7K Python + 3.3K C++)
```

**Also includes:**
- `tests/` - 27 test files (15.8K lines)
- `docs/` - 7 comprehensive guides
- `examples/` - Example scripts
- `benchmarks/` - Performance benchmarks
- `viz/` - Tauri desktop app (23.8K TypeScript)

## What Just Happened (Migration)

**Renamed package**: `metamaterial` → `strata_fdtd`
**Updated C++ extension**: `_fdtd_kernels` → `_kernels`
**Reorganized**: Flat package → Modular subpackages
**Updated imports**: All 2,015 files processed automatically

**NOT included** (stayed in private repo):
- Codec specifications (Shamrock, Infinity, Foundations)
- ML codec implementations
- Document tooling scripts (critical-review.py, etc.)

## Common Import Patterns

```python
from strata_fdtd import FDTDSolver, GaussianPulse, PML
from strata_fdtd.materials import AIR_20C, FIBERGLASS_48
from strata_fdtd.geometry import Box, Sphere, Union
from strata_fdtd.manufacturing import export_dxf, export_stl
from strata_fdtd.io import HDF5ResultWriter, HDF5ResultReader
```

## Immediate Priorities

The migration is complete, but these tasks remain:

1. **Merge migration branch** to main
2. **Test installation** on different platforms
3. **Validate native kernel builds**
4. **Set up CI/CD** (GitHub Actions workflows)
5. **Update documentation** for new package structure
6. **Prepare for PyPI release**

## How to Help

**Before starting any work:**
1. Read `README.md` for project overview
2. Review `AGENT_ONBOARDING.md` for detailed context
3. Check `CHANGELOG.md` for what changed
4. Look at relevant test files for usage examples

**When working:**
- Follow existing code patterns for consistency
- Update tests when changing functionality
- Run `pytest tests/ -v` before committing
- Check that syntax is valid: `python -c "import strata_fdtd"`
- Reference the migration commit (d201a4f) if you need to understand what changed

## Key Files to Know

- `pyproject.toml` - Python packaging, dependencies
- `src/strata_fdtd/__init__.py` - Top-level package exports
- `src/strata_fdtd/core/solver.py` - Main FDTD implementation (3,447 lines)
- `src/strata_fdtd/_kernels/CMakeLists.txt` - C++ build config
- `docs/getting-started.md` - Installation guide

## Development Commands

```bash
# Install package
pip install -e ".[dev,native]"

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/profile_fdtd.py --size 100

# Build desktop app
cd viz && pnpm install && pnpm tauri dev

# Validate Python syntax
python -c "import strata_fdtd; print(strata_fdtd.__version__)"
```

## Quick Facts

- **Total code**: ~66K lines (Python + C++ + TypeScript)
- **Python version**: 3.10+
- **Build system**: scikit-build-core + CMake
- **Performance**: 10-75x speedup with native kernels
- **License**: MIT

## Getting Oriented

If you're asked to work on something specific:
1. Check if it involves **core solver** → `src/strata_fdtd/core/`
2. Check if it involves **geometry** → `src/strata_fdtd/geometry/`
3. Check if it involves **C++ kernels** → `src/strata_fdtd/_kernels/`
4. Check if it involves **visualization** → `viz/`
5. Look for existing tests in `tests/` for examples
6. Review examples in `examples/` for usage patterns

---

**Need more details?** Read `AGENT_ONBOARDING.md` in this repository.

**Repository**: https://github.com/rjwalters/strata-fdtd
**Status**: Migration complete, ready for validation and release preparation
