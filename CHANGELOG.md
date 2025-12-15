# Changelog

All notable changes to Strata FDTD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-14

### Added
- Initial public release of Strata FDTD
- Core 3D acoustic FDTD solver with pressure-velocity formulation
- Native C++ kernels with OpenMP parallelization (10-75x speedup)
- GPU-accelerated solver via PyTorch
- Complete materials system with ADE (Auxiliary Differential Equation) support
- Frequency-dependent porous materials (Debye poles)
- Solid materials and resonators (Lorentz poles)
- Geometry modeling with SDF (Signed Distance Functions)
- CSG operations (union, intersection, difference, smooth variants)
- Parametric geometry for optimization workflows
- Loudspeaker enclosure templates (tower, bookshelf designs)
- Helmholtz resonator arrays
- Manufacturing tools (lamination, constraint checking)
- CAD export (DXF for laser cutting, STL for 3D printing, JSON)
- HDF5-based I/O for simulation results
- Desktop visualization app (Tauri) with 3D rendering
- Command-line interface (fdtd-compute)
- Comprehensive test suite (15.8K lines, 27 test files)
- Complete documentation (7 guides)
- Example scripts and benchmarks

### Migration Notes
- Migrated from private ml-audio-codecs repository
- Reorganized package structure: metamaterial → strata_fdtd
- Modular organization with clear subpackages:
  - `strata_fdtd.core` - Core FDTD solver
  - `strata_fdtd.boundaries` - Boundary conditions (PML, rigid walls)
  - `strata_fdtd.materials` - Acoustic materials
  - `strata_fdtd.geometry` - Geometric modeling (SDF, CSG, primitives)
  - `strata_fdtd.manufacturing` - Lamination and export tools
  - `strata_fdtd.io` - HDF5 I/O
  - `strata_fdtd.analysis` - Post-processing (A/C weighting, SPL)
  - `strata_fdtd.cli` - Command-line interface

### Changed
- Renamed C++ extension: `_fdtd_kernels` → `_kernels`
- Updated all import paths throughout codebase
- Modernized build system: scikit-build-core for C++ extensions
- Updated pyproject.toml with optional dependency groups

### Technical Details
- Python 3.10+ required
- Dependencies: numpy, scipy, h5py, click, rich, psutil
- Optional dependencies: torch (GPU), ezdxf/numpy-stl (CAD export)
- C++ extension: pybind11 with CMake build system
- OpenMP support for parallelization
- SIMD vectorization (NEON on ARM64, SSE4.2/AVX2 on x86_64)

## [Unreleased]

### Planned
- PyPI package release
- Conda package
- Documentation website (GitHub Pages or Read the Docs)
- Additional examples (horn loudspeakers, metamaterial designs)
- Web-based viewer (complementing desktop app)
- Improved GPU backend with multi-GPU support
- MPI support for distributed simulations

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
