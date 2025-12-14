# Strata FDTD

**FDTD acoustic simulation toolchain for loudspeaker design and metamaterials**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Strata FDTD is a comprehensive Python package for 3D acoustic simulation using the Finite-Difference Time-Domain (FDTD) method. It features native C++ kernels with OpenMP parallelization (10-75x speedup), GPU acceleration, and a complete toolkit for designing loudspeaker enclosures and acoustic metamaterials.

## Features

### Core FDTD Solver
- **High-performance 3D acoustic FDTD** with pressure-velocity formulation
- **Native C++ kernels** with OpenMP parallelization (10-75x faster than pure Python)
- **GPU acceleration** via PyTorch for large-scale simulations
- **Uniform and nonuniform grids** for adaptive resolution
- **Advanced boundary conditions**: PML (Perfectly Matched Layer), rigid walls, radiation impedance
- **Frequency-dependent materials** via ADE (Auxiliary Differential Equation) method

### Geometry Modeling
- **SDF (Signed Distance Function) primitives**: boxes, spheres, cylinders, cones, horns
- **CSG operations**: union, intersection, difference, smooth blending
- **Parametric geometry** for optimization workflows
- **Loudspeaker enclosure** templates (tower, bookshelf designs)
- **Helmholtz resonator** arrays and metamaterial unit cells

### Manufacturing Tools
- **Slice/Stack lamination** for composite structures
- **Manufacturing constraints** checking (minimum wall thickness, gaps, connectivity)
- **CAD export**: DXF (laser cutting), STL (3D printing), JSON (data interchange)

### Visualization
- **Desktop app** (Tauri) with 3D pressure field rendering, spectrum plots, waterfall plots
- **Real-time playback** controls for time-domain visualization
- **Web viewer** for sharing results

## Installation

### Basic Installation

```bash
pip install strata-fdtd
```

### With Native C++ Kernels (Recommended)

**macOS:**
```bash
brew install cmake libomp
pip install strata-fdtd[native]
```

**Linux:**
```bash
sudo apt install cmake g++ libomp-dev
pip install strata-fdtd[native]
```

**Windows:**
```bash
# Install Visual Studio Build Tools and CMake first
pip install strata-fdtd[native]
```

### Optional Features

```bash
# GPU acceleration (requires CUDA)
pip install strata-fdtd[gpu]

# Geometry modeling and CAD export
pip install strata-fdtd[geometry]

# Full installation (all features)
pip install strata-fdtd[all]
```

## Quick Start

```python
import numpy as np
from strata_fdtd import FDTDSolver, GaussianPulse, PML, UniformGrid
from strata_fdtd.materials import AIR_20C
from strata_fdtd.io import HDF5ResultWriter

# Create simulation grid (100x100x100 cells, 1mm resolution)
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

# Create solver with PML boundaries
solver = FDTDSolver(
    grid=grid,
    material=AIR_20C,
    boundary=PML(thickness=10),
    backend="native"  # Use C++ kernels
)

# Add Gaussian pulse source at center
solver.add_source(GaussianPulse(
    position=(50, 50, 50),
    frequency=1000,  # 1 kHz
    amplitude=1.0,
))

# Add probe to record pressure
solver.add_probe(position=(75, 50, 50), name="output")

# Run simulation
output = HDF5ResultWriter("results.h5")
solver.run(steps=1000, output=output)

# Analyze results
from strata_fdtd.io import HDF5ResultReader
reader = HDF5ResultReader("results.h5")
pressure = reader.read_probe("output")
print(f"Recorded {len(pressure)} samples")
```

## Performance

Strata FDTD delivers exceptional performance through optimized native kernels:

| Grid Size | Python | Native C++ | GPU | Speedup |
|-----------|--------|-----------|-----|---------|
| 50³       | 28 M cells/s | 72 M cells/s | 450 M cells/s | 2.6x / 16x |
| 100³      | 32 M cells/s | 531 M cells/s | 2100 M cells/s | 16.4x / 65x |
| 200³      | 33 M cells/s | 2469 M cells/s | 8500 M cells/s | 75x / 258x |

*Benchmarked on Apple M1 Max (28 threads) and NVIDIA A100 GPU*

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and first simulation
- [API Reference](docs/api-reference.md) - Complete API documentation
- [CLI Reference](docs/cli-reference.md) - Command-line tool usage
- [Viewer Guide](docs/viewer-guide.md) - Visualization desktop app
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Examples

Check out the `examples/` directory for complete examples:

- **basic_pulse.py** - Simple Gaussian pulse propagation
- **sdf_csg/helmholtz_resonator.py** - Helmholtz resonator design
- **sdf_csg/ported_enclosure.py** - Loudspeaker enclosure with port

Run examples via CLI:

```bash
fdtd-compute --script examples/basic_pulse.py --output results.h5
```

## Command-Line Interface

```bash
# Run simulation from Python script
fdtd-compute --script sim.py --output results.h5

# Specify backend
fdtd-compute --script sim.py --backend native --output results.h5

# Control thread count
export OMP_NUM_THREADS=8
fdtd-compute --script sim.py --backend native --output results.h5
```

## Development

### Building from Source

```bash
git clone https://github.com/rjwalters/strata-fdtd.git
cd strata-fdtd
pip install -e ".[dev,native]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Benchmarking

```bash
# Profile FDTD solver
python benchmarks/profile_fdtd.py --size 100 --steps 100

# Benchmark ADE materials
python benchmarks/benchmark_ade.py
```

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

## Related Projects

This package was extracted from [ml-audio-codecs](https://github.com/rjwalters/ml-audio-codecs) to provide a focused, public toolkit for FDTD acoustic simulation.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Strata FDTD in your research, please cite:

```bibtex
@software{strata_fdtd,
  title = {Strata FDTD: FDTD Acoustic Simulation Toolchain},
  author = {Walters, R.},
  year = {2024},
  url = {https://github.com/rjwalters/strata-fdtd}
}
```

## Support

- **GitHub Issues**: https://github.com/rjwalters/strata-fdtd/issues
- **Documentation**: https://github.com/rjwalters/strata-fdtd/tree/main/docs
