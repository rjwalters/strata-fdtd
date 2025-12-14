# Strata FDTD

FDTD acoustic simulation toolchain for loudspeaker design and metamaterials.

## Overview

Strata FDTD provides high-performance finite-difference time-domain (FDTD) simulation tools for acoustic wave propagation. Designed for loudspeaker cabinet design, metamaterial development, and general acoustic modeling.

## Features

- 3D FDTD acoustic wave solver
- Native C++ kernels with OpenMP parallelization
- Nonuniform grids for efficient computation
- PML absorbing boundaries
- Material modeling (frequency-dependent absorption)
- Interactive web-based visualization
- Desktop application for simulation management

## Installation

```bash
pip install strata-fdtd
```

## Quick Start

```python
from strata import FDTDSolver

# Create solver
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm grid spacing
)

# Add a point source
solver.add_source(position=(50, 50, 50), frequency=1000)

# Run simulation
solver.run(duration=0.01)

# Visualize results
solver.plot()
```

## Documentation

Full documentation coming soon.

## License

MIT License - see LICENSE file for details.

## Status

⚠️ **Pre-release**: This repository is under active development as part of the migration from `ml-audio-codecs`. The package will be published to PyPI once the migration is complete.

## Repository

Part of the Strata Audio ecosystem.
