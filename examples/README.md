# FDTD Simulation Examples

This directory contains example simulation scripts demonstrating various features of the strata_fdtd FDTD simulator.

## Running Examples

```bash
# Run any example directly
python examples/basic_pulse.py

# Or using the CLI tool (once implemented)
fdtd-compute examples/basic_pulse.py
```

## Example Index

### Basic Examples

| Example | Description | Grid Size | Runtime | Features |
|---------|-------------|-----------|---------|----------|
| `basic_pulse.py` | Gaussian pulse in free field | 100³ | ~30s | Basic FDTD, source, probe |
| `multiple_probes.py` | Distance-dependent arrival times | 100³ | ~30s | Multiple probes, analysis |
| `material_sphere.py` | Scattering from water sphere | 100³ | ~45s | Materials, scattering |
| `nonuniform_grid.py` | Adaptive resolution demonstration | 100³ | ~40s | Nonuniform grids |

### Loudspeaker Design Examples

| Example | Description | Features |
|---------|-------------|----------|
| `optimize_horn.py` | Parametric horn optimization | Parametric geometry, optimization |
| `strata_midrange_chamber.py` | Metamaterial chamber design | Laminated geometry, Helmholtz |
| `sdf_csg/helmholtz_resonator.py` | SDF-based resonator | SDF primitives, CSG |
| `sdf_csg/ported_enclosure.py` | Ported loudspeaker enclosure | SDF, complex geometry |
| `sdf_csg/metamaterial_unit_cell.py` | Periodic strata_fdtd | SDF, unit cells |

### Advanced Examples

| Example | Description | Grid Size | Features |
|---------|-------------|-----------|----------|
| `pzt_transducer.py` | Piezoelectric transducer array | 150³ | PZT materials, arrays |
| `organ_pipes.py` | Helmholtz resonator array | 200³ | Resonator arrays, harmonics |
| `waveguide.py` | Rectangular acoustic waveguide | 64×64×256 | Waveguide modes |
| `frequency_sweep.py` | Chirp source for bandwidth test | 100³ | Chirp source, FFT analysis |

## Example Categories

### 1. Getting Started
- **basic_pulse.py** - Your first FDTD simulation
- **multiple_probes.py** - Understanding wave propagation

Start here if you're new to FDTD.

### 2. Materials and Geometry
- **material_sphere.py** - Adding solid objects
- **sdf_csg/** - Complex geometries with CSG

Learn how to build 3D structures.

### 3. Advanced Techniques
- **nonuniform_grid.py** - Memory-efficient simulations
- **pzt_transducer.py** - Frequency-dependent materials
- **organ_pipes.py** - Resonant systems

Explore advanced solver features.

### 4. Loudspeaker Design
- **optimize_horn.py** - Parametric optimization
- **strata_midrange_chamber.py** - Metamaterial enclosures

Design acoustic devices.

## Creating Your Own Examples

Use this template:

```python
"""
Example: [Brief description]
==========================
[Longer description of what this demonstrates]

Expected runtime: ~X seconds on modern hardware
Output: results.h5
"""

from strata_fdtd import FDTDSolver, GaussianPulse

# 1. Create solver with grid parameters
solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)

# 2. Add sources and probes
solver.add_source(GaussianPulse(
    position=(0.025, 0.05, 0.05),
    frequency=40e3,
))
solver.add_probe("probe1", position=(0.075, 0.05, 0.05))

# 3. Run simulation
print(f"Estimated memory: {solver.estimate_memory_mb():.1f} MB")
solver.run(duration=0.001, output_file="results.h5")

print("✓ Simulation complete!")
print("Upload results.h5 to the Viewer to visualize.")
```

## Example Naming Convention

- **Purpose-based**: `feature_test.py` (e.g., `nonuniform_grid.py`)
- **Application-based**: `device_type.py` (e.g., `pzt_transducer.py`)
- **Descriptive**: Clear, lowercase with underscores

## Output Files

Each example generates:
- **results_[hash].h5** - HDF5 file with pressure snapshots and probe data
- **[example]_analysis.png** - Optional analysis plots (if script creates them)

**Note**: Results files are gitignored (.h5 files are large).

## Getting Help

- **Documentation**: See [docs/](../docs/) directory
- **Getting Started**: [docs/getting-started.md](../docs/getting-started.md)
- **API Reference**: [docs/api-reference.md](../docs/api-reference.md)
- **Troubleshooting**: [docs/troubleshooting.md](../docs/troubleshooting.md)

## Contributing Examples

To contribute a new example:

1. Create a well-documented script in `examples/`
2. Follow the naming convention
3. Include docstring with description and expected runtime
4. Test on a standard machine (document actual runtime)
5. Add entry to this README
6. Submit PR with label `documentation`

---

**Note**: Some examples listed above are planned but not yet implemented. Check the actual files in this directory for currently available examples.
