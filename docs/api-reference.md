# API Reference

Complete reference for the `strata_fdtd` Python package for FDTD acoustic simulation.

## Quick Start

```python
from strata_fdtd import FDTDSolver, GaussianPulse

# Create solver with grid
solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)

# Add source and probe
solver.add_source(GaussianPulse(position=(0.025, 0.05, 0.05), frequency=40e3))
solver.add_probe("probe1", position=(0.075, 0.05, 0.05))

# Run simulation (duration in seconds)
solver.run(duration=0.001, output_file="results.h5")
```

---

## Core Classes

### Grid Specifications

#### UniformGrid

Uniform rectangular grid with constant cell spacing.

```python
from strata_fdtd import UniformGrid

grid = UniformGrid(
    shape=(Nx, Ny, Nz),    # Number of cells in each direction
    resolution=dx          # Cell size in meters
)
```

**Parameters:**
- **shape** (`tuple[int, int, int]`): Grid dimensions `(Nx, Ny, Nz)`
- **resolution** (`float`): Cell spacing in meters (e.g., `1e-3` = 1mm)

**Attributes:**
- **shape** (`tuple[int, int, int]`): Grid dimensions
- **resolution** (`float`): Cell spacing (m)
- **min_spacing** (`float`): Minimum cell size (= resolution for uniform)
- **max_spacing** (`float`): Maximum cell size (= resolution for uniform)
- **is_uniform** (`bool`): Always `True`
- **dx**, **dy**, **dz** (`np.ndarray`): Cell spacings per axis (all equal to resolution)
- **x_coords**, **y_coords**, **z_coords** (`np.ndarray`): Cell face coordinates

**Methods:**
- **physical_extent()** → `tuple[float, float, float]`: Domain size in meters `(Lx, Ly, Lz)`

**Example:**
```python
# 100mm x 100mm x 100mm grid, 1mm resolution
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

print(grid.shape)             # (100, 100, 100)
print(grid.physical_extent()) # (0.1, 0.1, 0.1) meters
print(grid.min_spacing)       # 0.001 meters
```

#### NonuniformGrid

Nonuniform grid with variable cell spacing (adaptive resolution).

**Method 1: Geometric stretch**

```python
from strata_fdtd import NonuniformGrid

grid = NonuniformGrid.from_stretch(
    shape=(Nx, Ny, Nz),
    base_resolution=dx,
    stretch_x=1.0,        # No stretch in x (uniform)
    stretch_y=1.0,        # No stretch in y (uniform)
    stretch_z=1.05,       # 5% growth per cell in z
    center_fine=True      # Finest cells at center (default)
)
```

**Parameters:**
- **shape** (`tuple[int, int, int]`): Grid dimensions
- **base_resolution** (`float`): Finest cell size (m)
- **stretch_x**, **stretch_y**, **stretch_z** (`float`): Growth factor per cell (1.0 = uniform)
- **center_fine** (`bool`): If `True`, finest cells at center; if `False`, finest at origin

**Method 2: Piecewise regions**

```python
grid = NonuniformGrid.from_regions(
    x_regions=[
        (0.0, 0.05, 2e-3),     # x ∈ [0, 50mm]: 2mm spacing
        (0.05, 0.15, 1e-3),    # x ∈ [50, 150mm]: 1mm spacing (fine)
        (0.15, 0.2, 2e-3),     # x ∈ [150, 200mm]: 2mm spacing
    ],
    y_regions=[(0.0, 0.1, 1e-3)],  # Uniform 1mm in y
    z_regions=[(0.0, 0.1, 1e-3)],  # Uniform 1mm in z
)
```

**Method 3: Direct coordinate arrays**

```python
import numpy as np

x = np.linspace(0, 0.1, 100)
y = np.linspace(0, 0.1, 100)
z = np.geomspace(0.001, 0.2, 200)  # Logarithmic spacing in z

grid = NonuniformGrid(x_coords=x, y_coords=y, z_coords=z)
```

**Attributes:**
Same as `UniformGrid`, plus:
- **stretch_ratio** (`tuple[float, float, float]`): Ratio of max/min spacing per axis

**Example:**
```python
# Fine resolution at center, coarse at edges
grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 200),
    base_resolution=1e-3,
    stretch_z=1.05,
    center_fine=True
)

print(grid.min_spacing)  # 0.001 meters (at center)
print(grid.max_spacing)  # ~0.0025 meters (at edges)
print(grid.stretch_ratio)  # (1.0, 1.0, ~2.5)
```

**When to use:**
- Large domains with localized features (sources, geometry)
- Memory-constrained simulations
- Gradual impedance matching (PML boundaries)

**Limitations:**
- CFL timestep determined by minimum spacing (finest cells)
- Native backend not optimized for nonuniform grids (use Python backend)

---

### Sources

#### GaussianPulse

Broadband Gaussian-modulated pulse source.

```python
from strata_fdtd import GaussianPulse

source = GaussianPulse(
    frequency=f0,          # Center frequency (Hz)
    position=(x, y, z),    # Source location (m)
    amplitude=A,           # Pressure amplitude (Pa)
    t0=None,               # Pulse arrival time (s), auto if None
    sigma=None             # Pulse width (s), auto if None
)
```

**Parameters:**
- **frequency** (`float`): Center frequency (Hz)
- **position** (`tuple[float, float, float]`): Source location (m)
- **amplitude** (`float`, optional): Pressure amplitude (Pa, default: 1.0)
- **t0** (`float`, optional): Arrival time (s, default: 5 × pulse width)
- **sigma** (`float`, optional): Pulse width (s, default: 2 / frequency)

**Use case:** Broadband excitation, impulse response, transient analysis

**Example:**
```python
# 40 kHz pulse at (25mm, 50mm, 50mm)
pulse = GaussianPulse(
    frequency=40e3,
    position=(0.025, 0.05, 0.05)
)
```

**Time-domain waveform:**
```
p(t) = A × exp(-((t - t0) / sigma)²) × sin(2π × f0 × t)
```

**Frequency content:** Centered at `f0` with bandwidth ~`1 / sigma`

---

### FDTD Solver

#### FDTDSolver

3D acoustic wave solver using Finite-Difference Time-Domain method.

**Method 1: With shape and resolution**

```python
from strata_fdtd import FDTDSolver

solver = FDTDSolver(
    shape=(100, 100, 100),   # Grid dimensions
    resolution=1e-3,         # Cell size in meters
    backend="auto"           # "auto", "native", or "python"
)
```

**Method 2: With grid object**

```python
from strata_fdtd import UniformGrid, FDTDSolver

grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
solver = FDTDSolver(grid=grid)
```

**Parameters:**
- **shape** (`tuple[int, int, int]`): Grid dimensions (Nx, Ny, Nz)
- **resolution** (`float`): Cell size in meters
- **grid** (`UniformGrid | NonuniformGrid`): Pre-constructed grid object
- **c** (`float`, optional): Speed of sound (m/s, default: 343)
- **rho** (`float`, optional): Density (kg/m³, default: 1.2)
- **backend** (`str`, optional): Computation backend (default: "auto")

**Attributes:**
- **dt** (`float`): Timestep (s), determined by CFL condition
- **grid** (`Grid`): Simulation grid
- **backend** (`str`): Current backend ("native" or "python")
- **num_threads** (`int`): OpenMP thread count (native backend only)
- **using_native** (`bool`): True if native backend active

**Methods:**

##### add_source

Add acoustic source to the simulation.

```python
solver.add_source(source)
```

**Parameters:**
- **source** (`GaussianPulse | Sinusoidal | Chirp`): Source object

**Example:**
```python
from strata_fdtd import GaussianPulse

solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),
        frequency=40e3,
        amplitude=1.0
    )
)
```

##### add_probe

Add pressure measurement probe.

```python
solver.add_probe(name, position)
```

**Parameters:**
- **name** (`str`): Probe identifier
- **position** (`tuple[float, float, float]`): Probe location in meters

**Example:**
```python
solver.add_probe("downstream", position=(0.075, 0.05, 0.05))
solver.add_probe("upstream", position=(0.025, 0.05, 0.05))
```

##### run

Execute simulation.

```python
solver.run(
    duration=0.001,             # Simulation duration in seconds
    output_file="results.h5",   # HDF5 output path
    snapshot_interval=100,      # Save pressure every N steps
    callback=None               # Optional progress callback
)
```

**Parameters:**
- **duration** (`float`): Simulation duration in seconds
- **output_file** (`str`, optional): HDF5 output path (default: no output)
- **snapshot_interval** (`int`, optional): Save pressure every N steps (default: 100)
- **callback** (`callable`, optional): Function called each step: `callback(step_number)`

**Returns:** None

**Output:** HDF5 file containing pressure snapshots and probe data

**Example:**
```python
# Run for 1 millisecond
solver.run(duration=0.001, output_file="results.h5")
```

**With progress callback:**
```python
def progress(step):
    if step % 100 == 0:
        print(f"Step {step}")

solver.run(duration=0.001, output_file="results.h5", callback=progress)
```

##### estimate_memory_mb

Estimate peak memory usage.

```python
memory_mb = solver.estimate_memory_mb()
```

**Returns:** `float` — Estimated memory (MB)

**Example:**
```python
if solver.estimate_memory_mb() > 8000:
    print("Warning: Simulation requires >8GB RAM")
```

##### set_num_threads

Set OpenMP thread count (native backend only).

```python
solver.set_num_threads(n)
```

**Parameters:**
- **n** (`int`): Number of threads

**Example:**
```python
solver.set_num_threads(8)
```

---

### Probes and Microphones

#### Probe

Simple pressure measurement point.

```python
from strata_fdtd import Probe

probe = Probe(
    position=(x, y, z),
    name="probe_name"
)
```

**Parameters:**
- **position** (`tuple[float, float, float]`): Probe location (m)
- **name** (`str`, optional): Probe identifier

**Use case:** Generic pressure measurement

#### Microphone

Directional pressure measurement with polar pattern.

```python
from strata_fdtd import Microphone, POLAR_PATTERNS

mic = Microphone(
    position=(x, y, z),
    orientation=(theta, phi),  # Spherical angles (radians)
    pattern=POLAR_PATTERNS["cardioid"],
    name="mic_name"
)
```

**Parameters:**
- **position** (`tuple[float, float, float]`): Microphone location (m)
- **orientation** (`tuple[float, float]`): Look direction (θ, φ) in radians
- **pattern** (`str | callable`): Polar pattern (see below)
- **name** (`str`, optional): Microphone identifier

**Polar patterns:**
- **"omnidirectional"**: Equal sensitivity in all directions
- **"cardioid"**: Heart-shaped, rejects rear sound
- **"figure8"**: Bidirectional, front and rear lobes
- **"supercardioid"**: Narrow front lobe, small rear lobe
- **Custom function**: `pattern(theta, phi) → sensitivity` (0-1)

**Example:**
```python
# Cardioid microphone pointing at +x
mic = Microphone(
    position=(0.1, 0.05, 0.05),
    orientation=(0, 0),  # θ=0, φ=0 (+x direction)
    pattern="cardioid",
    name="front_mic"
)
```

---

### Boundaries

#### PML (Perfectly Matched Layer)

Absorbing boundary condition to minimize reflections.

```python
from strata_fdtd import PML, FDTDSolver

solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
solver.set_boundary(PML(thickness=10))
```

**Parameters:**
- **thickness** (`int`): PML layer thickness in cells (default: 10)
- **sigma_max** (`float`, optional): Maximum absorption coefficient (auto if None)
- **kappa_max** (`float`, optional): Stretching parameter (auto if None)
- **alpha_max** (`float`, optional): CFS parameter for evanescent waves (auto if None)

**Use case:** Default boundary for most simulations

**Rule of thumb:** 10-20 cells thick for good absorption

#### RigidBoundary

Hard wall (zero particle velocity at boundary).

```python
from strata_fdtd import RigidBoundary

solver.set_boundary(RigidBoundary())
```

**Use case:** Simulating enclosed cavities, rigid obstacles

#### RadiationImpedance

Impedance boundary for radiating structures.

```python
from strata_fdtd import RadiationImpedance

solver.set_boundary(RadiationImpedance(c=343, rho=1.2))
```

**Parameters:**
- **c** (`float`): Sound speed (m/s)
- **rho** (`float`): Density (kg/m³)

**Use case:** Modeling radiation into infinite medium

---

### Materials

The `materials` module provides predefined and custom material models.

#### Predefined Materials

```python
from strata_fdtd import materials

# Acoustic materials
air = materials.air
water = materials.water

# Solids
aluminum = materials.aluminum
steel = materials.steel
glass = materials.glass

# Piezoelectric ceramics
pzt4 = materials.pzt4
pzt5 = materials.pzt5
```

**Material properties:**
- **c** (`float`): Sound speed (m/s)
- **rho** (`float`): Density (kg/m³)
- **impedance** (`float`): Acoustic impedance Z = ρc (kg/m²/s)

**Example:**
```python
print(materials.water.c)      # 1480 m/s
print(materials.water.rho)    # 1000 kg/m³
print(materials.aluminum.c)   # 6420 m/s
```

#### Custom Materials (Debye Model)

Frequency-dependent lossy material.

```python
from strata_fdtd.materials import DebyeModel

foam = DebyeModel(
    name="acoustic_foam",
    c=340,              # Sound speed (m/s)
    rho=50,             # Density (kg/m³)
    alpha_debye=0.5,    # Loss factor
    tau_debye=1e-6,     # Relaxation time (s)
)
```

**Parameters:**
- **name** (`str`): Material identifier
- **c** (`float`): Sound speed (m/s)
- **rho** (`float`): Density (kg/m³)
- **alpha_debye** (`float`): Absorption coefficient (0-1)
- **tau_debye** (`float`): Relaxation time (s)

**Use case:** Modeling lossy media (foam, tissue, porous materials)

---

## Utility Functions

### has_native_kernels

Check if native C++ backend is available.

```python
from strata_fdtd.fdtd import has_native_kernels

if has_native_kernels():
    print("Native backend available!")
else:
    print("Using pure Python backend (slower)")
```

**Returns:** `bool`

### get_native_info

Get native backend information.

```python
from strata_fdtd.fdtd import get_native_info

info = get_native_info()
print(info["version"])        # Native kernel version
print(info["has_openmp"])     # OpenMP support
print(info["num_threads"])    # Available threads
```

**Returns:** `dict` with keys:
- **version** (`str`): Kernel version
- **has_openmp** (`bool`): OpenMP support
- **num_threads** (`int`): Available threads
- **compiler** (`str`): C++ compiler used

---

## Advanced Topics

### GPU Acceleration (Optional)

Requires PyTorch with CUDA support.

```python
from strata_fdtd import GPUFDTDSolver, has_gpu_support

if has_gpu_support():
    solver = GPUFDTDSolver.from_scene(scene, duration=1000)
    solver.run(output_file="results.h5")
else:
    print("GPU not available, using CPU")
```

**Note:** GPU backend is experimental and requires CUDA-capable GPU.

### Batched GPU Simulation

Run multiple simulations in parallel on GPU.

```python
from strata_fdtd import BatchedGPUFDTDSolver

solver = BatchedGPUFDTDSolver.from_scenes(
    scenes=[scene1, scene2, scene3],
    duration=1000
)
solver.run_batch()
```

**Use case:** Parameter sweeps, Monte Carlo simulations

---

## Complete Example

```python
from strata_fdtd import *

# 1. Create grid (100mm cube, 1mm resolution)
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

# 2. Create solver
solver = FDTDSolver(grid=grid)

# 3. Add Gaussian pulse source
solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),  # 25mm from origin
        frequency=40e3,  # 40 kHz
    )
)

# 4. Add probes
solver.add_probe("far_field", position=(0.075, 0.05, 0.05))
solver.add_probe("center", position=(0.05, 0.05, 0.05))

# 5. Print info
print(f"Grid: {solver.grid.shape}")
print(f"Timestep: {solver.dt*1e9:.2f} ns")
print(f"Memory: {solver.estimate_memory_mb():.1f} MB")
print(f"Backend: {solver.backend}")

# 8. Run simulation
solver.run(output_file="scattering_results.h5")

print("✓ Simulation complete!")
```

---

## Further Reading

- **[Getting Started Guide](getting-started.md)** - Tutorial
- **[Builder Mode Guide](builder-guide.md)** - Visual editor
- **[CLI Reference](cli-reference.md)** - Command-line tool
- **[Viewer Mode Guide](viewer-guide.md)** - Visualization
- **[Troubleshooting](troubleshooting.md)** - Common issues

---

## Function Index

### Grid
- `UniformGrid` - Uniform grid specification
- `NonuniformGrid` - Adaptive resolution grid
- `NonuniformGrid.from_stretch` - Geometric stretch grid
- `NonuniformGrid.from_regions` - Piecewise regional grid

### Sources
- `GaussianPulse` - Broadband pulse source
- `Sinusoidal` - Single-frequency tone
- `Chirp` - Frequency sweep

### Solver
- `FDTDSolver` - Main solver class
- `FDTDSolver.add_source` - Add acoustic source
- `FDTDSolver.add_probe` - Add measurement probe
- `FDTDSolver.run` - Execute simulation
- `FDTDSolver.estimate_memory_mb` - Memory estimation
- `FDTDSolver.set_num_threads` - Set thread count

### Probes
- `Probe` - Simple pressure probe
- `Microphone` - Directional microphone

### Boundaries
- `PML` - Perfectly Matched Layer (absorbing)
- `RigidBoundary` - Hard wall
- `RadiationImpedance` - Radiation boundary

### Materials
- `materials.air` - Air properties
- `materials.water` - Water properties
- `materials.aluminum` - Aluminum properties
- `materials.pzt5` - PZT-5 piezoelectric ceramic
- `DebyeModel` - Custom lossy material

### Utilities
- `has_native_kernels` - Check native backend
- `get_native_info` - Native backend info
- `has_gpu_support` - Check GPU support
- `get_gpu_info` - GPU information
