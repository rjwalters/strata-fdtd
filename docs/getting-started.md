# Getting Started with FDTD Acoustic Simulator

This guide walks you through your first FDTD (Finite-Difference Time-Domain) acoustic simulation, from installation to visualization.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or later** installed
- **pip** or **[uv](https://docs.astral.sh/uv/)** package manager
- **8GB+ RAM** for running simulations
- **Modern web browser** with WebGL 2.0 support (for visualization)
- **Node.js 18+** and **pnpm** (for web visualizer)

Optional but recommended:
- **C++ compiler** and **CMake** (for 10-20x performance boost with native backend)
- **Multi-core CPU** (for parallel computation)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/rjwalters/ml-audio-codecs.git
cd ml-audio-codecs
```

### Step 2: Install Python Package

**Option A: Using uv (recommended)**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with native performance
uv sync --extra native

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

**Option B: Using pip**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install with native performance
pip install -e ".[native]"
```

### Step 3: Verify Installation

```bash
# Check that the package is installed
python -c "from strata_fdtd import FDTDSolver; print('✓ FDTD installed successfully')"

# Check if native kernels are available (optional but recommended)
python -c "from strata_fdtd.fdtd import has_native_kernels, get_native_info; print(f'Native kernels: {has_native_kernels()}'); print(get_native_info() if has_native_kernels() else 'Install with: pip install -e \".[native]\"')"
```

If native kernels are available, you should see information about OpenMP threads and compiler version.

### Step 4: Install Web Visualizer (Optional)

To visualize simulation results, install the web-based viewer:

```bash
cd viz

# Install pnpm if not already installed
npm install -g pnpm

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

The visualizer will be available at http://localhost:5173.

## Your First Simulation

Let's create a simple simulation of a Gaussian pulse propagating through air.

### Step 1: Create a Simulation Script

Create a new file called `my_first_sim.py`:

```python
"""
My First FDTD Simulation
========================
A Gaussian pulse propagating through air in a 100mm cube.
"""

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
)

# Create the FDTD solver
# - 100x100x100 cells
# - 1mm resolution
# - Total domain: 100mm x 100mm x 100mm
# - CFL condition automatically determines timestep
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm in meters
)

# Add a Gaussian pulse source
# - 40 kHz center frequency
# - Located 25mm from the origin
solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),  # x=25mm, y=50mm, z=50mm (meters)
        frequency=40e3,  # 40 kHz
        amplitude=1.0,
    )
)

# Add a probe to measure pressure
# - Located 75mm from the origin (50mm from source)
solver.add_probe(
    "downstream",
    position=(0.075, 0.05, 0.05),  # x=75mm, y=50mm, z=50mm (meters)
)

# Print simulation info
print(f"Grid shape: {solver.grid.shape}")
print(f"Grid resolution: {solver.grid.resolution*1e3:.2f} mm")
print(f"Timestep: {solver.dt*1e9:.3f} ns")
print(f"Estimated memory: {solver.estimate_memory_mb():.1f} MB")
print(f"Backend: {solver.backend}")
print(f"Threads: {solver.num_threads}")
print()

# Run the simulation
# Duration is in seconds (0.001s = 1ms)
print("Running simulation...")
solver.run(duration=0.001, output_file="results.h5")
print("✓ Simulation complete!")
print(f"Output saved to: results.h5")
```

### Step 2: Run the Simulation

```bash
python my_first_sim.py
```

**Expected output:**
```
Grid shape: (100, 100, 100)
Grid resolution: 1.00 mm
Timestep: 1.942 ns
Estimated memory: 240.0 MB
Backend: native
Threads: 8

Running simulation...
[████████████████████████████████] 100% | 515000/515000 steps
✓ Simulation complete!
Output saved to: results.h5
```

**What just happened?**

1. A 100×100×100 grid was created (1 million cells, ~240 MB)
2. A 40 kHz Gaussian pulse was injected at x=25mm
3. The wave propagated through the grid for 1ms of simulated time (~515,000 timesteps)
4. Pressure fields were saved to `results.h5` every few timesteps
5. The probe recorded pressure over time at x=75mm

### Step 3: Visualize the Results

#### Option A: Using the Web Visualizer

1. Make sure the visualizer is running (`pnpm dev` in the `viz/` directory)
2. Open http://localhost:5173 in your browser
3. Click "Upload Results" and select `results.h5`
4. Use the playback controls to scrub through time
5. Watch the pulse propagate from the source to the probe

**Visualizer features:**
- **Playback controls**: Play, pause, scrub through time
- **Slice planes**: View XY, XZ, or YZ cross-sections
- **Colormaps**: Diverging (red/blue), magnitude, or viridis
- **Probe plots**: Time series and FFT of pressure data
- **Export**: Screenshots (PNG), animations (GIF), or data (CSV/JSON)

#### Option B: Using Python (Quick Analysis)

```python
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load results
with h5py.File("results.h5", "r") as f:
    # Load probe data
    probe_data = f["probes/downstream/pressure"][:]
    probe_time = f["probes/downstream/time"][:]

    # Load a snapshot at t=500 steps
    snapshot = f["snapshots/500"][:]

# Plot probe time series
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(probe_time * 1e6, probe_data)
plt.xlabel("Time (µs)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure at Probe")
plt.grid(True)

# Plot FFT
plt.subplot(1, 2, 2)
dt = probe_time[1] - probe_time[0]
freq = np.fft.rfftfreq(len(probe_data), dt)
fft = np.abs(np.fft.rfft(probe_data))
plt.plot(freq / 1e3, fft)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.xlim(0, 100)
plt.grid(True)

plt.tight_layout()
plt.savefig("probe_analysis.png", dpi=150)
plt.show()

print(f"✓ Plots saved to probe_analysis.png")
```

## Understanding the Results

### Pressure Field Evolution

The simulation captured the propagation of the acoustic pulse:

- **t = 0-200 steps**: Pulse emerges from the source
- **t = 200-600 steps**: Pulse propagates through the domain
- **t = 600-800 steps**: Pulse reaches the probe location
- **t = 800-1000 steps**: Pulse continues and begins to exit the domain

### Probe Data

The probe at x=75mm recorded:

- **Arrival time**: ~600 steps (~1.2 µs)
- **Distance traveled**: 50mm
- **Wave speed**: ~343 m/s (speed of sound in air)
- **Peak frequency**: ~40 kHz (as designed)

### Boundary Effects

You may notice reflections from the edges of the domain. The simulation uses:

- **PML boundaries** (Perfectly Matched Layer) to absorb outgoing waves
- Some residual reflections are normal for short simulations
- Increase PML thickness or grid size to minimize reflections

## Next Steps

Now that you've run your first simulation, try these exercises:

### Exercise 1: Change the Source Frequency

Modify the `GaussianPulse` frequency to 100 kHz:

```python
solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),
        frequency=100e3,  # 100 kHz instead of 40 kHz
    )
)
```

**Questions:**
- How does the wavelength change?
- Does the pulse arrive at the same time?
- How does the FFT spectrum change?

### Exercise 2: Add More Probes

Add probes at different distances:

```python
solver.add_probe("near", position=(0.035, 0.05, 0.05))    # 10mm from source
solver.add_probe("mid", position=(0.055, 0.05, 0.05))     # 30mm from source
solver.add_probe("far", position=(0.075, 0.05, 0.05))     # 50mm from source
```

**Questions:**
- How does the arrival time scale with distance?
- Can you measure the speed of sound from the time differences?
- How does the pulse amplitude decay with distance?

### Exercise 3: Explore Longer Simulations

Try increasing the simulation duration to observe more wave behavior:

```python
# Run for longer to see multiple reflections
solver.run(duration=0.005, output_file="results.h5")  # 5ms instead of 1ms
```

**Questions:**
- How many times does the wave traverse the domain?
- Do you see reflections from the boundaries?
- How does the PML boundary affect wave absorption?

### Exercise 4: Use Nonuniform Grids

Use a nonuniform grid to save memory while maintaining fine resolution near the source:

```python
from strata_fdtd import NonuniformGrid, FDTDSolver, GaussianPulse

# Fine resolution near center, coarser toward edges
grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 100),
    base_resolution=1e-3,
    stretch_z=1.05,  # 5% growth per cell in z
    center_fine=True
)

# Create solver with nonuniform grid
solver = FDTDSolver(grid=grid)
solver.add_source(GaussianPulse(position=(0.025, 0.05, 0.05), frequency=40e3))
solver.add_probe("downstream", position=(0.075, 0.05, 0.05))
solver.run(duration=0.001, output_file="results.h5")
```

**Questions:**
- How much memory does this save compared to uniform grid?
- How does the CFL timestep change?
- Does the simulation still capture the physics correctly?

## Common Pitfalls

### Memory Errors

**Problem**: `MemoryError: Unable to allocate array`

**Solutions:**
- Reduce grid size (e.g., `shape=(50, 50, 50)` instead of `(100, 100, 100)`)
- Increase resolution (e.g., `resolution=2e-3` instead of `1e-3`)
- Use nonuniform grids to concentrate resolution where needed
- Close other applications to free RAM

### Slow Performance

**Problem**: Simulation takes too long (>5 minutes for small grids)

**Solutions:**
- Ensure native backend is installed: `pip install -e ".[native]"`
- Check backend: `print(solver.backend)` should show `"native"` not `"python"`
- Increase thread count: `solver.set_num_threads(8)` or set `OMP_NUM_THREADS=8`
- Reduce grid size or number of timesteps

### Unexpected Results

**Problem**: Waves behaving strangely, numerical instability

**Solutions:**
- Check CFL condition: `print(solver.dt)` - should be small (~1e-9 seconds)
- Ensure resolution is fine enough: wavelength should span at least 10 cells
  - Rule of thumb: `resolution < wavelength / 10 = c / (10 * frequency)`
  - For 40 kHz in air: `resolution < 343 / (10 * 40000) ≈ 0.86mm`
- Increase PML boundary thickness if seeing reflections
- Check that materials have realistic properties

## Further Reading

- **[Builder Mode Guide](builder-guide.md)** - Use the web interface to build simulations visually
- **[CLI Reference](cli-reference.md)** - Command-line tool for running simulations
- **[Viewer Mode Guide](viewer-guide.md)** - Advanced visualization techniques
- **[API Reference](api-reference.md)** - Complete Python API documentation
- **[Troubleshooting](troubleshooting.md)** - Solutions to common problems

## Example Gallery

Explore pre-built simulations in the `examples/` directory:

- `basic_pulse.py` - Simple Gaussian pulse (this tutorial)
- `pzt_transducer.py` - Piezoelectric transducer array
- `organ_pipes.py` - Helmholtz resonator array
- `nonuniform_grid.py` - Adaptive resolution example
- `materials_demo.py` - Different material properties

Run any example:

```bash
python examples/basic_pulse.py
```

## Getting Help

- **Documentation**: See [docs/](../) directory
- **Examples**: See [examples/](../examples/) directory
- **Issues**: [GitHub Issues](https://github.com/rjwalters/ml-audio-codecs/issues)
- **API Docs**: [api-reference.md](api-reference.md)

Happy simulating!
