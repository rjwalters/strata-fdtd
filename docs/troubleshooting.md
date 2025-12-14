# Troubleshooting Guide

Solutions to common problems when using the FDTD Acoustic Simulator.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Simulation Setup](#simulation-setup)
- [Runtime Errors](#runtime-errors)
- [Performance Problems](#performance-problems)
- [Visualization Issues](#visualization-issues)
- [Numerical Issues](#numerical-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Problem: Native Backend Won't Build

**Symptoms:**
```
ERROR: Failed building wheel for ml-audio-codecs
CMake Error: ...
```

**Solutions:**

1. **Check compiler availability:**
```bash
# On macOS
xcode-select --install

# On Linux (Ubuntu/Debian)
sudo apt install build-essential cmake

# On Linux (RHEL/CentOS)
sudo yum groupinstall "Development Tools"
sudo yum install cmake
```

2. **Install OpenMP:**
```bash
# On macOS
brew install libomp

# On Linux
sudo apt install libomp-dev  # Ubuntu/Debian
sudo yum install libomp-devel  # RHEL/CentOS
```

3. **Try building without native:**
```bash
# Skip native kernels, use pure Python
pip install -e .  # Without [native]
```

4. **Check CMake version:**
```bash
cmake --version  # Should be 3.15+
```

### Problem: Import Error After Installation

**Symptoms:**
```python
>>> from strata_fdtd import FDTDSolver
ModuleNotFoundError: No module named 'strata_fdtd'
```

**Solutions:**

1. **Check virtual environment:**
```bash
# Ensure you're in the right venv
which python
pip list | grep ml-audio-codecs
```

2. **Reinstall in development mode:**
```bash
pip install -e .
```

3. **Check PYTHONPATH:**
```bash
# Should include your project directory
echo $PYTHONPATH
```

### Problem: Missing Dependencies

**Symptoms:**
```
ImportError: No module named 'numpy'
ImportError: No module named 'h5py'
```

**Solutions:**

```bash
# Install all dependencies
pip install -e ".[native,dev]"

# Or minimal install
pip install -e .
```

---

## Simulation Setup

### Problem: Grid Too Large (Memory Error Before Simulation)

**Symptoms:**
```python
MemoryError: Unable to allocate 64.0 GB for pressure field
```

**Solutions:**

1. **Reduce grid size:**
```python
# Before: 300³ = 27M cells
grid = UniformGrid(shape=(300, 300, 300), resolution=1e-3)

# After: 200³ = 8M cells (3x less memory)
grid = UniformGrid(shape=(200, 200, 200), resolution=1e-3)
```

2. **Increase resolution (coarser grid):**
```python
# Before: 200³ @ 1mm = 8M cells
grid = UniformGrid(shape=(200, 200, 200), resolution=1e-3)

# After: 200³ @ 2mm = 1M cells (8x less memory)
grid = UniformGrid(shape=(200, 200, 200), resolution=2e-3)
```

3. **Use nonuniform grids:**
```python
# Fine resolution only where needed
grid = NonuniformGrid.from_stretch(
    shape=(200, 200, 200),
    base_resolution=1e-3,
    stretch_z=1.05,  # Coarser toward edges
    center_fine=True
)
```

**Memory estimates:**
| Grid Size | Memory (approx) |
|-----------|----------------|
| 50³       | 4 MB           |
| 100³      | 30 MB          |
| 150³      | 100 MB         |
| 200³      | 240 MB         |
| 300³      | 800 MB         |
| 500³      | 3.7 GB         |
| 1000³     | 30 GB          |

### Problem: CFL Timestep Too Small

**Symptoms:**
```
Timestep: 1.2e-11 s (very small)
Estimated runtime: 45 hours
```

**Cause:** Grid resolution is too fine relative to sound speed.

**Solutions:**

1. **Increase resolution (coarser grid):**
```python
# Before: 0.1mm resolution → dt ~ 1e-11 s
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-4)

# After: 1mm resolution → dt ~ 1e-9 s (100x larger)
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
```

2. **Reduce duration (fewer steps):**
```python
# Before: 100,000 steps
solver = FDTDSolver.from_scene(scene, duration=100000)

# After: 10,000 steps (10x faster)
solver = FDTDSolver.from_scene(scene, duration=10000)
```

**CFL timestep formula:**
```
dt = (min_spacing) / (sqrt(3) × c)

where:
  min_spacing = grid resolution (m)
  c = sound speed (343 m/s in air)
```

### Problem: Source Not Producing Waves

**Symptoms:**
Simulation runs but pressure field remains zero or very small.

**Causes & Solutions:**

1. **Source frequency too low for grid:**
```python
# Wavelength λ = c / f
# Rule: λ should span at least 10 cells

# Example: 1 kHz in air
f = 1e3  # 1 kHz
c = 343  # m/s
wavelength = c / f  # 0.343 m = 343 mm

# Resolution should be < wavelength / 10
resolution = wavelength / 10  # 34 mm

# If your resolution is 1mm, wavelength spans 343 cells ✓ Good
# If your resolution is 50mm, wavelength spans only 6.9 cells ✗ Bad
```

2. **Source positioned outside grid:**
```python
# Check that source position is within grid bounds
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
# Grid extent: x ∈ [0, 0.1m], y ∈ [0, 0.1m], z ∈ [0, 0.1m]

# This is OUTSIDE the grid ✗
scene.add_source(GaussianPulse(frequency=40e3, position=(0.15, 0.05, 0.05)))

# This is INSIDE the grid ✓
scene.add_source(GaussianPulse(frequency=40e3, position=(0.05, 0.05, 0.05)))
```

3. **Amplitude too small:**
```python
# Increase amplitude
scene.add_source(
    GaussianPulse(
        frequency=40e3,
        position=(0.025, 0.05, 0.05),
        amplitude=10.0  # Increase from default 1.0
    )
)
```

---

## Runtime Errors

### Problem: Out of Memory During Simulation

**Symptoms:**
```
Killed (OOM)
```

**Solutions:**

1. **Close other applications** to free RAM

2. **Reduce snapshot frequency:**
```python
# Before: Save every 10 steps (lots of snapshots)
solver = FDTDSolver.from_scene(scene, duration=1000, snapshot_interval=10)

# After: Save every 100 steps
solver = FDTDSolver.from_scene(scene, duration=1000, snapshot_interval=100)
```

3. **Reduce grid size** (see "Grid Too Large" above)

4. **Use swap space** (slower but prevents crash):
```bash
# On Linux, check swap
swapon --show

# Add swap file (4GB)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Problem: Simulation Crashes with NaN or Inf

**Symptoms:**
```
RuntimeError: NaN detected in pressure field at step 523
```

**Causes & Solutions:**

1. **CFL condition violated** (timestep too large):
```python
# Check timestep
print(f"dt = {solver.dt}")
print(f"CFL dt = {grid.min_spacing / (np.sqrt(3) * 343)}")

# If dt > CFL dt, the solver should auto-fix this
# If not, manually reduce timestep (advanced):
solver.dt = grid.min_spacing / (np.sqrt(3) * 343) * 0.9  # 90% of CFL limit
```

2. **Material properties invalid:**
```python
# Check that material properties are positive
print(materials.custom_material.c)    # Should be > 0
print(materials.custom_material.rho)  # Should be > 0

# Bad example ✗
bad_material = DebyeModel(c=-343, rho=1.2)  # Negative sound speed!

# Good example ✓
good_material = DebyeModel(c=343, rho=1.2)
```

3. **Source amplitude too large:**
```python
# Reduce amplitude to reasonable physical values
scene.add_source(
    GaussianPulse(
        frequency=40e3,
        position=(0.025, 0.05, 0.05),
        amplitude=1.0  # Pa (reasonable for acoustics)
        # NOT amplitude=1e20  # Unphysically large!
    )
)
```

### Problem: Simulation Hangs or Freezes

**Symptoms:**
Progress bar stops updating, no CPU activity.

**Solutions:**

1. **Interrupt with Ctrl+C** and check last output

2. **Run with verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

solver.run(output_file="results.h5")
```

3. **Check for deadlock** (rare, contact developers if persistent)

---

## Performance Problems

### Problem: Simulation Too Slow (Pure Python Backend)

**Symptoms:**
```
Backend: python
Speed: 0.05 Mcells/s (very slow)
```

**Solutions:**

1. **Install native backend:**
```bash
pip install -e ".[native]"
```

2. **Verify native backend:**
```python
from strata_fdtd.fdtd import has_native_kernels
print(has_native_kernels())  # Should be True
```

3. **Force native backend:**
```python
solver = FDTDSolver.from_scene(scene, duration=1000, backend="native")
```

**Expected speeds:**
| Backend | Speed (100³ grid) |
|---------|-------------------|
| Python  | 0.05 Mcells/s     |
| Native (1 thread) | 0.5 Mcells/s |
| Native (8 threads) | 3-6 Mcells/s |
| Native (16 threads) | 5-10 Mcells/s |

### Problem: Native Backend Not Using All Cores

**Symptoms:**
```
Threads: 8
CPU usage: ~12% (on 8-core machine, should be ~100%)
```

**Solutions:**

1. **Set thread count manually:**
```python
solver.set_num_threads(8)
```

2. **Use environment variable:**
```bash
export OMP_NUM_THREADS=8
python my_sim.py
```

3. **Check thread affinity:**
```bash
# On Linux
export OMP_PROC_BIND=close
python my_sim.py
```

### Problem: Diminishing Returns with More Threads

**Symptoms:**
Adding more threads doesn't speed up simulation proportionally.

**Cause:** Amdahl's Law - parallel speedup limited by serial portions.

**Solutions:**

1. **Use optimal thread count** (usually 8-16):
```python
# Test different thread counts
for t in [1, 2, 4, 8, 16]:
    solver.set_num_threads(t)
    start = time.time()
    solver.run()
    elapsed = time.time() - start
    print(f"{t} threads: {elapsed:.2f} s")
```

2. **For small grids** (<50³), use fewer threads (1-4)
3. **For large grids** (>200³), use more threads (8-16)

---

## Visualization Issues

### Problem: Viewer Won't Load Results File

**Symptoms:**
```
Failed to load results
Error: Invalid HDF5 file
```

**Solutions:**

1. **Check file format:**
```bash
# Must be .h5 or .hdf5
file results.h5  # Should show "Hierarchical Data Format"
```

2. **Verify file structure:**
```bash
h5dump -n results.h5  # List dataset names
```

Should contain:
- `/metadata/`
- `/snapshots/`
- `/probes/`

3. **Re-run simulation** if file is corrupted

4. **Check file size limit** (browser upload ~2GB max)

### Problem: Viewer Laggy or Slow

**Symptoms:**
Low FPS, stuttering playback, laggy camera rotation.

**Solutions:**

1. **Reduce threshold** (hide low-amplitude voxels):
```
Threshold slider → 0.2 (hide voxels < 20% of max)
```

2. **Reduce display fill**:
```
Display Fill slider → 0.2 (show only 20% of voxels)
```

3. **Use Point geometry** instead of Mesh

4. **Close other browser tabs**

5. **Enable hardware acceleration** in browser:
   - Chrome: `chrome://settings/` → Advanced → System → Use hardware acceleration

6. **Reduce grid size** for future simulations

### Problem: Probe Data Not Showing

**Symptoms:**
Probe dropdown is empty or plots are blank.

**Solutions:**

1. **Ensure probes were added:**
```python
scene.add_probe(position=(0.075, 0.05, 0.05), name="probe1")
```

2. **Check HDF5 file:**
```bash
h5dump -n results.h5 | grep probes
```

3. **Verify probe names** are unique

4. **Re-run simulation** if probes are missing

---

## Numerical Issues

### Problem: Strong Reflections from Boundaries

**Symptoms:**
Pressure waves reflect strongly from domain edges, contaminating results.

**Solutions:**

1. **Use PML boundaries** (default, but check thickness):
```python
from strata_fdtd import PML

solver = FDTDSolver.from_scene(scene, duration=1000)
solver.set_boundary(PML(thickness=20))  # Increase from 10 to 20
```

2. **Increase domain size** (more space before boundaries)

3. **Use nonuniform grids** (stretch cells toward boundaries for gradual impedance matching)

### Problem: Dispersion (Wave Speed Varies with Frequency)

**Symptoms:**
High-frequency components travel at different speeds than low-frequency.

**Cause:** Numerical dispersion in FDTD (inherent to method).

**Solutions:**

1. **Increase resolution** (finer grid reduces dispersion):
```python
# Before: 10 cells per wavelength
resolution = wavelength / 10

# After: 20 cells per wavelength (less dispersion)
resolution = wavelength / 20
```

2. **Lower source frequency** (longer wavelengths are better resolved)

3. **Accept small dispersion** for typical acoustics (<1% error at 10 cells/wavelength)

### Problem: Artificial Attenuation

**Symptoms:**
Waves decay faster than expected, no energy conservation.

**Causes & Solutions:**

1. **Lossy materials** (check material properties):
```python
# Ensure materials don't have unintended loss
print(materials.custom.alpha_debye)  # Should be 0 for lossless
```

2. **PML absorbing energy** (expected at boundaries)

3. **Numerical attenuation** (reduce by increasing resolution)

---

## Getting Help

### Before Asking for Help

1. **Check this guide** for common issues
2. **Read error messages carefully** (often contain solutions)
3. **Try minimal example** (simplify to isolate problem)
4. **Verify installation**:
```python
from strata_fdtd.fdtd import has_native_kernels, get_native_info
print(f"Native: {has_native_kernels()}")
print(get_native_info() if has_native_kernels() else {})
```

5. **Check package version**:
```bash
pip show ml-audio-codecs
```

### Reporting Bugs

When reporting issues, include:

1. **Error message** (full traceback)
2. **Minimal reproducible example** (short script that triggers error)
3. **System information**:
```python
import sys, platform, numpy, h5py
from strata_fdtd import fdtd

print(f"Python: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"NumPy: {numpy.__version__}")
print(f"h5py: {h5py.__version__}")
print(f"Native kernels: {fdtd.has_native_kernels()}")
if fdtd.has_native_kernels():
    print(fdtd.get_native_info())
```

4. **What you expected** vs. **what happened**

### Where to Ask

- **GitHub Issues**: [ml-audio-codecs/issues](https://github.com/rjwalters/ml-audio-codecs/issues)
- **Documentation**: See [docs/](../) directory
- **Examples**: See [examples/](../examples/) directory

---

## Quick Diagnostic Checklist

**Installation:**
- [ ] Python 3.10+ installed?
- [ ] Virtual environment activated?
- [ ] Package installed (`pip list | grep ml-audio-codecs`)?
- [ ] Native backend available (`has_native_kernels()`)?

**Simulation Setup:**
- [ ] Grid size reasonable (<8GB memory)?
- [ ] Source inside grid boundaries?
- [ ] Wavelength spans ≥10 cells?
- [ ] Material properties valid (c>0, rho>0)?

**Performance:**
- [ ] Using native backend (not Python)?
- [ ] Thread count set (`OMP_NUM_THREADS` or `set_num_threads`)?
- [ ] Grid size appropriate for hardware?

**Results:**
- [ ] Output file created?
- [ ] HDF5 file valid (`h5dump -n results.h5`)?
- [ ] Probes defined with names?
- [ ] Snapshots not empty?

**Visualization:**
- [ ] File size <2GB (for browser upload)?
- [ ] Viewer shows geometry and sources?
- [ ] Probe data visible in plots?
- [ ] Performance acceptable (>15 FPS)?

---

## Further Reading

- **[Getting Started Guide](getting-started.md)** - Tutorial and examples
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Builder Mode Guide](builder-guide.md)** - Visual editor
- **[CLI Reference](cli-reference.md)** - Command-line tool
- **[Viewer Mode Guide](viewer-guide.md)** - Visualization techniques
