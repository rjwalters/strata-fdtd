# CLI Reference: fdtd-compute

The `fdtd-compute` command-line tool executes FDTD simulation scripts with real-time progress tracking and resource monitoring.

## Installation

The CLI tool is installed as part of the `strata_fdtd` package:

```bash
# Install with native backend for best performance
pip install -e ".[native]"

# Verify installation
fdtd-compute --version
```

**Expected output:**
```
fdtd-compute version 0.1.0
```

## Basic Usage

```bash
fdtd-compute SCRIPT [OPTIONS]
```

**Arguments:**
- `SCRIPT`: Path to Python simulation script (required)

**Example:**
```bash
fdtd-compute my_simulation.py
```

## Output

The tool generates:
1. **HDF5 result file**: `results_{hash}.h5` containing pressure fields and probe data
2. **Console output**: Real-time progress, statistics, and summary
3. **Exit code**: `0` on success, non-zero on error

**Example run:**
```bash
$ fdtd-compute simulation_abc123de.py

FDTD Simulation: simulation_abc123de.py
────────────────────────────────────────────────────────
Grid:       100 × 100 × 100 (1.0M cells)
Resolution: 1.0 mm
Timestep:   1.65e-9 s
Duration:   1000 steps (1.65 μs)
Backend:    native (16 threads)
Output:     results_abc123de.h5
────────────────────────────────────────────────────────

[████████████░░░░░░░░] 60% | Step 600/1000
Elapsed: 1m 23s | ETA: 0m 55s
Speed: 7.2 Mcells/s | Memory: 1.8 GB / 2.1 GB

────────────────────────────────────────────────────────
✓ Simulation complete!
  Output: results_abc123de.h5 (145 MB)
  Runtime: 2m 18s

Upload to viewer: https://fdtd-viewer.example.com/
```

## Command-Line Options

### --output, -o PATH

Specify custom output file path.

**Default**: `results_{hash}.h5` where `{hash}` is first 8 characters of script content SHA256

**Example:**
```bash
fdtd-compute my_sim.py --output custom_results.h5
```

**Notes:**
- Creates parent directories if they don't exist
- Overwrites existing files without warning
- Use `.h5` or `.hdf5` extension

### --threads, -t N

Set number of OpenMP threads for parallel computation.

**Default**: All available CPU cores (from `os.cpu_count()`)

**Example:**
```bash
# Use 8 threads
fdtd-compute my_sim.py --threads 8

# Use half of available cores
fdtd-compute my_sim.py --threads $(nproc --all / 2)
```

**Notes:**
- Overrides `OMP_NUM_THREADS` environment variable
- More threads ≠ always faster (diminishing returns beyond ~8-16 threads)
- Reduce threads if running multiple simulations concurrently

**Performance guide:**
| Grid Size | Optimal Threads | Speedup vs 1 Thread |
|-----------|-----------------|---------------------|
| 50³       | 4               | 3.2x                |
| 100³      | 8               | 5.8x                |
| 200³      | 16              | 10.2x               |

### --backend {auto,native,python}

Force specific computation backend.

**Default**: `auto` (uses native if available, otherwise Python)

**Example:**
```bash
# Force native backend (fails if not installed)
fdtd-compute my_sim.py --backend native

# Force pure Python (slow, for debugging)
fdtd-compute my_sim.py --backend python
```

**Backend comparison:**
| Backend | Speed   | Requirements           | Use case              |
|---------|---------|------------------------|-----------------------|
| native  | Fast    | C++ compiler, CMake    | Production            |
| python  | Slow    | None                   | Debugging, small sims |
| auto    | Fastest | None (falls back)      | Default               |

### --verbose, -v

Enable verbose logging (DEBUG level).

**Default**: INFO level

**Example:**
```bash
fdtd-compute my_sim.py --verbose
```

**Output includes:**
- Detailed timestep logging
- Memory allocation details
- PML boundary setup
- Material initialization
- Grid creation steps

**Use for:**
- Debugging unexpected behavior
- Understanding performance bottlenecks
- Verifying configuration

### --dry-run

Validate script without running simulation.

**Example:**
```bash
fdtd-compute my_sim.py --dry-run
```

**Performs:**
- ✓ Script parsing and execution
- ✓ Grid and scene initialization
- ✓ Memory and timestep estimation
- ✗ Actual simulation (skipped)

**Output:**
```
FDTD Simulation: my_sim.py
────────────────────────────────────────────────────────
Grid:       100 × 100 × 100 (1.0M cells)
Resolution: 1.0 mm
Timestep:   1.65e-9 s
Duration:   1000 steps (1.65 μs)
Backend:    native (16 threads)
Output:     results.h5
────────────────────────────────────────────────────────

Dry run - simulation not executed
```

**Use for:**
- Validating scripts before submission to cluster
- Estimating resource requirements
- Testing script changes quickly

### --version

Show version and exit.

**Example:**
```bash
fdtd-compute --version
```

### --help, -h

Show help message and exit.

**Example:**
```bash
fdtd-compute --help
```

## Progress Display

The CLI provides real-time feedback during simulation execution.

### Progress Bar

```
[████████████░░░░░░░░] 60% | Step 600/1000
```

**Components:**
- **Bar**: Visual progress indicator (20 characters)
- **Percentage**: Completion percentage
- **Steps**: Current step / Total steps

### Statistics Line

```
Elapsed: 1m 23s | ETA: 0m 55s
Speed: 7.2 Mcells/s | Memory: 1.8 GB / 2.1 GB
```

**Metrics:**
- **Elapsed**: Time since simulation started
- **ETA**: Estimated time to completion (based on current speed)
- **Speed**: Cell updates per second (millions)
- **Memory**: Current RAM usage / Peak RAM usage

**Update rate**: 10 times per second (100ms interval)

## Script Requirements

The CLI tool expects simulation scripts to define a `solver` variable.

### Minimal Script

```python
from strata_fdtd import FDTDSolver, GaussianPulse

solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
solver.add_source(GaussianPulse(position=(0.025, 0.05, 0.05), frequency=40e3))
```

**Required:**
- `solver` variable must be defined at module level
- `solver` must be an instance of `FDTDSolver`

**Optional:**
- Add probes, materials, custom configuration
- Import additional modules (restricted - see Security)

### Script Execution Flow

1. **Parse script**: Load file and compile Python code
2. **Execute script**: Run in controlled namespace
3. **Extract solver**: Retrieve `solver` variable from namespace
4. **Configure solver**: Apply CLI options (threads, backend)
5. **Run simulation**: Call `solver.run(output_file=...)`
6. **Report results**: Print summary and output file path

## Security

Scripts are executed in a **restricted namespace** for safety.

### Allowed Imports

- `strata_fdtd` (all submodules)
- `numpy`
- `scipy`

### Blocked Imports

All other modules are blocked, including:
- `os`, `sys` (system access)
- `subprocess` (command execution)
- `socket`, `requests` (network access)
- `shutil`, `pathlib` (filesystem modification beyond output)

**Example:**
```python
import os  # ❌ ImportError: Import of 'os' is not allowed

from strata_fdtd import *  # ✓ Allowed
import numpy as np  # ✓ Allowed
```

### Why Restrict Imports?

- **Safety**: Prevent malicious code execution
- **Reproducibility**: Ensure simulations don't have hidden dependencies
- **Cluster compatibility**: Scripts can run in sandboxed environments

**Workaround for advanced use cases:**
Use Python directly instead of the CLI tool:
```bash
python my_custom_script.py  # No restrictions
```

## Output File Format

Results are saved in **HDF5** format (`.h5` extension).

### File Structure

```
results.h5
├── /metadata
│   ├── grid_shape: [100, 100, 100]
│   ├── resolution: 0.001
│   ├── timestep: 1.65e-9
│   └── ...
├── /snapshots
│   ├── /0: 3D array of pressure (100×100×100)
│   ├── /100: 3D array of pressure
│   ├── /200: 3D array of pressure
│   └── ...
└── /probes
    ├── /probe_name
    │   ├── pressure: 1D array of pressure values
    │   ├── time: 1D array of time values
    │   └── position: [x, y, z]
    └── ...
```

### Reading Results in Python

```python
import h5py
import numpy as np

with h5py.File("results.h5", "r") as f:
    # Read metadata
    grid_shape = f["metadata"]["grid_shape"][:]
    resolution = f["metadata"]["resolution"][()]

    # Read snapshot at step 500
    pressure_field = f["snapshots/500"][:]

    # Read probe data
    probe_pressure = f["probes/downstream/pressure"][:]
    probe_time = f["probes/downstream/time"][:]
```

See [Viewer Mode Guide](viewer-guide.md) for visualization tools.

## Examples

### Basic Simulation

```bash
fdtd-compute examples/basic_pulse.py
```

### Custom Output Path

```bash
fdtd-compute my_sim.py --output results/run_001.h5
```

### Limit Threads (for laptop or shared server)

```bash
fdtd-compute large_sim.py --threads 4
```

### Force Pure Python Backend (debugging)

```bash
fdtd-compute my_sim.py --backend python --verbose
```

### Batch Processing

```bash
# Run multiple simulations sequentially
for script in simulations/*.py; do
    fdtd-compute "$script" --threads 8
done
```

### Parallel Batch (GNU Parallel)

```bash
# Run multiple simulations in parallel (2 at a time, 4 threads each)
find simulations/ -name "*.py" | \
    parallel -j 2 fdtd-compute {} --threads 4
```

### Resource Estimation

```bash
# Estimate memory and runtime without running
fdtd-compute large_sim.py --dry-run
```

## Error Handling

### Script Not Found

```bash
$ fdtd-compute missing.py
Error: Script file not found: missing.py
```

**Solution**: Check file path and spelling.

### Syntax Error in Script

```bash
$ fdtd-compute bad_syntax.py
Error: Syntax error in script (line 12)
  scene.add_source(
                  ^
SyntaxError: unexpected EOF while parsing
```

**Solution**: Fix Python syntax errors.

### Missing Solver Variable

```bash
$ fdtd-compute no_solver.py
Error: Script must define a 'solver' variable
Found variables: grid, source
```

**Solution**: Create a `solver` variable using `solver = FDTDSolver(shape=(...), resolution=...)`.

### Out of Memory

```bash
$ fdtd-compute huge_sim.py
Error: Unable to allocate 64 GB for pressure field
MemoryError: Cannot allocate memory
```

**Solutions:**
- Reduce grid size
- Increase cell resolution (coarser grid)
- Use nonuniform grids
- Run on a machine with more RAM

### Keyboard Interrupt

```bash
$ fdtd-compute long_sim.py
[████░░░░░░░░░░░░░░░░] 20% | Step 200/1000
^C
Interrupted by user
```

**Behavior:**
- Simulation stops immediately
- Partial results are NOT saved
- Exit code: 1 (failure)

**To resume**: Re-run the script (no checkpointing yet).

### HDF5 Write Error

```bash
$ fdtd-compute my_sim.py
Error: Failed to write HDF5 file
OSError: Unable to create file (errno = 28, error message = 'No space left on device')
```

**Solutions:**
- Free disk space
- Use `--output` to specify different drive
- Reduce snapshot frequency in script

## Performance Tips

### 1. Use Native Backend

```bash
# Install native backend if not already
pip install -e ".[native]"

# Verify
python -c "from strata_fdtd.fdtd import has_native_kernels; print(has_native_kernels())"
```

**Speedup**: 10-20x over pure Python

### 2. Optimize Thread Count

```bash
# Find optimal thread count (usually 8-16 for large grids)
for t in 1 2 4 8 16; do
    echo "Testing $t threads"
    time fdtd-compute my_sim.py --threads $t
done
```

**Diminishing returns**: Speedup plateaus beyond 16 threads for typical grids.

### 3. Profile Before Scaling

```bash
# Test with small grid first
fdtd-compute small_test.py --dry-run

# Estimate scaling
# Memory scales as O(N³)
# Runtime scales as O(N³ × timesteps)
```

### 4. Batch Efficiently

```bash
# Run 4 sims in parallel, each using 2 threads (on 8-core machine)
parallel -j 4 fdtd-compute {} --threads 2 ::: sim_*.py
```

**Rule**: `(parallel jobs) × (threads per job) ≤ CPU cores`

## Environment Variables

### OMP_NUM_THREADS

Sets default thread count for native backend.

```bash
export OMP_NUM_THREADS=8
fdtd-compute my_sim.py  # Uses 8 threads
```

**Note**: `--threads` flag overrides this variable.

### PYTHONPATH

Allows imports from custom locations (advanced).

```bash
export PYTHONPATH=/path/to/custom/modules:$PYTHONPATH
fdtd-compute my_sim.py
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | General error (script error, out of memory, etc.) |
| 2    | Invalid command-line arguments |
| 130  | Interrupted by user (Ctrl+C) |

**Usage in scripts:**
```bash
fdtd-compute my_sim.py
if [ $? -eq 0 ]; then
    echo "Simulation succeeded"
    # Upload results, run analysis, etc.
else
    echo "Simulation failed"
    exit 1
fi
```

## Further Reading

- **[Getting Started Guide](getting-started.md)** - Your first simulation
- **[Builder Mode Guide](builder-guide.md)** - Visual script editor
- **[Viewer Mode Guide](viewer-guide.md)** - Result visualization
- **[API Reference](api-reference.md)** - Python API documentation
- **[Troubleshooting](troubleshooting.md)** - Common problems

---

**CLI Tool Status**: This documentation describes the planned functionality of the `fdtd-compute` CLI tool as specified in project requirements. The tool may still be under development. For current implementation status, check the project repository or use `fdtd-compute --help`.
