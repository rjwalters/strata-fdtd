## fdtd-compute CLI Tool

Command-line tool for executing FDTD simulation scripts with progress tracking and HDF5 output.

### Installation

The `fdtd-compute` command is installed automatically when you install the `ml-audio-codecs` package:

```bash
pip install -e .
```

### Usage

Create a simulation script that defines a `solver` variable:

```python
# my_simulation.py
from strata_fdtd import FDTDSolver, GaussianPulse

# Create solver
solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)

# Add sources
solver.add_source(GaussianPulse(position=(0.05, 0.05, 0.05), frequency=40e3))

# Add probes
solver.add_probe("center", (0.05, 0.05, 0.05))

# Optional: specify duration (otherwise runs for 1000 steps)
duration = 1000 * solver.dt
```

Run the simulation:

```bash
fdtd-compute my_simulation.py
```

### Options

```
--output PATH, -o PATH          Custom output file path (default: results_{hash}.h5)
--threads N, -t N               Number of threads for native backend
--backend {auto,native,python}  Force specific backend
--snapshot-interval N           Save pressure field every N steps
--verbose, -v                   Show debug information
--dry-run                       Validate script without running
--version                       Show version
--help                          Show help message
```

### Examples

**Basic simulation:**
```bash
fdtd-compute my_simulation.py
```

**Custom output path:**
```bash
fdtd-compute my_simulation.py --output my_results.h5
```

**With field snapshots:**
```bash
fdtd-compute my_simulation.py --snapshot-interval 10
```

**Force native backend with 8 threads:**
```bash
fdtd-compute my_simulation.py --backend native --threads 8
```

**Dry run to validate script:**
```bash
fdtd-compute my_simulation.py --dry-run
```

### Output Format

Results are saved in HDF5 format with this structure:

```
results_{hash}.h5
├─ /metadata          - Script hash, creation time, solver version
├─ /grid              - Grid shape, resolution, extent
├─ /simulation        - Timestep, number of steps, CFL number
├─ /sources           - Source definitions
├─ /probes            - Probe time series and positions
├─ /fields/pressure   - Pressure field snapshots (if enabled)
└─ /materials         - Geometry and material definitions
```

### Performance

HDF5 output performance varies significantly depending on whether field snapshots are enabled:

**Probe-only output** (no field snapshots):
- Write overhead: Typically 20-65% for small/medium simulations
- Overhead decreases with longer simulations as amortized cost reduces
- File size: <100 KB (metadata + probe time series)

**With field snapshots**:
- Overhead depends heavily on snapshot frequency
- Snapshot interval = 100: ~0-5% overhead (one snapshot at end)
- Snapshot interval = 10: 25-300% overhead depending on grid size
- Snapshot interval = 1: 100-1000% overhead (saving every step)
- File size scales with grid size and number of snapshots

**Read performance**:
- Single timestep load: <100ms for grids up to 200³
- 100³ grid: ~7ms average
- 200³ grid: ~50ms average
- Random access is efficient due to HDF5 chunking

**Recommendations**:
- For best performance, save snapshots sparingly (interval ≥ 100)
- Use probe data for time-series analysis when possible
- For visualization, save snapshots only at key timepoints
- Consider post-processing long simulations to extract snapshots

**Benchmarking**:
```bash
# Run full benchmark suite
python scripts/benchmark_hdf5.py

# Quick benchmark (smaller grids, fewer steps)
python scripts/benchmark_hdf5.py --quick

# JSON output for analysis
python scripts/benchmark_hdf5.py --json
```

### Security

The tool executes simulation scripts in a restricted environment that only allows imports from:
- `strata_fdtd`
- `numpy`
- `scipy`
- `math`
- `pathlib`

Attempts to import other modules (like `os`, `sys`, `subprocess`) will be blocked.

### Progress Display

The tool shows real-time progress with:
- Progress bar with percentage
- Elapsed time and ETA
- Computational throughput (Mcells/s)
- Memory usage (current and peak)

### Error Handling

- **Script not found**: Clear error message with file path
- **Syntax errors**: Python traceback is shown
- **Missing solver**: Helpful message explaining what's required
- **Import restrictions**: Lists allowed modules
- **Ctrl+C**: Graceful cleanup and exit
