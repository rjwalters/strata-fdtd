#!/usr/bin/env python3
"""
FDTD Solver Profiling Script

Profiles the FDTD acoustic solver to identify hotspots for pybind11 optimization.
Provides detailed timing breakdown and identifies parallelism/SIMD opportunities.

Usage:
    python3 scripts/profile_fdtd.py                    # Default 100³ grid
    python3 scripts/profile_fdtd.py --size 150        # Custom grid size
    python3 scripts/profile_fdtd.py --steps 200       # More timesteps
    python3 scripts/profile_fdtd.py --cprofile        # Full cProfile output
    python3 scripts/profile_fdtd.py --line-profile    # Line-by-line profiling
"""

import argparse
import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strata_fdtd.fdtd import FDTDSolver, GaussianPulse
from strata_fdtd.boundaries import PML


@dataclass
class TimingResult:
    """Stores timing data for a single operation."""
    name: str
    total_time: float
    calls: int

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0

    @property
    def percent(self) -> float:
        return 0.0  # Set externally


class ManualProfiler:
    """Manual profiler for fine-grained timing of FDTD operations."""

    def __init__(self):
        self.timings: dict[str, TimingResult] = {}
        self._start_time = 0.0
        self._current_op = ""

    @contextmanager
    def time_operation(self, name: str):
        """Context manager to time a specific operation."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        if name not in self.timings:
            self.timings[name] = TimingResult(name, 0.0, 0)
        self.timings[name].total_time += elapsed
        self.timings[name].calls += 1

    def report(self) -> str:
        """Generate a timing report."""
        if not self.timings:
            return "No timing data collected"

        total = sum(t.total_time for t in self.timings.values())

        lines = [
            "=" * 70,
            "FDTD OPERATION TIMING BREAKDOWN",
            "=" * 70,
            f"{'Operation':<35} {'Total (ms)':>10} {'%':>8} {'Calls':>8} {'Avg (μs)':>10}",
            "-" * 70,
        ]

        # Sort by total time descending
        sorted_timings = sorted(
            self.timings.values(),
            key=lambda t: t.total_time,
            reverse=True
        )

        for t in sorted_timings:
            pct = (t.total_time / total * 100) if total > 0 else 0
            lines.append(
                f"{t.name:<35} {t.total_time*1000:>10.2f} {pct:>7.1f}% "
                f"{t.calls:>8} {t.avg_time*1e6:>10.2f}"
            )

        lines.extend([
            "-" * 70,
            f"{'TOTAL':<35} {total*1000:>10.2f}",
            "=" * 70,
        ])

        return "\n".join(lines)


def profile_step_components(solver: FDTDSolver, n_steps: int = 100) -> ManualProfiler:
    """Profile individual components of the FDTD step function."""
    profiler = ManualProfiler()

    # Get references to solver fields
    p = solver.p
    vx, vy, vz = solver.vx, solver.vy, solver.vz
    coeff_v = solver._coeff_v
    coeff_p = solver._coeff_p
    geometry = solver.geometry
    nx, ny, nz = solver.shape

    for _ in range(n_steps):
        # ===== VELOCITY UPDATES =====
        # These are the main hotspots - large 3D array operations

        with profiler.time_operation("velocity_x_update"):
            vx[:-1, :, :] += coeff_v * (p[1:, :, :] - p[:-1, :, :])

        with profiler.time_operation("velocity_y_update"):
            vy[:, :-1, :] += coeff_v * (p[:, 1:, :] - p[:, :-1, :])

        with profiler.time_operation("velocity_z_update"):
            vz[:, :, :-1] += coeff_v * (p[:, :, 1:] - p[:, :, :-1])

        # ===== RIGID BOUNDARY APPLICATION =====
        with profiler.time_operation("rigid_boundary_mask_vx"):
            solid_vx = ~geometry[:-1, :, :] | ~geometry[1:, :, :]

        with profiler.time_operation("rigid_boundary_apply_vx"):
            vx[:-1, :, :][solid_vx] = 0.0

        with profiler.time_operation("rigid_boundary_mask_vy"):
            solid_vy = ~geometry[:, :-1, :] | ~geometry[:, 1:, :]

        with profiler.time_operation("rigid_boundary_apply_vy"):
            vy[:, :-1, :][solid_vy] = 0.0

        with profiler.time_operation("rigid_boundary_mask_vz"):
            solid_vz = ~geometry[:, :, :-1] | ~geometry[:, :, 1:]

        with profiler.time_operation("rigid_boundary_apply_vz"):
            vz[:, :, :-1][solid_vz] = 0.0

        # ===== DIVERGENCE COMPUTATION =====
        with profiler.time_operation("divergence_alloc"):
            div = np.zeros(solver.shape, dtype=np.float32)

        with profiler.time_operation("divergence_x"):
            div[0, :, :] = vx[0, :, :]
            div[1:, :, :] = vx[1:nx, :, :] - vx[:nx-1, :, :]

        with profiler.time_operation("divergence_y"):
            div[:, 0, :] += vy[:, 0, :]
            div[:, 1:, :] += vy[:, 1:ny, :] - vy[:, :ny-1, :]

        with profiler.time_operation("divergence_z"):
            div[:, :, 0] += vz[:, :, 0]
            div[:, :, 1:] += vz[:, :, 1:nz] - vz[:, :, :nz-1]

        # ===== PRESSURE UPDATE =====
        with profiler.time_operation("pressure_update"):
            p += coeff_p * div

        with profiler.time_operation("pressure_solid_zeroing"):
            p[~geometry] = 0.0

        # Update solver state
        solver._step_count += 1
        solver._time += solver.dt

    return profiler


def profile_with_pml(solver: FDTDSolver, pml: PML, n_steps: int = 100) -> ManualProfiler:
    """Profile FDTD with PML boundary conditions."""
    profiler = ManualProfiler()

    # Get PML damping arrays
    sigma_x = pml._sigma_x
    sigma_y = pml._sigma_y
    sigma_z = pml._sigma_z
    dt = solver.dt

    p = solver.p
    vx, vy, vz = solver.vx, solver.vy, solver.vz

    for _ in range(n_steps):
        # Standard step (simplified)
        with profiler.time_operation("velocity_updates"):
            vx[:-1, :, :] += solver._coeff_v * (p[1:, :, :] - p[:-1, :, :])
            vy[:, :-1, :] += solver._coeff_v * (p[:, 1:, :] - p[:, :-1, :])
            vz[:, :, :-1] += solver._coeff_v * (p[:, :, 1:] - p[:, :, :-1])

        # ===== PML VELOCITY DAMPING =====
        if sigma_x is not None:
            with profiler.time_operation("pml_exp_compute_x"):
                decay_x = np.exp(-sigma_x * dt)
            with profiler.time_operation("pml_velocity_damp_x"):
                vx *= decay_x[:, np.newaxis, np.newaxis]

        if sigma_y is not None:
            with profiler.time_operation("pml_exp_compute_y"):
                decay_y = np.exp(-sigma_y * dt)
            with profiler.time_operation("pml_velocity_damp_y"):
                vy *= decay_y[np.newaxis, :, np.newaxis]

        if sigma_z is not None:
            with profiler.time_operation("pml_exp_compute_z"):
                decay_z = np.exp(-sigma_z * dt)
            with profiler.time_operation("pml_velocity_damp_z"):
                vz *= decay_z[np.newaxis, np.newaxis, :]

        # Divergence and pressure (simplified)
        with profiler.time_operation("divergence_and_pressure"):
            nx, ny, nz = solver.shape
            div = np.zeros(solver.shape, dtype=np.float32)
            div[0, :, :] = vx[0, :, :]
            div[1:, :, :] = vx[1:nx, :, :] - vx[:nx-1, :, :]
            div[:, 0, :] += vy[:, 0, :]
            div[:, 1:, :] += vy[:, 1:ny, :] - vy[:, :ny-1, :]
            div[:, :, 0] += vz[:, :, 0]
            div[:, :, 1:] += vz[:, :, 1:nz] - vz[:, :, :nz-1]
            p += solver._coeff_p * div

        # ===== PML PRESSURE DAMPING =====
        if sigma_x is not None:
            with profiler.time_operation("pml_pressure_damp_x"):
                p *= decay_x[:, np.newaxis, np.newaxis]

        if sigma_y is not None:
            with profiler.time_operation("pml_pressure_damp_y"):
                p *= decay_y[np.newaxis, :, np.newaxis]

        if sigma_z is not None:
            with profiler.time_operation("pml_pressure_damp_z"):
                p *= decay_z[np.newaxis, np.newaxis, :]

        solver._step_count += 1
        solver._time += solver.dt

    return profiler


def run_cprofile(solver: FDTDSolver, n_steps: int = 100) -> str:
    """Run standard cProfile on solver.step()."""
    profiler = cProfile.Profile()

    profiler.enable()
    for _ in range(n_steps):
        solver.step()
    profiler.disable()

    # Generate report
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue()


def compute_theoretical_limits(shape: tuple[int, int, int], n_steps: int) -> dict:
    """Compute theoretical performance limits based on hardware."""
    nx, ny, nz = shape
    total_cells = nx * ny * nz

    # Memory bandwidth analysis
    # Each step reads/writes: p, vx, vy, vz (4 arrays * 4 bytes * total_cells)
    # Plus temporary divergence array
    bytes_per_step = 4 * 4 * total_cells + 4 * total_cells  # 20 bytes per cell

    # FLOPs analysis
    # Velocity updates: 3 * (2 mults + 1 add) = 9 FLOPs per cell
    # Divergence: ~6 ops per cell
    # Pressure update: 2 ops per cell
    # Approximate: ~20 FLOPs per cell per step
    flops_per_step = 20 * total_cells

    return {
        "total_cells": total_cells,
        "total_cell_updates": total_cells * n_steps,
        "bytes_per_step": bytes_per_step,
        "total_bytes": bytes_per_step * n_steps,
        "flops_per_step": flops_per_step,
        "total_flops": flops_per_step * n_steps,
        "memory_mb": (4 * 4 + 1) * total_cells / 1e6,  # 4 float32 + 1 bool arrays
    }


def benchmark_backends(shape: tuple[int, int, int], n_steps: int = 100) -> dict:
    """Benchmark Python vs C++ backends and report speedup.

    Args:
        shape: Grid dimensions (nx, ny, nz)
        n_steps: Number of timesteps to run

    Returns:
        Dict with timing results and speedup metrics
    """
    from strata_fdtd.fdtd import has_native_kernels, get_native_info

    results = {
        "shape": shape,
        "n_steps": n_steps,
        "total_cells": np.prod(shape),
        "python_time": None,
        "native_time": None,
        "speedup": None,
        "python_cells_per_sec": None,
        "native_cells_per_sec": None,
    }

    # Always run Python benchmark
    print(f"\n  Benchmarking Python backend...")
    solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
    solver_py.add_source(GaussianPulse(
        position=(shape[0]//4, shape[1]//2, shape[2]//2),
        frequency=1000
    ))

    # Warmup
    for _ in range(5):
        solver_py.step()
    solver_py.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        solver_py.step()
    python_time = time.perf_counter() - start
    results["python_time"] = python_time
    results["python_cells_per_sec"] = (np.prod(shape) * n_steps) / python_time

    # Store Python final state for correctness check
    python_final_p = solver_py.p.copy()

    print(f"    Time: {python_time*1000:.1f} ms ({n_steps} steps)")
    print(f"    Cells/sec: {results['python_cells_per_sec']/1e6:.2f} M")

    # Run native benchmark if available
    if has_native_kernels():
        native_info = get_native_info()
        print(f"\n  Benchmarking Native backend...")
        print(f"    Version: {native_info['version']}")
        print(f"    OpenMP: {native_info['has_openmp']} ({native_info['num_threads']} threads)")

        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")
        solver_native.add_source(GaussianPulse(
            position=(shape[0]//4, shape[1]//2, shape[2]//2),
            frequency=1000
        ))

        # Warmup
        for _ in range(5):
            solver_native.step()
        solver_native.reset()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_steps):
            solver_native.step()
        native_time = time.perf_counter() - start
        results["native_time"] = native_time
        results["native_cells_per_sec"] = (np.prod(shape) * n_steps) / native_time
        results["speedup"] = python_time / native_time

        print(f"    Time: {native_time*1000:.1f} ms ({n_steps} steps)")
        print(f"    Cells/sec: {results['native_cells_per_sec']/1e6:.2f} M")
        print(f"    Speedup: {results['speedup']:.1f}x")

        # Numerical correctness check
        native_final_p = solver_native.p
        max_diff = np.max(np.abs(python_final_p - native_final_p))
        mean_diff = np.mean(np.abs(python_final_p - native_final_p))
        results["max_diff"] = float(max_diff)
        results["mean_diff"] = float(mean_diff)

        # Relative tolerance for float32 (around 1e-6 relative error is typical)
        p_range = np.max(np.abs(python_final_p))
        if p_range > 0:
            rel_error = max_diff / p_range
            results["rel_error"] = float(rel_error)
            correct = rel_error < 1e-5  # Allow 0.001% relative error
        else:
            results["rel_error"] = 0.0
            correct = max_diff < 1e-7

        results["correct"] = correct
        if correct:
            print(f"    Numerical check: PASS (max diff: {max_diff:.2e}, rel: {results.get('rel_error', 0):.2e})")
        else:
            print(f"    Numerical check: FAIL (max diff: {max_diff:.2e}, rel: {results.get('rel_error', 0):.2e})")
    else:
        print(f"\n  Native backend not available - skipping native benchmark")
        print("  Install with: pip install -e '.[native]'")

    return results


def run_benchmark_suite():
    """Run benchmark suite across multiple grid sizes."""
    print("\n" + "="*70)
    print("FDTD BACKEND BENCHMARK SUITE")
    print("="*70)

    from strata_fdtd.fdtd import has_native_kernels, get_native_info

    if has_native_kernels():
        info = get_native_info()
        print(f"\nNative backend: {info['version']}")
        print(f"OpenMP: {info['has_openmp']} ({info['num_threads']} threads)")
    else:
        print("\nNative backend: Not available")

    # Test sizes from issue specification
    test_cases = [
        (50, 50, 50, 200),    # Small grid, more steps
        (100, 100, 100, 100), # Target: 10x speedup
        (150, 150, 150, 50),  # Target: 14x speedup
        (200, 200, 200, 25),  # Target: 18x speedup (if memory allows)
    ]

    all_results = []
    for nx, ny, nz, n_steps in test_cases:
        shape = (nx, ny, nz)
        mem_mb = (4 * 4 + 1) * nx * ny * nz / 1e6
        if mem_mb > 500:  # Skip if > 500MB memory required
            print(f"\n  Skipping {nx}³ ({mem_mb:.0f} MB) - too large")
            continue

        print(f"\n{'='*70}")
        print(f"Grid: {nx}³ ({nx*ny*nz/1e6:.1f}M cells, {mem_mb:.0f} MB)")
        print(f"Steps: {n_steps}")
        print("="*70)

        try:
            results = benchmark_backends(shape, n_steps)
            all_results.append(results)
        except MemoryError:
            print(f"  MemoryError - skipping")
            continue

    # Summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Grid':<12} {'Python':<15} {'Native':<15} {'Speedup':<10} {'Correct':<8}")
    print("-"*70)

    for r in all_results:
        nx = r['shape'][0]
        py_rate = f"{r['python_cells_per_sec']/1e6:.1f}M/s" if r['python_cells_per_sec'] else "N/A"
        native_rate = f"{r['native_cells_per_sec']/1e6:.1f}M/s" if r['native_cells_per_sec'] else "N/A"
        speedup = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        correct = "PASS" if r.get('correct', None) else ("FAIL" if r.get('correct') is False else "N/A")
        print(f"{nx}³{'':<8} {py_rate:<15} {native_rate:<15} {speedup:<10} {correct:<8}")

    print("="*70)

    # Performance targets from issue
    print("\nPerformance Targets (from issue #46):")
    print("  100³: 10x speedup")
    print("  150³: 14x speedup")
    print("  200³: 18x speedup")

    return all_results


def print_parallelism_analysis():
    """Print analysis of parallelism and SIMD opportunities."""
    analysis = """
================================================================================
PARALLELISM AND SIMD OPTIMIZATION ANALYSIS
================================================================================

With 28 cores available, here are the key optimization opportunities:

1. VELOCITY UPDATES (highest priority - ~40-50% of runtime)
   -------------------------------------------------------------------------
   Current:  vx[:-1,:,:] += coeff * (p[1:,:,:] - p[:-1,:,:])

   SIMD Opportunity: ★★★★★
   - Each element is independent - perfect for AVX2/AVX-512
   - 8 floats per AVX2 register, 16 per AVX-512
   - Potential speedup: 4-8x from SIMD alone

   Parallelism Opportunity: ★★★★★
   - Embarrassingly parallel across all cells
   - Split grid along slowest-varying dimension (x)
   - Each thread handles nx/28 slices
   - No synchronization needed during update

   C++ Implementation Strategy:
   ```cpp
   #pragma omp parallel for simd
   for (int i = 0; i < nx-1; i++) {
       for (int j = 0; j < ny; j++) {
           for (int k = 0; k < nz; k++) {
               vx[idx(i,j,k)] += coeff_v * (p[idx(i+1,j,k)] - p[idx(i,j,k)]);
           }
       }
   }
   ```

2. DIVERGENCE COMPUTATION (~20-25% of runtime)
   -------------------------------------------------------------------------
   Current: Allocates new array each step, computes x/y/z divergence separately

   SIMD Opportunity: ★★★★☆
   - Can fuse x, y, z contributions in single pass
   - Memory-bound operation - cache optimization critical

   Parallelism Opportunity: ★★★★★
   - Independent per-cell computation
   - Fuse with pressure update to reduce memory traffic

   C++ Optimization:
   - Pre-allocate divergence buffer (reuse across steps)
   - Fuse divergence + pressure update in single kernel
   - Use cache blocking for better memory locality

   ```cpp
   #pragma omp parallel for collapse(2)
   for (int i = 0; i < nx; i++) {
       for (int j = 0; j < ny; j++) {
           // Vectorizes well along k
           for (int k = 0; k < nz; k++) {
               float div = compute_div(vx, vy, vz, i, j, k);
               p[idx(i,j,k)] += coeff_p * div;
           }
       }
   }
   ```

3. RIGID BOUNDARY APPLICATION (~10-15% of runtime)
   -------------------------------------------------------------------------
   Current: Boolean mask creation + masked assignment

   SIMD Opportunity: ★★★☆☆
   - Masking operations have some SIMD support
   - But branch-heavy code limits vectorization

   Parallelism Opportunity: ★★★★☆
   - Can parallelize mask creation and application
   - Consider pre-computing boundary cell lists

   C++ Optimization Strategy:
   - Pre-compute lists of boundary cells at initialization
   - Only iterate over boundary cells (sparse operation)
   - Avoids creating full boolean mask each step

   ```cpp
   // Pre-computed at init time
   std::vector<int> boundary_vx_cells;

   // At runtime - much faster
   #pragma omp parallel for
   for (int idx : boundary_vx_cells) {
       vx[idx] = 0.0f;
   }
   ```

4. PML DAMPING (~10-15% of runtime when enabled)
   -------------------------------------------------------------------------
   Current: exp() computed each step, broadcasting 1D to 3D

   SIMD Opportunity: ★★★★★
   - Element-wise multiply is ideal for SIMD
   - Can use fast_exp approximations

   Parallelism Opportunity: ★★★★★
   - Fully independent operations

   C++ Optimization:
   - Pre-compute decay factors (exp(-sigma*dt)) at initialization
   - Store as 1D arrays, apply with strided access
   - Consider SIMD exp approximation (if recomputing is needed)

   ```cpp
   // Pre-computed: decay_x[nx], decay_y[ny], decay_z[nz]
   #pragma omp parallel for simd
   for (int i = 0; i < nx; i++) {
       float dx = decay_x[i];
       for (int j = 0; j < ny; j++) {
           for (int k = 0; k < nz; k++) {
               vx[idx(i,j,k)] *= dx;
           }
       }
   }
   ```

5. MEMORY LAYOUT OPTIMIZATION
   -------------------------------------------------------------------------
   Current: Row-major (C) ordering, float32 arrays

   Recommendations:
   - Keep float32 (good cache efficiency, sufficient precision)
   - Consider Array-of-Structs-of-Arrays (AoSoA) for SIMD
   - Align arrays to 64-byte boundaries for AVX-512
   - Use cache blocking for large grids

   Block sizes for optimal cache usage:
   - L1 cache (32KB): ~8 x 8 x 8 blocks
   - L2 cache (256KB): ~16 x 16 x 16 blocks
   - L3 cache (shared): larger blocks for multi-thread coordination

6. THREAD SCHEDULING FOR 28 CORES
   -------------------------------------------------------------------------
   Recommended OpenMP settings:
   - OMP_NUM_THREADS=28
   - OMP_SCHEDULE=static (for regular grid operations)
   - OMP_PROC_BIND=close (keep threads on same socket when possible)

   Grid decomposition strategy:
   - For 100³ grid: 4 slices of 25 cells each per thread pair
   - For 200³ grid: 7-8 slices of ~28 cells each per thread
   - Dynamic scheduling for irregular operations (masked updates)

================================================================================
RECOMMENDED PYBIND11 PORTING ORDER
================================================================================

Priority 1 (highest impact):
  □ fdtd_step_kernel()     - Fused velocity + divergence + pressure update
                             Expected speedup: 10-20x

Priority 2 (significant impact):
  □ apply_rigid_boundaries() - Pre-computed boundary cell iteration
                               Expected speedup: 5-10x
  □ apply_pml_damping()      - Vectorized decay application
                               Expected speedup: 5-10x

Priority 3 (cleanup/completeness):
  □ compute_energy()         - Reduction operation
  □ inject_sources()         - Minor, but can fuse with step
  □ record_probes()          - Negligible

================================================================================
"""
    print(analysis)


def main():
    parser = argparse.ArgumentParser(description="Profile FDTD solver")
    parser.add_argument("--size", type=int, default=100, help="Grid size (NxNxN)")
    parser.add_argument("--steps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--cprofile", action="store_true", help="Run cProfile")
    parser.add_argument("--line-profile", action="store_true", help="Line profiling (requires line_profiler)")
    parser.add_argument("--with-pml", action="store_true", help="Include PML boundaries")
    parser.add_argument("--with-geometry", action="store_true", help="Include geometry mask")
    parser.add_argument("--analysis", action="store_true", help="Print parallelism analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run Python vs Native benchmark suite")
    args = parser.parse_args()

    if args.analysis:
        print_parallelism_analysis()
        return

    if args.benchmark:
        run_benchmark_suite()
        return

    shape = (args.size, args.size, args.size)
    print(f"\n{'='*70}")
    print(f"FDTD PROFILING - {args.size}³ grid ({args.size**3:,} cells)")
    print(f"{'='*70}")

    # Print theoretical limits
    limits = compute_theoretical_limits(shape, args.steps)
    print(f"\nTheoretical analysis:")
    print(f"  Memory for fields: {limits['memory_mb']:.1f} MB")
    print(f"  Bytes transferred per step: {limits['bytes_per_step']/1e6:.1f} MB")
    print(f"  Total cell updates: {limits['total_cell_updates']/1e6:.1f} M")
    print(f"  Total FLOPs: {limits['total_flops']/1e9:.2f} GFLOPs")

    # Create solver
    solver = FDTDSolver(shape=shape, resolution=1e-3)

    # Add geometry if requested
    if args.with_geometry:
        # Create a simple geometry with some solid regions
        geometry = np.ones(shape, dtype=bool)
        # Add a solid block in the center
        cx, cy, cz = args.size // 2, args.size // 2, args.size // 2
        geometry[cx-10:cx+10, cy-10:cy+10, cz-10:cz+10] = False
        solver.set_geometry(geometry)
        print(f"  Geometry: {np.sum(~geometry):,} solid cells ({100*np.sum(~geometry)/np.prod(shape):.1f}%)")

    # Add PML if requested
    pml = None
    if args.with_pml:
        pml = PML(depth=10, axis='all')
        solver.add_boundary(pml)
        print(f"  PML: 10-cell depth on all boundaries")

    # Add a source for realism
    solver.add_source(GaussianPulse(
        position=(args.size//4, args.size//2, args.size//2),
        frequency=1000
    ))

    print(f"\nRunning {args.steps} timesteps...")

    # Warmup
    for _ in range(10):
        solver.step()
    solver.reset()

    # Profile
    if args.cprofile:
        print("\n" + "="*70)
        print("cProfile Results")
        print("="*70)
        print(run_cprofile(solver, args.steps))

    # Manual component profiling
    solver.reset()

    if args.with_pml and pml is not None:
        print("\nProfiling with PML...")
        profiler = profile_with_pml(solver, pml, args.steps)
    else:
        print("\nProfiling step components...")
        profiler = profile_step_components(solver, args.steps)

    print("\n" + profiler.report())

    # Overall throughput
    total_time = sum(t.total_time for t in profiler.timings.values())
    cells_per_sec = (np.prod(shape) * args.steps) / total_time
    steps_per_sec = args.steps / total_time

    print(f"\nOverall Performance:")
    print(f"  Steps/sec: {steps_per_sec:.1f}")
    print(f"  Cell updates/sec: {cells_per_sec/1e6:.2f} M")
    print(f"  Effective bandwidth: {limits['total_bytes']/total_time/1e9:.2f} GB/s")
    print(f"  Effective GFLOP/s: {limits['total_flops']/total_time/1e9:.2f}")

    # Print optimization recommendations
    print("\n" + "="*70)
    print("TOP OPTIMIZATION TARGETS")
    print("="*70)
    sorted_timings = sorted(profiler.timings.values(), key=lambda t: t.total_time, reverse=True)
    for i, t in enumerate(sorted_timings[:5], 1):
        pct = t.total_time / total_time * 100
        print(f"  {i}. {t.name}: {pct:.1f}% of runtime")

    print("\nRun with --analysis for detailed parallelism/SIMD recommendations")


if __name__ == "__main__":
    main()
