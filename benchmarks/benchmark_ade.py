#!/usr/bin/env python3
"""
ADE Material Benchmark Script

Benchmarks native C++ ADE (Auxiliary Differential Equation) kernels against
the pure Python implementation to quantify performance improvements.

This script measures:
- Grid size scaling (50³, 100³, 150³, 200³)
- Material configurations (single material, multiple materials)
- Pole count variations (Debye and Lorentz poles)
- Thread scaling (1, 2, 4, 8 threads)

Usage:
    python3 scripts/benchmark_ade.py                 # Run full benchmark suite
    python3 scripts/benchmark_ade.py --quick         # Quick benchmark (smaller grids)
    python3 scripts/benchmark_ade.py --grid-only     # Only grid scaling tests
    python3 scripts/benchmark_ade.py --threads-only  # Only thread scaling tests
    python3 scripts/benchmark_ade.py --json          # Output results as JSON
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strata_fdtd.fdtd import FDTDSolver, GaussianPulse, has_native_kernels, get_native_info
from strata_fdtd.materials import (
    Pole,
    PoleType,
    SimpleMaterial,
    PorousMaterial,
    HelmholtzResonator,
    FIBERGLASS_48,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    grid_size: tuple[int, int, int]
    n_steps: int
    n_materials: int
    n_debye_poles: int
    n_lorentz_poles: int
    python_time_ms: float
    native_time_ms: float | None
    speedup: float | None
    python_cells_per_sec: float
    native_cells_per_sec: float | None
    correct: bool | None
    max_diff: float | None


def create_test_material(n_debye: int = 2, n_lorentz: int = 0, name: str = "test") -> SimpleMaterial:
    """Create a test material with specified pole count.

    Args:
        n_debye: Number of Debye poles
        n_lorentz: Number of Lorentz poles
        name: Material name

    Returns:
        SimpleMaterial with specified poles
    """
    poles = []

    # Add Debye poles with varying relaxation times
    for i in range(n_debye):
        tau = 1e-6 * (10 ** i)  # 1μs, 10μs, 100μs, ...
        poles.append(Pole(
            pole_type=PoleType.DEBYE,
            delta_chi=0.1 + 0.05 * i,
            tau=tau,
            target="modulus" if i % 2 == 0 else "density",
        ))

    # Add Lorentz poles with varying resonance frequencies
    for i in range(n_lorentz):
        f0 = 100 * (2 ** i)  # 100Hz, 200Hz, 400Hz, ...
        omega_0 = 2 * np.pi * f0
        poles.append(Pole(
            pole_type=PoleType.LORENTZ,
            delta_chi=0.2 + 0.05 * i,
            omega_0=omega_0,
            gamma=omega_0 * 0.1,  # Q = 10
            target="modulus" if i % 2 == 0 else "density",
        ))

    return SimpleMaterial(name=name, _rho=1.2, _c=343.0, _poles=poles)


def setup_solver_with_materials(
    shape: tuple[int, int, int],
    materials: list,
    backend: str = "auto",
) -> FDTDSolver:
    """Create solver and register materials.

    Args:
        shape: Grid dimensions
        materials: List of (material, fraction) tuples
        backend: Backend to use ("python", "native", or "auto")

    Returns:
        Configured FDTDSolver
    """
    solver = FDTDSolver(shape=shape, resolution=1e-3, backend=backend)

    # Add source for realistic simulation
    solver.add_source(GaussianPulse(
        position=(shape[0] // 4, shape[1] // 2, shape[2] // 2),
        frequency=1000,
    ))

    # Register materials and assign regions
    nx, ny, nz = shape
    total_assigned = 0

    for material, fraction in materials:
        mat_id = solver.register_material(material)

        # Assign a fraction of the grid to this material
        start = total_assigned
        end = int(start + fraction * nx)
        end = min(end, nx)

        if start < end:
            mask = np.zeros(shape, dtype=bool)
            mask[start:end, :, :] = True
            solver.set_material_region(mask, material_id=mat_id)
            total_assigned = end

    return solver


def run_benchmark(
    name: str,
    shape: tuple[int, int, int],
    materials: list,
    n_steps: int = 50,
    warmup_steps: int = 5,
) -> BenchmarkResult:
    """Run a single benchmark comparison.

    Args:
        name: Benchmark name
        shape: Grid dimensions
        materials: List of (material, fraction) tuples
        n_steps: Number of timesteps
        warmup_steps: Warmup steps before timing

    Returns:
        BenchmarkResult with timing data
    """
    # Count poles
    n_debye = sum(
        sum(1 for p in m.poles if p.is_debye)
        for m, _ in materials
    )
    n_lorentz = sum(
        sum(1 for p in m.poles if p.is_lorentz)
        for m, _ in materials
    )

    total_cells = np.prod(shape)

    # Run Python benchmark
    solver_py = setup_solver_with_materials(shape, materials, backend="python")

    # Warmup
    for _ in range(warmup_steps):
        solver_py.step()
    solver_py.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        solver_py.step()
    python_time = time.perf_counter() - start
    python_final = solver_py.p.copy()

    python_time_ms = python_time * 1000
    python_cells_per_sec = (total_cells * n_steps) / python_time

    # Run native benchmark if available
    native_time_ms = None
    native_cells_per_sec = None
    speedup = None
    correct = None
    max_diff = None

    if has_native_kernels():
        solver_native = setup_solver_with_materials(shape, materials, backend="native")

        # Warmup
        for _ in range(warmup_steps):
            solver_native.step()
        solver_native.reset()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_steps):
            solver_native.step()
        native_time = time.perf_counter() - start
        native_final = solver_native.p.copy()

        native_time_ms = native_time * 1000
        native_cells_per_sec = (total_cells * n_steps) / native_time
        speedup = python_time / native_time

        # Numerical correctness check
        # Note: Float32 accumulation over many timesteps can have significant drift
        # We use a relaxed tolerance for benchmarking purposes
        max_diff = float(np.max(np.abs(python_final - native_final)))
        p_range = np.max(np.abs(python_final))
        if p_range > 0:
            rel_error = max_diff / p_range
            # Allow 0.1% relative error for float32 across many timesteps
            correct = rel_error < 1e-3
        else:
            correct = max_diff < 1e-4

    return BenchmarkResult(
        name=name,
        grid_size=shape,
        n_steps=n_steps,
        n_materials=len(materials),
        n_debye_poles=n_debye,
        n_lorentz_poles=n_lorentz,
        python_time_ms=python_time_ms,
        native_time_ms=native_time_ms,
        speedup=speedup,
        python_cells_per_sec=python_cells_per_sec,
        native_cells_per_sec=native_cells_per_sec,
        correct=correct,
        max_diff=max_diff,
    )


def benchmark_grid_scaling(quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark different grid sizes.

    Args:
        quick: Use smaller grids and fewer steps

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 70)
    print("GRID SIZE SCALING BENCHMARK")
    print("=" * 70)

    # Test material: porous absorber with 6 Debye poles
    material = FIBERGLASS_48
    materials = [(material, 0.25)]  # 25% of grid

    if quick:
        test_cases = [
            ((30, 30, 30), 30),
            ((50, 50, 50), 20),
            ((75, 75, 75), 15),
        ]
    else:
        test_cases = [
            ((50, 50, 50), 100),
            ((100, 100, 100), 50),
            ((150, 150, 150), 25),
            ((200, 200, 200), 15),
        ]

    results = []
    for shape, n_steps in test_cases:
        mem_mb = (4 * 4 + 1) * np.prod(shape) / 1e6
        if mem_mb > 500:
            print(f"  Skipping {shape[0]}³ ({mem_mb:.0f} MB) - too large")
            continue

        print(f"\n  Grid: {shape[0]}³ ({np.prod(shape)/1e6:.1f}M cells)")
        result = run_benchmark(
            name=f"grid_{shape[0]}",
            shape=shape,
            materials=materials,
            n_steps=n_steps,
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_material_configs(quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark different material configurations.

    Args:
        quick: Use smaller grids

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 70)
    print("MATERIAL CONFIGURATION BENCHMARK")
    print("=" * 70)

    shape = (50, 50, 50) if quick else (100, 100, 100)
    n_steps = 30 if quick else 50

    test_cases = [
        ("single_porous", [(FIBERGLASS_48, 0.25)]),
        ("dual_porous", [
            (FIBERGLASS_48, 0.25),
            (PorousMaterial(name="mineral_wool", flow_resistivity=40000), 0.25),
        ]),
        ("porous_resonator", [
            (FIBERGLASS_48, 0.25),
            (HelmholtzResonator(name="trap_100Hz", cavity_volume=0.02), 0.25),
        ]),
        ("complex_multi", [
            (FIBERGLASS_48, 0.15),
            (PorousMaterial(name="foam", flow_resistivity=15000), 0.15),
            (HelmholtzResonator(name="trap_200Hz", cavity_volume=0.01), 0.15),
            (HelmholtzResonator(name="trap_400Hz", cavity_volume=0.005), 0.15),
        ]),
    ]

    results = []
    for name, materials in test_cases:
        print(f"\n  Config: {name}")
        print(f"    Materials: {len(materials)}")
        result = run_benchmark(
            name=name,
            shape=shape,
            materials=materials,
            n_steps=n_steps,
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_pole_counts(quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark different pole configurations.

    Args:
        quick: Use smaller grids

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 70)
    print("POLE COUNT BENCHMARK")
    print("=" * 70)

    shape = (50, 50, 50) if quick else (100, 100, 100)
    n_steps = 30 if quick else 50

    test_cases = [
        ("1_debye", 1, 0),
        ("2_debye", 2, 0),
        ("4_debye", 4, 0),
        ("6_debye", 6, 0),
        ("1_lorentz", 0, 1),
        ("2_lorentz", 0, 2),
        ("4_lorentz", 0, 4),
        ("2d_2l_mixed", 2, 2),
        ("4d_2l_mixed", 4, 2),
    ]

    results = []
    for name, n_debye, n_lorentz in test_cases:
        print(f"\n  Config: {name}")
        print(f"    Debye poles: {n_debye}, Lorentz poles: {n_lorentz}")

        material = create_test_material(n_debye=n_debye, n_lorentz=n_lorentz, name=name)
        materials = [(material, 0.25)]

        result = run_benchmark(
            name=name,
            shape=shape,
            materials=materials,
            n_steps=n_steps,
        )
        results.append(result)
        print_result(result)

    return results


def benchmark_thread_scaling(quick: bool = False) -> list[BenchmarkResult]:
    """Benchmark thread scaling for native backend.

    Args:
        quick: Use smaller grids

    Returns:
        List of benchmark results
    """
    if not has_native_kernels():
        print("\n  Native backend not available - skipping thread scaling")
        return []

    print("\n" + "=" * 70)
    print("THREAD SCALING BENCHMARK")
    print("=" * 70)

    try:
        from strata_fdtd import _kernels
    except ImportError:
        print("  Cannot import _kernels - skipping")
        return []

    shape = (75, 75, 75) if quick else (150, 150, 150)
    n_steps = 20 if quick else 30

    material = FIBERGLASS_48
    materials = [(material, 0.25)]

    # Get max threads
    max_threads = _kernels.get_num_threads()
    thread_counts = [1, 2, 4, 8]
    thread_counts = [t for t in thread_counts if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts.sort()

    results = []
    baseline_time = None

    for n_threads in thread_counts:
        print(f"\n  Threads: {n_threads}")
        _kernels.set_num_threads(n_threads)

        solver = setup_solver_with_materials(shape, materials, backend="native")

        # Warmup
        for _ in range(3):
            solver.step()
        solver.reset()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        elapsed = time.perf_counter() - start

        if baseline_time is None:
            baseline_time = elapsed

        time_ms = elapsed * 1000
        cells_per_sec = (np.prod(shape) * n_steps) / elapsed
        parallel_speedup = baseline_time / elapsed

        # Create result (native-only benchmark)
        result = BenchmarkResult(
            name=f"threads_{n_threads}",
            grid_size=shape,
            n_steps=n_steps,
            n_materials=len(materials),
            n_debye_poles=sum(1 for p in material.poles if p.is_debye),
            n_lorentz_poles=sum(1 for p in material.poles if p.is_lorentz),
            python_time_ms=0,  # Not measured
            native_time_ms=time_ms,
            speedup=parallel_speedup,
            python_cells_per_sec=0,
            native_cells_per_sec=cells_per_sec,
            correct=True,
            max_diff=0,
        )
        results.append(result)

        print(f"    Time: {time_ms:.1f} ms")
        print(f"    Cells/sec: {cells_per_sec/1e6:.2f} M")
        print(f"    Scaling: {parallel_speedup:.2f}x vs 1 thread")

    # Restore max threads
    _kernels.set_num_threads(max_threads)

    return results


def print_result(result: BenchmarkResult) -> None:
    """Print a single benchmark result."""
    print(f"    Python: {result.python_time_ms:.1f} ms "
          f"({result.python_cells_per_sec/1e6:.2f} M cells/s)")

    if result.native_time_ms is not None:
        print(f"    Native: {result.native_time_ms:.1f} ms "
              f"({result.native_cells_per_sec/1e6:.2f} M cells/s)")
        print(f"    Speedup: {result.speedup:.1f}x")
        status = "PASS" if result.correct else "FAIL"
        print(f"    Correctness: {status} (max diff: {result.max_diff:.2e})")
    else:
        print("    Native: N/A")


def print_summary(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print summary table of all results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    has_native = has_native_kernels()
    if has_native:
        info = get_native_info()
        print(f"\nNative backend: {info['version']}")
        print(f"OpenMP: {info['has_openmp']} ({info['num_threads']} threads)")
    else:
        print("\nNative backend: Not available")

    # Grid scaling summary
    if "grid" in all_results and all_results["grid"]:
        print("\n--- Grid Scaling ---")
        print(f"{'Grid':<12} {'Python':<15} {'Native':<15} {'Speedup':<10}")
        print("-" * 52)
        for r in all_results["grid"]:
            size = f"{r.grid_size[0]}³"
            py = f"{r.python_cells_per_sec/1e6:.1f}M/s"
            native = f"{r.native_cells_per_sec/1e6:.1f}M/s" if r.native_cells_per_sec else "N/A"
            speedup = f"{r.speedup:.1f}x" if r.speedup else "N/A"
            print(f"{size:<12} {py:<15} {native:<15} {speedup:<10}")

    # Material config summary
    if "materials" in all_results and all_results["materials"]:
        print("\n--- Material Configurations ---")
        print(f"{'Config':<20} {'Poles':<10} {'Speedup':<10}")
        print("-" * 40)
        for r in all_results["materials"]:
            poles = f"{r.n_debye_poles}D/{r.n_lorentz_poles}L"
            speedup = f"{r.speedup:.1f}x" if r.speedup else "N/A"
            print(f"{r.name:<20} {poles:<10} {speedup:<10}")

    # Pole count summary
    if "poles" in all_results and all_results["poles"]:
        print("\n--- Pole Count Impact ---")
        print(f"{'Config':<15} {'Debye':<8} {'Lorentz':<8} {'Speedup':<10}")
        print("-" * 41)
        for r in all_results["poles"]:
            speedup = f"{r.speedup:.1f}x" if r.speedup else "N/A"
            print(f"{r.name:<15} {r.n_debye_poles:<8} {r.n_lorentz_poles:<8} {speedup:<10}")

    # Thread scaling summary
    if "threads" in all_results and all_results["threads"]:
        print("\n--- Thread Scaling ---")
        print(f"{'Threads':<10} {'Cells/sec':<15} {'Scaling':<10}")
        print("-" * 35)
        for r in all_results["threads"]:
            n_threads = r.name.split("_")[1]
            rate = f"{r.native_cells_per_sec/1e6:.1f}M/s"
            scaling = f"{r.speedup:.2f}x"
            print(f"{n_threads:<10} {rate:<15} {scaling:<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ADE material kernels")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (smaller grids)")
    parser.add_argument("--grid-only", action="store_true", help="Only run grid scaling tests")
    parser.add_argument("--materials-only", action="store_true", help="Only run material config tests")
    parser.add_argument("--poles-only", action="store_true", help="Only run pole count tests")
    parser.add_argument("--threads-only", action="store_true", help="Only run thread scaling tests")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("ADE MATERIAL BENCHMARK SUITE")
    print("=" * 70)

    # Check native backend
    if has_native_kernels():
        info = get_native_info()
        print(f"\nNative backend: {info['version']}")
        print(f"OpenMP: {info['has_openmp']} ({info['num_threads']} threads)")
    else:
        print("\nNative backend: Not available")
        print("Install with: pip install -e '.[native]'")

    all_results: dict[str, list[BenchmarkResult]] = {}

    # Run selected benchmarks
    run_all = not any([args.grid_only, args.materials_only, args.poles_only, args.threads_only])

    if run_all or args.grid_only:
        all_results["grid"] = benchmark_grid_scaling(quick=args.quick)

    if run_all or args.materials_only:
        all_results["materials"] = benchmark_material_configs(quick=args.quick)

    if run_all or args.poles_only:
        all_results["poles"] = benchmark_pole_counts(quick=args.quick)

    if run_all or args.threads_only:
        all_results["threads"] = benchmark_thread_scaling(quick=args.quick)

    # Output
    if args.json:
        # Convert to JSON-serializable format
        output = {
            category: [asdict(r) for r in results]
            for category, results in all_results.items()
        }
        print("\n" + json.dumps(output, indent=2))
    else:
        print_summary(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
