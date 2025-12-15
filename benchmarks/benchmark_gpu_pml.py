#!/usr/bin/env python3
"""
GPU PML Performance Benchmark Script

Empirically measures the performance impact of PML (Perfectly Matched Layer)
absorbing boundaries on GPU FDTD solvers to validate documented overhead claims.

This script benchmarks:
1. Single-band GPU solver: Rigid vs PML overhead across grid sizes
2. Batched GPU solver: Verify PML overhead is constant (doesn't scale with band count)
3. Memory usage: Validate minimal overhead from PML decay tensors

Expected results (from PR #161 documentation):
- Memory overhead: 3 small 1D decay tensors
- Compute overhead: ~5-10% per timestep (3 element-wise multiplications)
- Batched solver: No N× scaling (PML shared across bands)

Usage:
    python3 scripts/benchmark_gpu_pml.py                    # Run full benchmark suite
    python3 scripts/benchmark_gpu_pml.py --quick            # Quick benchmark (smaller grids)
    python3 scripts/benchmark_gpu_pml.py --single-only      # Only single-band tests
    python3 scripts/benchmark_gpu_pml.py --batched-only     # Only batched solver tests
    python3 scripts/benchmark_gpu_pml.py --memory-only      # Only memory usage tests
    python3 scripts/benchmark_gpu_pml.py --json             # Output results as JSON
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

from strata_fdtd.fdtd_gpu import (
    GPUFDTDSolver,
    BatchedGPUFDTDSolver,
    has_gpu_support,
    get_gpu_info,
)


@dataclass
class SingleBandResult:
    """Results from a single-band benchmark run."""
    name: str
    grid_size: tuple[int, int, int]
    n_steps: int
    pml_layers: int
    time_ms: float
    cells_per_sec: float
    overhead_pct: float | None  # Overhead vs rigid (0 PML layers)
    memory_mb: float


@dataclass
class BatchedResult:
    """Results from a batched solver benchmark run."""
    name: str
    grid_size: tuple[int, int, int]
    n_bands: int
    n_steps: int
    pml_layers: int
    time_ms: float
    cells_per_sec: float
    overhead_pct: float | None  # Overhead vs rigid (0 PML layers)
    memory_mb: float


@dataclass
class MemoryResult:
    """Memory usage comparison."""
    name: str
    grid_size: tuple[int, int, int]
    n_bands: int | None  # None for single-band
    pml_layers: int
    memory_mb: float
    overhead_mb: float | None  # Overhead vs rigid
    overhead_pct: float | None


def run_single_band_benchmark(
    shape: tuple[int, int, int],
    pml_layers: int,
    n_steps: int = 100,
    warmup_steps: int = 10,
) -> SingleBandResult:
    """Run a single-band GPU solver benchmark.

    Args:
        shape: Grid dimensions
        pml_layers: Number of PML layers (0 = rigid boundaries)
        n_steps: Number of timesteps
        warmup_steps: Warmup steps before timing

    Returns:
        SingleBandResult with timing and throughput data
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for GPU benchmarks")

    solver = GPUFDTDSolver(
        shape=shape,
        resolution=1e-3,
        pml_layers=pml_layers,
    )

    # Add source for realistic simulation
    solver.add_source(frequency=1000.0)

    # Warmup
    for _ in range(warmup_steps):
        solver.step()

    # Ensure GPU synchronization
    if solver.using_gpu:
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        solver.step()

    # Synchronize before stopping timer
    if solver.using_gpu:
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    # Compute metrics
    total_cells = np.prod(shape)
    time_ms = elapsed * 1000
    cells_per_sec = (total_cells * n_steps) / elapsed
    memory_mb = solver.memory_usage_mb()

    name = f"{shape[0]}^3_pml{pml_layers}"

    return SingleBandResult(
        name=name,
        grid_size=shape,
        n_steps=n_steps,
        pml_layers=pml_layers,
        time_ms=time_ms,
        cells_per_sec=cells_per_sec,
        overhead_pct=None,  # Computed later
        memory_mb=memory_mb,
    )


def run_batched_benchmark(
    shape: tuple[int, int, int],
    n_bands: int,
    pml_layers: int,
    n_steps: int = 50,
    warmup_steps: int = 5,
) -> BatchedResult:
    """Run a batched GPU solver benchmark.

    Args:
        shape: Grid dimensions
        n_bands: Number of frequency bands
        pml_layers: Number of PML layers (0 = rigid boundaries)
        n_steps: Number of timesteps
        warmup_steps: Warmup steps before timing

    Returns:
        BatchedResult with timing and throughput data
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for GPU benchmarks")

    # Create bands with different frequencies
    bands = []
    for i in range(n_bands):
        freq = 250 * (2 ** i)  # 250, 500, 1000, 2000, ...
        bw = freq * 0.5
        bands.append((freq, bw))

    solver = BatchedGPUFDTDSolver(
        shape=shape,
        resolution=1e-3,
        bands=bands,
        pml_layers=pml_layers,
    )

    # Warmup
    for _ in range(warmup_steps):
        solver.step()

    # Ensure GPU synchronization
    if solver.using_gpu:
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        solver.step()

    # Synchronize before stopping timer
    if solver.using_gpu:
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    # Compute metrics
    total_cells = np.prod(shape)
    time_ms = elapsed * 1000
    cells_per_sec = (total_cells * n_steps) / elapsed
    memory_mb = solver.memory_usage_mb()

    name = f"{shape[0]}^3_{n_bands}bands_pml{pml_layers}"

    return BatchedResult(
        name=name,
        grid_size=shape,
        n_bands=n_bands,
        n_steps=n_steps,
        pml_layers=pml_layers,
        time_ms=time_ms,
        cells_per_sec=cells_per_sec,
        overhead_pct=None,  # Computed later
        memory_mb=memory_mb,
    )


def benchmark_single_band_overhead(quick: bool = False) -> list[SingleBandResult]:
    """Benchmark single-band solver with varying PML depths.

    Measures execution time for rigid vs PML boundaries across grid sizes.

    Args:
        quick: Use smaller grids and fewer steps

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 70)
    print("SINGLE-BAND GPU SOLVER: PML OVERHEAD")
    print("=" * 70)

    if quick:
        grid_sizes = [(50, 50, 50), (75, 75, 75)]
        pml_depths = [0, 10, 15]
        n_steps = 50
    else:
        grid_sizes = [(50, 50, 50), (100, 100, 100), (150, 150, 150)]
        pml_depths = [0, 10, 15, 20]
        n_steps = 100

    results = []
    baseline_times = {}  # Store rigid boundary times for overhead calculation

    for shape in grid_sizes:
        print(f"\n  Grid: {shape[0]}³ ({np.prod(shape)/1e6:.1f}M cells)")

        for pml in pml_depths:
            print(f"    PML layers: {pml}")
            result = run_single_band_benchmark(
                shape=shape,
                pml_layers=pml,
                n_steps=n_steps,
            )

            # Store baseline (rigid boundary) time
            if pml == 0:
                baseline_times[shape] = result.time_ms
                result.overhead_pct = 0.0
            else:
                # Compute overhead vs rigid
                baseline = baseline_times[shape]
                overhead_pct = ((result.time_ms - baseline) / baseline) * 100
                result.overhead_pct = overhead_pct

            results.append(result)
            print_single_result(result)

    return results


def benchmark_batched_scaling(quick: bool = False) -> list[BatchedResult]:
    """Benchmark batched solver to verify PML doesn't scale with band count.

    Measures overhead as band count increases, verifying PML tensors are
    shared across bands (no N× scaling).

    Args:
        quick: Use smaller grids and fewer bands

    Returns:
        List of benchmark results
    """
    print("\n" + "=" * 70)
    print("BATCHED GPU SOLVER: PML SCALING WITH BAND COUNT")
    print("=" * 70)

    shape = (75, 75, 75) if quick else (100, 100, 100)
    pml_layers = 10
    n_steps = 30 if quick else 50

    if quick:
        band_counts = [1, 2, 4]
    else:
        band_counts = [1, 2, 4, 8, 16]

    print(f"\n  Grid: {shape[0]}³, PML layers: {pml_layers}")

    results = []
    baseline_times = {}  # Store rigid times per band count

    for n_bands in band_counts:
        # Test with rigid boundaries
        print(f"\n    {n_bands} bands (rigid)")
        result_rigid = run_batched_benchmark(
            shape=shape,
            n_bands=n_bands,
            pml_layers=0,
            n_steps=n_steps,
        )
        baseline_times[n_bands] = result_rigid.time_ms
        result_rigid.overhead_pct = 0.0
        results.append(result_rigid)
        print_batched_result(result_rigid)

        # Test with PML
        print(f"    {n_bands} bands (PML)")
        result_pml = run_batched_benchmark(
            shape=shape,
            n_bands=n_bands,
            pml_layers=pml_layers,
            n_steps=n_steps,
        )
        baseline = baseline_times[n_bands]
        overhead_pct = ((result_pml.time_ms - baseline) / baseline) * 100
        result_pml.overhead_pct = overhead_pct
        results.append(result_pml)
        print_batched_result(result_pml)

    return results


def benchmark_memory_usage(quick: bool = False) -> list[MemoryResult]:
    """Measure GPU memory consumption with and without PML.

    Verifies minimal overhead from PML decay tensors.

    Args:
        quick: Use smaller grids

    Returns:
        List of memory usage results
    """
    print("\n" + "=" * 70)
    print("MEMORY USAGE COMPARISON")
    print("=" * 70)

    if quick:
        grid_sizes = [(50, 50, 50), (100, 100, 100)]
        band_counts = [1, 4]
    else:
        grid_sizes = [(50, 50, 50), (100, 100, 100), (150, 150, 150)]
        band_counts = [1, 4, 8]

    pml_layers = 15

    results = []

    # Single-band solver
    print("\n  Single-Band Solver:")
    for shape in grid_sizes:
        # Rigid boundaries
        solver_rigid = GPUFDTDSolver(shape=shape, resolution=1e-3, pml_layers=0)
        mem_rigid = solver_rigid.memory_usage_mb()

        result_rigid = MemoryResult(
            name=f"{shape[0]}^3_single_rigid",
            grid_size=shape,
            n_bands=None,
            pml_layers=0,
            memory_mb=mem_rigid,
            overhead_mb=0.0,
            overhead_pct=0.0,
        )
        results.append(result_rigid)

        # PML boundaries
        solver_pml = GPUFDTDSolver(shape=shape, resolution=1e-3, pml_layers=pml_layers)
        mem_pml = solver_pml.memory_usage_mb()

        overhead_mb = mem_pml - mem_rigid
        overhead_pct = (overhead_mb / mem_rigid) * 100

        result_pml = MemoryResult(
            name=f"{shape[0]}^3_single_pml{pml_layers}",
            grid_size=shape,
            n_bands=None,
            pml_layers=pml_layers,
            memory_mb=mem_pml,
            overhead_mb=overhead_mb,
            overhead_pct=overhead_pct,
        )
        results.append(result_pml)

        print(f"    {shape[0]}³: Rigid={mem_rigid:.1f}MB, "
              f"PML={mem_pml:.1f}MB, Overhead={overhead_mb:.2f}MB ({overhead_pct:.1f}%)")

    # Batched solver
    print("\n  Batched Solver:")
    shape = grid_sizes[1]  # Use medium-sized grid
    for n_bands in band_counts:
        # Create bands
        bands = [(250 * (2 ** i), 250 * (2 ** i) * 0.5) for i in range(n_bands)]

        # Rigid boundaries
        solver_rigid = BatchedGPUFDTDSolver(
            shape=shape, resolution=1e-3, bands=bands, pml_layers=0
        )
        mem_rigid = solver_rigid.memory_usage_mb()

        result_rigid = MemoryResult(
            name=f"{shape[0]}^3_{n_bands}bands_rigid",
            grid_size=shape,
            n_bands=n_bands,
            pml_layers=0,
            memory_mb=mem_rigid,
            overhead_mb=0.0,
            overhead_pct=0.0,
        )
        results.append(result_rigid)

        # PML boundaries
        solver_pml = BatchedGPUFDTDSolver(
            shape=shape, resolution=1e-3, bands=bands, pml_layers=pml_layers
        )
        mem_pml = solver_pml.memory_usage_mb()

        overhead_mb = mem_pml - mem_rigid
        overhead_pct = (overhead_mb / mem_rigid) * 100

        result_pml = MemoryResult(
            name=f"{shape[0]}^3_{n_bands}bands_pml{pml_layers}",
            grid_size=shape,
            n_bands=n_bands,
            pml_layers=pml_layers,
            memory_mb=mem_pml,
            overhead_mb=overhead_mb,
            overhead_pct=overhead_pct,
        )
        results.append(result_pml)

        print(f"    {n_bands} bands: Rigid={mem_rigid:.1f}MB, "
              f"PML={mem_pml:.1f}MB, Overhead={overhead_mb:.2f}MB ({overhead_pct:.1f}%)")

    return results


def print_single_result(result: SingleBandResult) -> None:
    """Print a single-band benchmark result."""
    print(f"      Time: {result.time_ms:.1f} ms")
    print(f"      Throughput: {result.cells_per_sec/1e6:.2f} M cells/s")
    if result.overhead_pct is not None:
        print(f"      Overhead: {result.overhead_pct:+.1f}%")
    print(f"      Memory: {result.memory_mb:.1f} MB")


def print_batched_result(result: BatchedResult) -> None:
    """Print a batched benchmark result."""
    print(f"      Time: {result.time_ms:.1f} ms")
    print(f"      Throughput: {result.cells_per_sec/1e6:.2f} M cells/s")
    if result.overhead_pct is not None:
        print(f"      Overhead: {result.overhead_pct:+.1f}%")
    print(f"      Memory: {result.memory_mb:.1f} MB")


def print_summary(
    single_results: list[SingleBandResult] | None,
    batched_results: list[BatchedResult] | None,
    memory_results: list[MemoryResult] | None,
) -> None:
    """Print summary table of all results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\nGPU Backend: {gpu_info['backend'].upper()}")
        print(f"PyTorch: {gpu_info['pytorch_version']}")
    else:
        print("\nGPU Backend: Not available")

    # Single-band summary
    if single_results:
        print("\n--- Single-Band PML Overhead ---")
        print(f"{'Grid':<12} {'PML':<8} {'Time (ms)':<12} {'Overhead':<12} {'Throughput':<15}")
        print("-" * 59)
        for r in single_results:
            grid = f"{r.grid_size[0]}³"
            pml = str(r.pml_layers)
            time = f"{r.time_ms:.1f}"
            overhead = f"{r.overhead_pct:+.1f}%" if r.overhead_pct is not None else "baseline"
            throughput = f"{r.cells_per_sec/1e6:.2f}M/s"
            print(f"{grid:<12} {pml:<8} {time:<12} {overhead:<12} {throughput:<15}")

    # Batched scaling summary
    if batched_results:
        print("\n--- Batched Solver: PML Scaling ---")
        print(f"{'Bands':<8} {'PML':<8} {'Time (ms)':<12} {'Overhead':<12}")
        print("-" * 40)
        for r in batched_results:
            bands = str(r.n_bands)
            pml = "rigid" if r.pml_layers == 0 else f"PML{r.pml_layers}"
            time = f"{r.time_ms:.1f}"
            overhead = f"{r.overhead_pct:+.1f}%" if r.overhead_pct is not None else "baseline"
            print(f"{bands:<8} {pml:<8} {time:<12} {overhead:<12}")

    # Memory summary
    if memory_results:
        print("\n--- Memory Overhead ---")
        print(f"{'Config':<25} {'Memory':<12} {'Overhead':<15}")
        print("-" * 52)
        for r in memory_results:
            config = r.name.replace("_", " ")
            mem = f"{r.memory_mb:.1f}MB"
            if r.overhead_mb is not None and r.overhead_mb > 0:
                overhead = f"+{r.overhead_mb:.2f}MB ({r.overhead_pct:.1f}%)"
            else:
                overhead = "baseline"
            print(f"{config:<25} {mem:<12} {overhead:<15}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU PML performance impact"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (smaller grids)")
    parser.add_argument("--single-only", action="store_true", help="Only run single-band tests")
    parser.add_argument("--batched-only", action="store_true", help="Only run batched solver tests")
    parser.add_argument("--memory-only", action="store_true", help="Only run memory usage tests")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU PML PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Check GPU support
    if not has_gpu_support():
        print("\nERROR: GPU (MPS) support not available")
        print("This benchmark requires PyTorch with MPS backend (Apple Silicon)")
        return 1

    gpu_info = get_gpu_info()
    print(f"\nGPU Backend: {gpu_info['backend'].upper()}")
    print(f"PyTorch: {gpu_info['pytorch_version']}")

    # Run selected benchmarks
    run_all = not any([args.single_only, args.batched_only, args.memory_only])

    single_results = None
    batched_results = None
    memory_results = None

    if run_all or args.single_only:
        single_results = benchmark_single_band_overhead(quick=args.quick)

    if run_all or args.batched_only:
        batched_results = benchmark_batched_scaling(quick=args.quick)

    if run_all or args.memory_only:
        memory_results = benchmark_memory_usage(quick=args.quick)

    # Output
    if args.json:
        # Convert to JSON-serializable format
        output = {}
        if single_results:
            output["single_band"] = [asdict(r) for r in single_results]
        if batched_results:
            output["batched"] = [asdict(r) for r in batched_results]
        if memory_results:
            output["memory"] = [asdict(r) for r in memory_results]
        print("\n" + json.dumps(output, indent=2))
    else:
        print_summary(single_results, batched_results, memory_results)

    # Validate results against documented claims
    print("\n" + "=" * 70)
    print("VALIDATION AGAINST DOCUMENTED CLAIMS")
    print("=" * 70)

    if single_results:
        # Check that PML overhead is within 5-10% range
        pml_overheads = [
            r.overhead_pct for r in single_results
            if r.pml_layers > 0 and r.overhead_pct is not None
        ]
        if pml_overheads:
            avg_overhead = sum(pml_overheads) / len(pml_overheads)
            print(f"\nSingle-band PML overhead: {avg_overhead:.1f}% (expected: 5-10%)")
            if 0 <= avg_overhead <= 15:
                print("✓ PASS: Overhead within expected range")
            else:
                print("✗ FAIL: Overhead outside expected range")

    if batched_results:
        # Check that PML overhead doesn't scale with band count
        pml_results = [r for r in batched_results if r.pml_layers > 0]
        if len(pml_results) >= 2:
            overheads = [r.overhead_pct for r in pml_results if r.overhead_pct is not None]
            if overheads:
                overhead_variance = max(overheads) - min(overheads)
                print(f"\nBatched PML overhead variance: {overhead_variance:.1f}% "
                      f"(expected: minimal)")
                if overhead_variance < 5.0:
                    print("✓ PASS: Overhead constant across band counts")
                else:
                    print("⚠ WARNING: Overhead varies with band count")

    if memory_results:
        # Check that memory overhead is minimal
        pml_mem_results = [r for r in memory_results if r.pml_layers > 0 and r.overhead_pct is not None]
        if pml_mem_results:
            avg_mem_overhead = sum(r.overhead_pct for r in pml_mem_results) / len(pml_mem_results)
            print(f"\nMemory overhead: {avg_mem_overhead:.1f}% (expected: minimal)")
            if avg_mem_overhead < 5.0:
                print("✓ PASS: Memory overhead minimal")
            else:
                print("⚠ WARNING: Memory overhead higher than expected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
