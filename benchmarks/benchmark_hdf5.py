#!/usr/bin/env python3
"""
Benchmark script for HDF5 output performance.

Validates performance requirements from issue #211:
- Write overhead: <5% vs no output
- Read performance: Can load single timestep in <100ms for 100³ grid

Usage:
    python scripts/benchmark_hdf5.py
    python scripts/benchmark_hdf5.py --quick
    python scripts/benchmark_hdf5.py --json
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from strata_fdtd import FDTDSolver, GaussianPulse
from strata_fdtd.io.hdf5 import HDF5ResultReader


@dataclass
class WriteResult:
    """Results from a write overhead benchmark."""

    grid_size: int
    num_steps: int
    snapshot_interval: int | None
    baseline_time_sec: float
    with_output_time_sec: float
    overhead_percent: float
    file_size_mb: float | None


@dataclass
class ReadResult:
    """Results from a read performance benchmark."""

    grid_size: int
    num_snapshots: int
    avg_load_time_ms: float
    min_load_time_ms: float
    max_load_time_ms: float


def benchmark_write_overhead(
    grid_size: int,
    num_steps: int = 100,
    snapshot_interval: int | None = 10,
) -> WriteResult:
    """Measure write overhead compared to no output.

    Args:
        grid_size: Grid size (creates grid_size³ cells)
        num_steps: Number of timesteps to simulate
        snapshot_interval: Save snapshots every N steps (None = no snapshots)

    Returns:
        WriteResult with timing and overhead information
    """
    # Create solver
    solver = FDTDSolver(shape=(grid_size,) * 3, resolution=1e-3)
    # Position source at center of domain (grid indices)
    center_idx = grid_size // 2
    solver.add_source(
        GaussianPulse(
            position=(center_idx, center_idx, center_idx),
            frequency=40e3,
        )
    )
    duration = num_steps * solver.dt

    # Benchmark without output (baseline)
    start = time.perf_counter()
    solver.run(duration=duration)
    baseline_time = time.perf_counter() - start

    # Reset solver for second run
    solver = FDTDSolver(shape=(grid_size,) * 3, resolution=1e-3)
    # Position source at center of domain (grid indices)
    center_idx = grid_size // 2
    solver.add_source(
        GaussianPulse(
            position=(center_idx, center_idx, center_idx),
            frequency=40e3,
        )
    )

    # Benchmark with HDF5 output
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        script_content = "# Benchmark script"
        start = time.perf_counter()
        solver.run(
            duration=duration,
            output_file=str(output_path),
            script_content=script_content,
            snapshot_interval=snapshot_interval,
        )
        output_time = time.perf_counter() - start

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

    finally:
        output_path.unlink()

    overhead_percent = ((output_time - baseline_time) / baseline_time) * 100

    return WriteResult(
        grid_size=grid_size,
        num_steps=num_steps,
        snapshot_interval=snapshot_interval,
        baseline_time_sec=baseline_time,
        with_output_time_sec=output_time,
        overhead_percent=overhead_percent,
        file_size_mb=file_size_mb,
    )


def benchmark_read_performance(
    grid_size: int, num_snapshots: int = 20, num_reads: int = 10
) -> ReadResult:
    """Measure random timestep access performance.

    Args:
        grid_size: Grid size (creates grid_size³ cells)
        num_snapshots: Number of snapshots to create
        num_reads: Number of random reads to perform

    Returns:
        ReadResult with average, min, and max read times
    """
    # Create test file with snapshots
    solver = FDTDSolver(shape=(grid_size,) * 3, resolution=1e-3)
    # Position source at center of domain (grid indices)
    center_idx = grid_size // 2
    solver.add_source(
        GaussianPulse(
            position=(center_idx, center_idx, center_idx),
            frequency=40e3,
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Create file with snapshots
        duration = num_snapshots * solver.dt
        solver.run(
            duration=duration,
            output_file=str(output_path),
            script_content="# Benchmark script",
            snapshot_interval=1,  # Save every step
        )

        # Benchmark random access
        reader = HDF5ResultReader(output_path)

        # Generate random indices
        indices = [random.randint(0, num_snapshots - 1) for _ in range(num_reads)]

        times = []
        for idx in indices:
            start = time.perf_counter()
            _ = reader.load_timestep(idx)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        reader.close()

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return ReadResult(
            grid_size=grid_size,
            num_snapshots=num_snapshots,
            avg_load_time_ms=avg_time,
            min_load_time_ms=min_time,
            max_load_time_ms=max_time,
        )

    finally:
        output_path.unlink()


def run_write_benchmarks(quick: bool = False) -> list[WriteResult]:
    """Run write overhead benchmarks.

    Args:
        quick: If True, use smaller configurations for faster testing

    Returns:
        List of write benchmark results
    """
    results = []

    if quick:
        # Quick mode: smaller grids and fewer steps
        configs = [
            (50, 50, 10),
            (100, 50, 10),
        ]
    else:
        # Full benchmark suite
        configs = [
            (50, 100, 10),
            (100, 100, 10),
            (200, 100, 10),
            # Test different snapshot intervals
            (100, 100, 1),  # Every step
            (100, 100, 100),  # Once at end
            (100, 100, None),  # No snapshots
        ]

    for grid_size, num_steps, snapshot_interval in configs:
        print(
            f"  Benchmarking write overhead: {grid_size}³ grid, "
            f"{num_steps} steps, interval={snapshot_interval}"
        )
        result = benchmark_write_overhead(grid_size, num_steps, snapshot_interval)
        results.append(result)

    return results


def run_read_benchmarks(quick: bool = False) -> list[ReadResult]:
    """Run read performance benchmarks.

    Args:
        quick: If True, use smaller configurations for faster testing

    Returns:
        List of read benchmark results
    """
    results = []

    if quick:
        # Quick mode: smaller grids
        configs = [
            (50, 20),
            (100, 20),
        ]
    else:
        # Full benchmark suite
        configs = [
            (50, 20),
            (100, 20),
            (200, 20),
        ]

    for grid_size, num_snapshots in configs:
        print(
            f"  Benchmarking read performance: {grid_size}³ grid, "
            f"{num_snapshots} snapshots"
        )
        result = benchmark_read_performance(grid_size, num_snapshots)
        results.append(result)

    return results


def print_write_results(results: list[WriteResult]) -> None:
    """Print formatted write benchmark results."""
    print("\n" + "=" * 80)
    print("WRITE OVERHEAD BENCHMARKS")
    print("=" * 80)

    for result in results:
        snapshot_str = (
            f"interval={result.snapshot_interval}"
            if result.snapshot_interval is not None
            else "no snapshots"
        )
        print(f"\n{result.grid_size}³ grid, {result.num_steps} steps, {snapshot_str}")
        print("-" * 60)
        print(f"  Baseline (no output):     {result.baseline_time_sec:.3f} s")
        print(f"  With HDF5 output:         {result.with_output_time_sec:.3f} s")
        print(f"  Overhead:                 {result.overhead_percent:+.2f}%")
        if result.file_size_mb is not None:
            print(f"  File size:                {result.file_size_mb:.2f} MB")

        # Check against requirement
        if result.snapshot_interval is not None:
            if result.overhead_percent < 5.0:
                status = "✓ PASS"
            else:
                status = "✗ EXCEEDS TARGET"
            print(f"  Requirement (<5%):        {status}")


def print_read_results(results: list[ReadResult]) -> None:
    """Print formatted read benchmark results."""
    print("\n" + "=" * 80)
    print("READ PERFORMANCE BENCHMARKS")
    print("=" * 80)

    for result in results:
        print(f"\n{result.grid_size}³ grid, {result.num_snapshots} snapshots")
        print("-" * 60)
        print(f"  Average load time:        {result.avg_load_time_ms:.2f} ms")
        print(f"  Min load time:            {result.min_load_time_ms:.2f} ms")
        print(f"  Max load time:            {result.max_load_time_ms:.2f} ms")

        # Check against requirement for 100³ grid
        if result.grid_size == 100:
            if result.avg_load_time_ms < 100.0:
                status = "✓ PASS"
            else:
                status = "✗ EXCEEDS TARGET"
            print(f"  Requirement (<100ms):     {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HDF5 output performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Requirements (from issue #211):
  - Write overhead: <5% vs no output
  - Read performance: <100ms to load single timestep for 100³ grid

Examples:
  python scripts/benchmark_hdf5.py              # Full benchmark
  python scripts/benchmark_hdf5.py --quick      # Quick test
  python scripts/benchmark_hdf5.py --json       # JSON output
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller configurations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    if not args.json:
        print("=" * 80)
        print("HDF5 OUTPUT PERFORMANCE BENCHMARK")
        print("=" * 80)
        print()

    # Run write benchmarks
    if not args.json:
        print("Running write overhead benchmarks...")
    write_results = run_write_benchmarks(quick=args.quick)

    # Run read benchmarks
    if not args.json:
        print("\nRunning read performance benchmarks...")
    read_results = run_read_benchmarks(quick=args.quick)

    # Output results
    if args.json:
        output = {
            "write_overhead": [
                {
                    "grid_size": r.grid_size,
                    "num_steps": r.num_steps,
                    "snapshot_interval": r.snapshot_interval,
                    "baseline_time_sec": r.baseline_time_sec,
                    "with_output_time_sec": r.with_output_time_sec,
                    "overhead_percent": r.overhead_percent,
                    "file_size_mb": r.file_size_mb,
                }
                for r in write_results
            ],
            "read_performance": [
                {
                    "grid_size": r.grid_size,
                    "num_snapshots": r.num_snapshots,
                    "avg_load_time_ms": r.avg_load_time_ms,
                    "min_load_time_ms": r.min_load_time_ms,
                    "max_load_time_ms": r.max_load_time_ms,
                }
                for r in read_results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_write_results(write_results)
        print_read_results(read_results)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Check write overhead requirement
        write_with_snapshots = [r for r in write_results if r.snapshot_interval is not None]
        max_overhead = max(r.overhead_percent for r in write_with_snapshots)
        write_pass = max_overhead < 5.0

        # Check read performance requirement
        read_100 = [r for r in read_results if r.grid_size == 100]
        read_pass = all(r.avg_load_time_ms < 100.0 for r in read_100) if read_100 else False

        print(f"\nWrite overhead (<5%):        {'✓ PASS' if write_pass else '✗ FAIL'}")
        if not write_pass:
            print(f"  Maximum overhead: {max_overhead:.2f}%")

        print(f"Read performance (<100ms):   {'✓ PASS' if read_pass else '✗ FAIL'}")
        if read_100 and not read_pass:
            avg_100 = read_100[0].avg_load_time_ms
            print(f"  100³ grid average: {avg_100:.2f} ms")


if __name__ == "__main__":
    main()
