"""
Comprehensive FDTD performance benchmarks and regression tests.

This module provides:
1. Micro-benchmarks for individual FDTD operations
2. Full-step throughput benchmarks across grid sizes
3. Thread scaling tests for OpenMP parallelization
4. Memory bandwidth measurements
5. Python/C++ backend equivalence validation

Run benchmarks:
    pytest tests/strata_fdtd/test_fdtd_benchmarks.py -v

Run with JSON output for CI:
    pytest tests/strata_fdtd/test_fdtd_benchmarks.py -v \
        --benchmark-json=benchmarks/results.json

Skip slow benchmarks:
    pytest tests/strata_fdtd/test_fdtd_benchmarks.py -v -m "not slow"

Dependencies:
    pytest-benchmark>=4.0
"""

import os
import platform
from dataclasses import dataclass

import numpy as np
import pytest

from strata_fdtd.core.solver import (
    FDTDSolver,
    GaussianPulse,
    get_native_info,
    has_native_kernels,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    # Grid sizes for parametrized tests
    micro_sizes: list[int]
    full_sizes: list[int]
    scaling_size: int

    # Number of steps for various benchmarks
    micro_steps: int
    full_steps: int
    scaling_steps: int

    # Thread counts for scaling tests
    thread_counts: list[int]

    # Regression thresholds
    regression_threshold: float  # 10% = 0.10


# Detect system capabilities for adaptive configuration
def _get_default_config() -> BenchmarkConfig:
    """Generate benchmark config based on system capabilities."""
    # Get number of CPU cores
    cpu_count = os.cpu_count() or 4

    # Thread counts for scaling tests (powers of 2 up to cpu_count)
    thread_counts = [1]
    t = 2
    while t <= cpu_count:
        thread_counts.append(t)
        t *= 2
    if cpu_count not in thread_counts:
        thread_counts.append(cpu_count)

    return BenchmarkConfig(
        micro_sizes=[64, 100],
        full_sizes=[64, 100, 150],
        scaling_size=100,
        micro_steps=50,
        full_steps=50,
        scaling_steps=20,
        thread_counts=thread_counts,
        regression_threshold=0.10,
    )


CONFIG = _get_default_config()


# =============================================================================
# Micro-benchmarks: Individual Operations
# =============================================================================


class TestMicroBenchmarks:
    """Micro-benchmarks for individual FDTD operations.

    These tests isolate specific operations to identify hotspots
    and measure the impact of optimizations.
    """

    @pytest.mark.benchmark(group="micro-velocity")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_velocity_update_x(self, benchmark, size: int):
        """Benchmark velocity-x update kernel."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        p = solver.p
        vx = solver.vx
        coeff_v = solver._coeff_v

        def velocity_x_update():
            vx[:-1, :, :] += coeff_v * (p[1:, :, :] - p[:-1, :, :])

        # Record metadata before benchmarking
        cells = (size - 1) * size * size
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["cells"] = cells

        benchmark(velocity_x_update)

    @pytest.mark.benchmark(group="micro-velocity")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_velocity_update_y(self, benchmark, size: int):
        """Benchmark velocity-y update kernel."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        p = solver.p
        vy = solver.vy
        coeff_v = solver._coeff_v

        def velocity_y_update():
            vy[:, :-1, :] += coeff_v * (p[:, 1:, :] - p[:, :-1, :])

        cells = size * (size - 1) * size
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["cells"] = cells

        benchmark(velocity_y_update)

    @pytest.mark.benchmark(group="micro-velocity")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_velocity_update_z(self, benchmark, size: int):
        """Benchmark velocity-z update kernel."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        p = solver.p
        vz = solver.vz
        coeff_v = solver._coeff_v

        def velocity_z_update():
            vz[:, :, :-1] += coeff_v * (p[:, :, 1:] - p[:, :, :-1])

        cells = size * size * (size - 1)
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["cells"] = cells

        benchmark(velocity_z_update)

    @pytest.mark.benchmark(group="micro-divergence")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_divergence_computation(self, benchmark, size: int):
        """Benchmark divergence computation."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        vx, vy, vz = solver.vx, solver.vy, solver.vz
        nx, ny, nz = size, size, size

        def compute_divergence():
            div = np.zeros((nx, ny, nz), dtype=np.float32)
            div[0, :, :] = vx[0, :, :]
            div[1:, :, :] = vx[1:nx, :, :] - vx[: nx - 1, :, :]
            div[:, 0, :] += vy[:, 0, :]
            div[:, 1:, :] += vy[:, 1:ny, :] - vy[:, : ny - 1, :]
            div[:, :, 0] += vz[:, :, 0]
            div[:, :, 1:] += vz[:, :, 1:nz] - vz[:, :, : nz - 1]
            return div

        total_cells = size**3
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["cells"] = total_cells

        benchmark(compute_divergence)

    @pytest.mark.benchmark(group="micro-pressure")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_pressure_update(self, benchmark, size: int):
        """Benchmark pressure update."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        p = solver.p
        coeff_p = solver._coeff_p
        div = np.random.randn(size, size, size).astype(np.float32)

        def pressure_update():
            p[:] += coeff_p * div

        total_cells = size**3
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["cells"] = total_cells

        benchmark(pressure_update)

    @pytest.mark.benchmark(group="micro-boundary")
    @pytest.mark.parametrize("size", CONFIG.micro_sizes)
    def test_rigid_boundary_application(self, benchmark, size: int):
        """Benchmark rigid boundary mask application."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")

        # Create geometry with some solid regions (20% solid)
        geometry = np.random.random((size, size, size)) > 0.2
        solver.set_geometry(geometry)

        vx = solver.vx
        geom = solver.geometry

        def apply_boundary():
            solid_vx = ~geom[:-1, :, :] | ~geom[1:, :, :]
            vx[:-1, :, :][solid_vx] = 0.0

        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["solid_fraction"] = 1.0 - geometry.mean()

        benchmark(apply_boundary)


# =============================================================================
# Full-Step Benchmarks
# =============================================================================


class TestFullStepBenchmarks:
    """Benchmark complete FDTD timesteps across backends and grid sizes."""

    @pytest.mark.benchmark(group="full-step")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    def test_step_python(self, benchmark, size: int):
        """Benchmark full FDTD step with Python backend."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        solver.add_source(
            GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
        )

        # Warmup
        for _ in range(5):
            solver.step()

        total_cells = size**3
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["backend"] = "python"
        benchmark.extra_info["total_cells"] = total_cells

        benchmark(solver.step)

    @pytest.mark.benchmark(group="full-step")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_step_native(self, benchmark, size: int):
        """Benchmark full FDTD step with native C++ backend."""
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="native")
        solver.add_source(
            GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
        )

        # Warmup
        for _ in range(5):
            solver.step()

        total_cells = size**3
        native_info = get_native_info()
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["backend"] = "native"
        benchmark.extra_info["total_cells"] = total_cells
        benchmark.extra_info["num_threads"] = native_info.get("num_threads", 1)

        benchmark(solver.step)

    @pytest.mark.benchmark(group="full-step-multi")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    def test_multistep_python(self, benchmark, size: int):
        """Benchmark multiple FDTD steps with Python backend."""
        n_steps = CONFIG.full_steps
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
        solver.add_source(
            GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
        )

        def run_steps():
            for _ in range(n_steps):
                solver.step()

        total_cells = size**3 * n_steps
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["n_steps"] = n_steps
        benchmark.extra_info["backend"] = "python"
        benchmark.extra_info["total_cells"] = total_cells

        benchmark(run_steps)

    @pytest.mark.benchmark(group="full-step-multi")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_multistep_native(self, benchmark, size: int):
        """Benchmark multiple FDTD steps with native C++ backend."""
        n_steps = CONFIG.full_steps
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="native")
        solver.add_source(
            GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
        )

        def run_steps():
            for _ in range(n_steps):
                solver.step()

        total_cells = size**3 * n_steps
        native_info = get_native_info()
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["n_steps"] = n_steps
        benchmark.extra_info["backend"] = "native"
        benchmark.extra_info["total_cells"] = total_cells
        benchmark.extra_info["num_threads"] = native_info.get("num_threads", 1)

        benchmark(run_steps)


# =============================================================================
# Thread Scaling Benchmarks
# =============================================================================


class TestThreadScaling:
    """Measure parallel efficiency across different thread counts."""

    @pytest.mark.benchmark(group="thread-scaling")
    @pytest.mark.slow
    @pytest.mark.parametrize("threads", CONFIG.thread_counts)
    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_thread_scaling(self, benchmark, threads: int):
        """Measure performance at different thread counts."""
        from strata_fdtd import _kernels

        native_info = get_native_info()
        if not native_info.get("has_openmp", False):
            pytest.skip("OpenMP not available in native build")

        # Set thread count
        _kernels.set_num_threads(threads)

        size = CONFIG.scaling_size
        n_steps = CONFIG.scaling_steps
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="native")
        solver.add_source(
            GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
        )

        def run_steps():
            for _ in range(n_steps):
                solver.step()

        total_cells = size**3 * n_steps
        benchmark.extra_info["threads"] = threads
        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["total_cells"] = total_cells

        benchmark(run_steps)

    @pytest.mark.slow
    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_parallel_efficiency_report(self):
        """Generate a parallel efficiency report (not a benchmark)."""
        from strata_fdtd import _kernels

        native_info = get_native_info()
        if not native_info.get("has_openmp", False):
            pytest.skip("OpenMP not available in native build")

        size = CONFIG.scaling_size
        n_steps = CONFIG.scaling_steps
        results = {}

        for threads in CONFIG.thread_counts:
            _kernels.set_num_threads(threads)

            solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="native")
            solver.add_source(
                GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
            )

            # Warmup
            for _ in range(3):
                solver.step()
            solver.reset()

            # Time
            import time

            start = time.perf_counter()
            for _ in range(n_steps):
                solver.step()
            elapsed = time.perf_counter() - start

            results[threads] = elapsed

        # Calculate parallel efficiency
        baseline = results[1]
        print("\n\nParallel Scaling Report")
        print("=" * 50)
        print(f"{'Threads':<10} {'Time (ms)':<12} {'Speedup':<10} {'Efficiency':<10}")
        print("-" * 50)

        for threads, elapsed in sorted(results.items()):
            speedup = baseline / elapsed
            efficiency = speedup / threads * 100
            print(f"{threads:<10} {elapsed*1000:<12.2f} {speedup:<10.2f} {efficiency:<10.1f}%")


# =============================================================================
# Memory Bandwidth Tests
# =============================================================================


class TestMemoryBandwidth:
    """Measure effective memory bandwidth utilization."""

    @pytest.mark.benchmark(group="bandwidth")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    def test_memory_bandwidth_python(self, benchmark, size: int):
        """Measure effective memory bandwidth (Python backend)."""
        n_steps = CONFIG.full_steps
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")

        def run_steps():
            for _ in range(n_steps):
                solver.step()

        # ~20 bytes per cell per step (4 fields * 4 bytes + overhead)
        bytes_per_step = 20 * size**3
        total_bytes = bytes_per_step * n_steps

        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["backend"] = "python"
        benchmark.extra_info["total_bytes"] = total_bytes

        benchmark(run_steps)

    @pytest.mark.benchmark(group="bandwidth")
    @pytest.mark.parametrize("size", CONFIG.full_sizes)
    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_memory_bandwidth_native(self, benchmark, size: int):
        """Measure effective memory bandwidth (native backend)."""
        n_steps = CONFIG.full_steps
        solver = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="native")

        def run_steps():
            for _ in range(n_steps):
                solver.step()

        bytes_per_step = 20 * size**3
        total_bytes = bytes_per_step * n_steps

        benchmark.extra_info["grid_size"] = size
        benchmark.extra_info["backend"] = "native"
        benchmark.extra_info["total_bytes"] = total_bytes

        benchmark(run_steps)


# =============================================================================
# Backend Equivalence Tests
# =============================================================================


class TestBackendEquivalence:
    """Verify Python and C++ backends produce identical results."""

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_single_step_equivalence(self):
        """Verify single step produces identical results."""
        shape = (50, 50, 50)

        # Create identical solvers
        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        # Add identical sources
        source_pos = (shape[0] // 4, shape[1] // 2, shape[2] // 2)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        # Run single step
        solver_py.step()
        solver_native.step()

        # Compare results
        np.testing.assert_allclose(
            solver_py.p, solver_native.p, rtol=1e-5, atol=1e-7, err_msg="Pressure mismatch"
        )
        np.testing.assert_allclose(
            solver_py.vx, solver_native.vx, rtol=1e-5, atol=1e-7, err_msg="Velocity-x mismatch"
        )
        np.testing.assert_allclose(
            solver_py.vy, solver_native.vy, rtol=1e-5, atol=1e-7, err_msg="Velocity-y mismatch"
        )
        np.testing.assert_allclose(
            solver_py.vz, solver_native.vz, rtol=1e-5, atol=1e-7, err_msg="Velocity-z mismatch"
        )

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_multistep_equivalence(self):
        """Verify 100 steps produces equivalent results (within floating point tolerance)."""
        shape = (50, 50, 50)
        n_steps = 100

        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        source_pos = (shape[0] // 4, shape[1] // 2, shape[2] // 2)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        for _ in range(n_steps):
            solver_py.step()
            solver_native.step()

        # Allow slightly larger tolerance for accumulated differences
        np.testing.assert_allclose(
            solver_py.p,
            solver_native.p,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Pressure mismatch after {n_steps} steps",
        )

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_equivalence_with_geometry(self):
        """Verify equivalence with complex geometry."""
        shape = (50, 50, 50)
        n_steps = 50

        # Create geometry with solid block
        geometry = np.ones(shape, dtype=bool)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        geometry[cx - 5 : cx + 5, cy - 5 : cy + 5, cz - 5 : cz + 5] = False

        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        solver_py.set_geometry(geometry)
        solver_native.set_geometry(geometry)

        source_pos = (shape[0] // 4, shape[1] // 2, shape[2] // 2)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        for _ in range(n_steps):
            solver_py.step()
            solver_native.step()

        np.testing.assert_allclose(
            solver_py.p,
            solver_native.p,
            rtol=1e-4,
            atol=1e-5,  # Slightly larger tolerance for geometry (boundary reflections accumulate)
            err_msg="Pressure mismatch with geometry",
        )

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_energy_conservation_equivalence(self):
        """Verify both backends conserve energy similarly."""
        shape = (60, 60, 60)
        n_steps = 200

        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        source_pos = (shape[0] // 4, shape[1] // 2, shape[2] // 2)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        # Run simulation
        for _ in range(n_steps):
            solver_py.step()
            solver_native.step()

        # Compare total energy
        energy_py = np.sum(solver_py.p**2)
        energy_native = np.sum(solver_native.p**2)

        # Energies should be very close (within 1%)
        rel_diff = abs(energy_py - energy_native) / max(energy_py, energy_native)
        assert rel_diff < 0.01, f"Energy difference {rel_diff:.4%} exceeds 1% threshold"


# =============================================================================
# Machine Information
# =============================================================================


class TestMachineInfo:
    """Capture machine information for benchmark reports."""

    def test_machine_info(self):
        """Record machine information (for benchmark JSON output)."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }

        # Add native kernel info
        if has_native_kernels():
            native_info = get_native_info()
            info["native_version"] = native_info.get("version")
            info["native_openmp"] = native_info.get("has_openmp")
            info["native_threads"] = native_info.get("num_threads")

        print("\nMachine Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Always pass - this test is for information capture
        assert True


# =============================================================================
# Speedup Comparison Test
# =============================================================================


class TestSpeedupComparison:
    """Compare Python vs Native backend performance."""

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_speedup_report(self):
        """Generate speedup comparison report."""
        import time

        sizes = [50, 75, 100]
        n_steps = 50
        results = []

        for size in sizes:
            # Python backend
            solver_py = FDTDSolver(shape=(size, size, size), resolution=1e-3, backend="python")
            solver_py.add_source(
                GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
            )

            # Warmup
            for _ in range(3):
                solver_py.step()
            solver_py.reset()

            start = time.perf_counter()
            for _ in range(n_steps):
                solver_py.step()
            python_time = time.perf_counter() - start

            # Native backend
            solver_native = FDTDSolver(
                shape=(size, size, size), resolution=1e-3, backend="native"
            )
            solver_native.add_source(
                GaussianPulse(position=(size // 4, size // 2, size // 2), frequency=1000)
            )

            # Warmup
            for _ in range(3):
                solver_native.step()
            solver_native.reset()

            start = time.perf_counter()
            for _ in range(n_steps):
                solver_native.step()
            native_time = time.perf_counter() - start

            speedup = python_time / native_time
            results.append(
                {
                    "size": size,
                    "python_time": python_time,
                    "native_time": native_time,
                    "speedup": speedup,
                }
            )

        # Print report
        print("\n\nBackend Speedup Comparison")
        print("=" * 60)
        print(f"{'Grid':<10} {'Python (ms)':<15} {'Native (ms)':<15} {'Speedup':<10}")
        print("-" * 60)

        for r in results:
            print(
                f"{r['size']}³{'':<6} {r['python_time']*1000:<15.2f} "
                f"{r['native_time']*1000:<15.2f} {r['speedup']:<10.2f}x"
            )

        print("=" * 60)

        # Verify we get at least 2x speedup on 100³ grid
        for r in results:
            if r["size"] >= 100:
                assert r["speedup"] >= 2.0, (
                    f"Expected at least 2x speedup on {r['size']}³ grid, "
                    f"got {r['speedup']:.2f}x"
                )
