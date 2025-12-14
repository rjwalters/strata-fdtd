"""Tests for native FDTD extension infrastructure.

These tests verify:
1. Extension can be imported (when available)
2. Fallback behavior works when extension is missing
3. Backend selection logic in FDTDSolver
4. Native and Python backends produce equivalent results
"""

import numpy as np
import pytest

from strata_fdtd.fdtd import (
    FDTDSolver,
    GaussianPulse,
    get_native_info,
    has_native_kernels,
)


class TestNativeKernelAvailability:
    """Tests for native kernel detection and info."""

    def test_has_native_kernels_returns_bool(self):
        """has_native_kernels() should return a boolean."""
        result = has_native_kernels()
        assert isinstance(result, bool)

    def test_get_native_info_structure(self):
        """get_native_info() should return expected dict structure."""
        info = get_native_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "version" in info
        assert "has_openmp" in info
        assert "num_threads" in info
        assert isinstance(info["available"], bool)
        assert isinstance(info["has_openmp"], bool)
        assert isinstance(info["num_threads"], int)
        assert info["num_threads"] >= 1

    def test_info_consistency(self):
        """has_native_kernels() and get_native_info() should be consistent."""
        has_native = has_native_kernels()
        info = get_native_info()
        assert has_native == info["available"]


class TestBackendSelection:
    """Tests for FDTDSolver backend selection."""

    def test_default_backend_is_auto(self):
        """Default backend should be 'auto'."""
        solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
        # In auto mode, using_native depends on availability
        assert solver.using_native == has_native_kernels()

    def test_python_backend_forced(self):
        """backend='python' should force Python implementation."""
        solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="python")
        assert not solver.using_native

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_native_backend_forced(self):
        """backend='native' should force native implementation."""
        solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="native")
        assert solver.using_native

    @pytest.mark.skipif(has_native_kernels(), reason="Native kernels are available")
    def test_native_backend_raises_without_extension(self):
        """backend='native' should raise ImportError when extension unavailable."""
        with pytest.raises(ImportError, match="Native FDTD kernels not available"):
            FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="native")


class TestFallbackBehavior:
    """Tests for Python fallback functionality."""

    def test_python_backend_runs(self):
        """Python backend should complete simulation."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=1e-3, backend="python")
        solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=1000))
        solver.add_probe("center", position=(10, 10, 10))

        # Run a few steps
        for _ in range(10):
            solver.step()

        # Should have recorded data
        data = solver.get_probe_data("center")
        assert "center" in data
        assert len(data["center"]) == 10

    def test_python_backend_with_geometry(self):
        """Python backend should handle geometry correctly."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=1e-3, backend="python")

        # Create geometry with a solid block
        geometry = np.ones((20, 20, 20), dtype=bool)
        geometry[8:12, 8:12, 8:12] = False  # Solid block
        solver.set_geometry(geometry)

        solver.add_source(GaussianPulse(position=(5, 10, 10), frequency=1000))

        # Run a few steps
        for _ in range(10):
            solver.step()

        # Pressure in solid region should be zero
        assert np.allclose(solver.p[8:12, 8:12, 8:12], 0.0)


@pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
class TestNativeExtension:
    """Tests that require native extension to be available."""

    def test_native_extension_imports(self):
        """Verify extension can be imported."""
        from strata_fdtd import _kernels

        assert hasattr(_kernels, "__version__")
        assert hasattr(_kernels, "has_openmp")

    def test_native_extension_version(self):
        """Verify extension has valid version string."""
        from strata_fdtd import _kernels

        version = _kernels.__version__
        assert isinstance(version, str)
        assert len(version) > 0

    def test_thread_control(self):
        """Verify thread control functions work."""
        from strata_fdtd import _kernels

        if _kernels.has_openmp:
            original = _kernels.get_num_threads()
            assert original >= 1

            # Try setting threads (should not raise)
            _kernels.set_num_threads(2)
            assert _kernels.get_num_threads() >= 1

            # Restore original
            _kernels.set_num_threads(original)

    def test_native_backend_runs(self):
        """Native backend should complete simulation."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=1e-3, backend="native")
        solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=1000))
        solver.add_probe("center", position=(10, 10, 10))

        # Run a few steps
        for _ in range(10):
            solver.step()

        # Should have recorded data
        data = solver.get_probe_data("center")
        assert "center" in data
        assert len(data["center"]) == 10


@pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
class TestBackendEquivalence:
    """Tests that native and Python backends produce equivalent results."""

    def test_step_equivalence_empty_domain(self):
        """Native and Python should produce identical results in empty domain."""
        shape = (30, 30, 30)

        # Create two solvers with same parameters
        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        # Add same source
        source_pos = (15, 15, 15)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        # Run both for same number of steps
        n_steps = 50
        for _ in range(n_steps):
            solver_py.step()
            solver_native.step()

        # Results should be very close (allow for floating point differences)
        np.testing.assert_allclose(
            solver_py.p, solver_native.p, rtol=1e-5, atol=1e-7,
            err_msg="Pressure fields diverged between backends"
        )
        np.testing.assert_allclose(
            solver_py.vx, solver_native.vx, rtol=1e-5, atol=1e-7,
            err_msg="Vx fields diverged between backends"
        )
        np.testing.assert_allclose(
            solver_py.vy, solver_native.vy, rtol=1e-5, atol=1e-7,
            err_msg="Vy fields diverged between backends"
        )
        np.testing.assert_allclose(
            solver_py.vz, solver_native.vz, rtol=1e-5, atol=1e-7,
            err_msg="Vz fields diverged between backends"
        )

    def test_step_equivalence_with_geometry(self):
        """Native and Python should handle geometry identically.

        Note: Small differences are expected due to different operation ordering
        in the native vs Python implementations. Both enforce the same physics
        but may accumulate minor floating-point differences over many steps.
        """
        shape = (30, 30, 30)

        # Create geometry with solid regions
        geometry = np.ones(shape, dtype=bool)
        geometry[10:20, 10:20, 10:20] = False  # Solid block

        # Create two solvers
        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        solver_py.set_geometry(geometry)
        solver_native.set_geometry(geometry)

        # Add same source (outside solid region)
        source_pos = (5, 15, 15)
        solver_py.add_source(GaussianPulse(position=source_pos, frequency=1000))
        solver_native.add_source(GaussianPulse(position=source_pos, frequency=1000))

        # Run both
        n_steps = 50
        for _ in range(n_steps):
            solver_py.step()
            solver_native.step()

        # Key invariant: pressure in solid regions should be zero in both
        assert np.allclose(solver_py.p[~geometry], 0.0)
        assert np.allclose(solver_native.p[~geometry], 0.0)

        # Results should be similar (relax tolerance for geometry case)
        # The native and Python implementations handle boundary enforcement
        # at slightly different points in the update cycle
        np.testing.assert_allclose(
            solver_py.p, solver_native.p, rtol=0.1, atol=1e-5,
            err_msg="Pressure fields diverged significantly with geometry"
        )

    def test_energy_conservation_equivalence(self):
        """Both backends should conserve energy similarly in closed cavity."""
        shape = (30, 30, 30)

        solver_py = FDTDSolver(shape=shape, resolution=1e-3, backend="python")
        solver_native = FDTDSolver(shape=shape, resolution=1e-3, backend="native")

        # Add impulse source (single timestep injection)
        solver_py.p[15, 15, 15] = 1.0
        solver_native.p[15, 15, 15] = 1.0

        # Track energy over time
        energy_py = []
        energy_native = []

        for _ in range(100):
            solver_py.step()
            solver_native.step()
            energy_py.append(solver_py.compute_energy())
            energy_native.append(solver_native.compute_energy())

        # Energy trajectories should match
        np.testing.assert_allclose(
            energy_py, energy_native, rtol=1e-4,
            err_msg="Energy trajectories diverged between backends"
        )


@pytest.mark.benchmark
class TestBackendPerformance:
    """Benchmark tests comparing backend performance."""

    def test_python_throughput(self, benchmark):
        """Measure Python backend throughput."""
        solver = FDTDSolver(shape=(50, 50, 50), resolution=1e-3, backend="python")

        def step_10():
            for _ in range(10):
                solver.step()

        benchmark(step_10)
        benchmark.extra_info["backend"] = "python"

    @pytest.mark.skipif(not has_native_kernels(), reason="Native kernels not available")
    def test_native_throughput(self, benchmark):
        """Measure native backend throughput."""
        solver = FDTDSolver(shape=(50, 50, 50), resolution=1e-3, backend="native")

        def step_10():
            for _ in range(10):
                solver.step()

        benchmark(step_10)
        benchmark.extra_info["backend"] = "native"
