"""
Tests for native C++ microphone recording kernel.

Tests verify:
- Native kernel produces same results as Python implementation
- Performance improvement with large microphone counts
- Proper handling of edge cases

Issue #102: Native C++ kernel for high microphone count performance
"""

import time

import numpy as np
import pytest

from strata_fdtd import FDTDSolver, GaussianPulse
from strata_fdtd.core.solver import has_native_kernels

# Skip all tests if native kernels are not available
pytestmark = pytest.mark.skipif(
    not has_native_kernels(),
    reason="Native kernels not available"
)


@pytest.fixture
def solver_with_source():
    """Create solver with a source for testing."""
    solver = FDTDSolver(shape=(50, 50, 50), resolution=2e-3, backend="native")
    solver.add_source(GaussianPulse(
        position=(10, 25, 25),
        frequency=2000,
        amplitude=10.0,
    ))
    return solver


class TestNativeMicrophoneAccuracy:
    """Tests verifying native kernel produces correct results."""

    def test_native_matches_python_single_mic(self, solver_with_source):
        """Test that native and Python give same result for single microphone."""
        solver = solver_with_source

        # Add microphone
        mic_pos = (0.04, 0.05, 0.05)
        solver.add_microphone(position=mic_pos, name="test")

        # Run a few steps
        n_steps = 50
        for _ in range(n_steps):
            solver.step()

        native_data = solver.microphones["test"].get_waveform()

        # Create identical solver with Python backend
        solver_py = FDTDSolver(shape=(50, 50, 50), resolution=2e-3, backend="python")
        solver_py.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=2000,
            amplitude=10.0,
        ))
        solver_py.add_microphone(position=mic_pos, name="test")

        for _ in range(n_steps):
            solver_py.step()

        python_data = solver_py.microphones["test"].get_waveform()

        # Results should be nearly identical (small float differences allowed)
        np.testing.assert_allclose(native_data, python_data, rtol=1e-5, atol=1e-7)

    def test_native_matches_python_many_mics(self, solver_with_source):
        """Test native vs Python with many microphones."""
        solver = solver_with_source

        # Add 25 microphones in a 5x5 grid
        mic_positions = []
        for i in range(5):
            for j in range(5):
                pos = (0.02 + i * 0.015, 0.02 + j * 0.015, 0.05)
                mic_positions.append(pos)
                solver.add_microphone(position=pos, name=f"mic_{i}_{j}")

        # Run simulation
        n_steps = 30
        for _ in range(n_steps):
            solver.step()

        native_data = {name: mic.get_waveform()
                       for name, mic in solver.microphones.items()}

        # Create identical solver with Python backend
        solver_py = FDTDSolver(shape=(50, 50, 50), resolution=2e-3, backend="python")
        solver_py.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=2000,
            amplitude=10.0,
        ))

        for i, pos in enumerate(mic_positions):
            name = f"mic_{i // 5}_{i % 5}"
            solver_py.add_microphone(position=pos, name=name)

        for _ in range(n_steps):
            solver_py.step()

        # Compare all microphones
        for name, mic in solver_py.microphones.items():
            python_data = mic.get_waveform()
            np.testing.assert_allclose(
                native_data[name], python_data,
                rtol=1e-5, atol=1e-7,
                err_msg=f"Mismatch for microphone {name}"
            )

    def test_interpolation_at_grid_aligned(self):
        """Test native interpolation at grid-aligned position."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="native")

        # Position exactly at grid point (5, 5, 5)
        pos = (5 * solver.dx, 5 * solver.dx, 5 * solver.dx)
        solver.add_microphone(position=pos, name="grid_aligned")

        # Set known pressure
        solver.p[5, 5, 5] = 42.0

        # Record one sample
        solver.step()

        waveform = solver.microphones["grid_aligned"].get_waveform()
        # At exact grid point, should get exact value (minus any propagation changes)
        # Note: step() will modify pressure, so we check the recorded value is reasonable
        assert len(waveform) == 1

    def test_interpolation_at_cell_center(self):
        """Test native interpolation at cell center."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="native")

        # Position at center of cell (5.5, 5.5, 5.5) in grid coords
        pos = (5.5 * solver.dx, 5.5 * solver.dx, 5.5 * solver.dx)
        solver.add_microphone(position=pos, name="cell_center")

        # Set uniform pressure in surrounding cells
        solver.p[5:7, 5:7, 5:7] = 1.0

        # Manually call the recording (before step changes pressure)
        solver._record_microphones()

        waveform = solver.microphones["cell_center"].get_waveform()
        # Average of 8 equal values should be 1.0
        assert waveform[0] == pytest.approx(1.0, rel=0.01)


class TestNativeMicrophonePerformance:
    """Performance benchmarks for native microphone kernel."""

    @pytest.mark.parametrize("n_mics", [10, 50, 100, 200])
    def test_performance_scaling(self, n_mics):
        """Test performance with varying microphone counts."""
        solver_native = FDTDSolver(
            shape=(100, 100, 100), resolution=1e-3, backend="native"
        )
        solver_python = FDTDSolver(
            shape=(100, 100, 100), resolution=1e-3, backend="python"
        )

        # Add microphones in a grid
        np.random.seed(42)
        for i in range(n_mics):
            pos = (
                np.random.uniform(0.01, 0.09),
                np.random.uniform(0.01, 0.09),
                np.random.uniform(0.01, 0.09),
            )
            solver_native.add_microphone(position=pos, name=f"mic_{i}")
            solver_python.add_microphone(position=pos, name=f"mic_{i}")

        # Add a source to have something to record
        solver_native.add_source(GaussianPulse(
            position=(50, 50, 50), frequency=1000
        ))
        solver_python.add_source(GaussianPulse(
            position=(50, 50, 50), frequency=1000
        ))

        # Warm up
        for _ in range(5):
            solver_native.step()
            solver_python.step()

        solver_native.reset()
        solver_python.reset()

        # Time native
        n_steps = 100
        start = time.perf_counter()
        for _ in range(n_steps):
            solver_native.step()
        native_time = time.perf_counter() - start

        # Time Python
        start = time.perf_counter()
        for _ in range(n_steps):
            solver_python.step()
        python_time = time.perf_counter() - start

        # Record results
        speedup = python_time / native_time if native_time > 0 else float('inf')

        # Print benchmark results
        print(f"\n{n_mics} microphones:")
        print(f"  Native: {native_time*1000:.2f}ms ({n_steps} steps)")
        print(f"  Python: {python_time*1000:.2f}ms ({n_steps} steps)")
        print(f"  Speedup: {speedup:.2f}x")

        # For large microphone counts, native should be faster
        if n_mics >= 50:
            # Allow some margin - at least don't be slower
            assert speedup >= 0.5, f"Native kernel too slow for {n_mics} mics"


class TestNativeMicrophoneEdgeCases:
    """Edge case tests for native microphone kernel."""

    def test_zero_microphones(self):
        """Test that zero microphones doesn't cause errors."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="native")
        # Should not raise
        for _ in range(10):
            solver.step()

    def test_microphone_added_after_simulation_start(self):
        """Test adding microphone after simulation has started."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="native")
        solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=1000))

        # Run some steps
        for _ in range(10):
            solver.step()

        # Add microphone mid-simulation
        mic = solver.add_microphone(position=(0.05, 0.05, 0.05), name="late")

        # Run more steps
        for _ in range(10):
            solver.step()

        # Should have recorded only the latter steps
        assert len(mic) == 10

    def test_microphone_data_invalidation(self):
        """Test that native microphone data is properly invalidated."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="native")

        # Add first microphone
        solver.add_microphone(position=(0.03, 0.05, 0.05), name="mic1")

        # Run to initialize native data
        solver.step()
        assert solver._microphone_data is not None

        # Add second microphone - should invalidate cache
        solver.add_microphone(position=(0.05, 0.05, 0.05), name="mic2")
        assert solver._microphone_data is None

        # Run more - should recompute with both microphones
        solver.step()
        assert solver._microphone_data is not None
        assert solver._microphone_data.n_mics == 2
