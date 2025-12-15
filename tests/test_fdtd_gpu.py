"""
Unit tests for GPU-accelerated FDTD simulation.

Tests verify:
- GPU availability detection
- Single-band solver correctness
- Batched multi-band solver functionality
- Wave propagation on GPU matches physics
"""

import numpy as np
import pytest

from strata_fdtd.core.solver_gpu import (
    BatchedGPUFDTDSolver,
    GPUFDTDSolver,
    GPUGaussianPulse,
    get_gpu_info,
    has_gpu_support,
)

# Skip all tests if PyTorch is not available
pytest.importorskip("torch")

# Mark all tests to skip if MPS is not available
pytestmark = pytest.mark.skipif(
    not has_gpu_support(),
    reason="GPU (MPS) support not available"
)


# =============================================================================
# GPU Availability Tests
# =============================================================================


class TestGPUAvailability:
    """Tests for GPU availability detection."""

    def test_has_gpu_support_returns_bool(self):
        """has_gpu_support() should return a boolean."""
        result = has_gpu_support()
        assert isinstance(result, bool)

    def test_get_gpu_info_returns_dict(self):
        """get_gpu_info() should return a dictionary with expected keys."""
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "backend" in info
        assert "pytorch_version" in info

    def test_gpu_info_consistency(self):
        """GPU info should be consistent with has_gpu_support()."""
        info = get_gpu_info()
        assert info["available"] == has_gpu_support()
        if info["available"]:
            assert info["backend"] == "mps"
            assert info["pytorch_version"] is not None


# =============================================================================
# GPUGaussianPulse Tests
# =============================================================================


class TestGPUGaussianPulse:
    """Tests for GPU Gaussian pulse source."""

    def test_default_bandwidth(self):
        """Default bandwidth should be 2x frequency."""
        source = GPUGaussianPulse(position=(10, 10, 10), frequency=1000)
        assert source.bandwidth == 2000

    def test_custom_bandwidth(self):
        """Custom bandwidth should be respected."""
        source = GPUGaussianPulse(
            position=(10, 10, 10), frequency=1000, bandwidth=500
        )
        assert source.bandwidth == 500

    def test_evaluate_returns_float(self):
        """Pulse evaluation should return a float."""
        source = GPUGaussianPulse(position=(10, 10, 10), frequency=1000)
        value = source.evaluate(0.001)
        assert isinstance(value, (int, float))

    def test_pulse_is_zero_at_early_times(self):
        """Pulse should be near zero before t_peak."""
        source = GPUGaussianPulse(position=(10, 10, 10), frequency=1000)
        # Well before t_peak (4*sigma_t), amplitude should be tiny
        value = source.evaluate(0.0)
        assert abs(value) < 0.01

    def test_pulse_peaks_near_t_peak(self):
        """Pulse should have significant amplitude near t_peak."""
        source = GPUGaussianPulse(
            position=(10, 10, 10), frequency=1000, bandwidth=500
        )
        # t_peak = 4 * sigma_t = 4 / (2*pi*500) ≈ 1.27ms
        t_peak = 4.0 / (2.0 * np.pi * 500)
        # At t_peak, envelope is 1.0, carrier is sin(2*pi*f*0) = 0
        # Peak amplitude happens slightly after
        max_val = max(abs(source.evaluate(t_peak + dt))
                      for dt in np.linspace(-0.0005, 0.0005, 100))
        assert max_val > 0.5


# =============================================================================
# GPUFDTDSolver Tests
# =============================================================================


class TestGPUFDTDSolver:
    """Tests for single-band GPU FDTD solver."""

    @pytest.fixture
    def small_gpu_solver(self):
        """Create a small GPU solver for fast tests."""
        return GPUFDTDSolver(shape=(20, 20, 20), resolution=5e-3)

    def test_solver_creation(self, small_gpu_solver):
        """Solver should initialize without errors."""
        assert small_gpu_solver.shape == (20, 20, 20)
        assert small_gpu_solver.resolution == 5e-3

    def test_using_gpu_property(self, small_gpu_solver):
        """Solver should report GPU usage."""
        if has_gpu_support():
            assert small_gpu_solver.using_gpu is True
            assert small_gpu_solver.device == "mps"

    def test_timestep_satisfies_cfl(self, small_gpu_solver):
        """Timestep should satisfy CFL condition."""
        dt_max = small_gpu_solver.resolution / (
            small_gpu_solver.c * np.sqrt(3)
        )
        assert small_gpu_solver.dt <= dt_max

    def test_add_source(self, small_gpu_solver):
        """Adding a source should work."""
        source = small_gpu_solver.add_source(
            position=(5, 10, 10), frequency=1000, bandwidth=500
        )
        assert source is not None
        assert source.position == (5, 10, 10)
        assert source.frequency == 1000

    def test_add_source_default_position(self, small_gpu_solver):
        """Source without position should default to center."""
        source = small_gpu_solver.add_source(frequency=1000)
        assert source.position == (10, 10, 10)

    def test_add_probe(self, small_gpu_solver):
        """Adding a probe should work."""
        small_gpu_solver.add_probe("test", position=(15, 10, 10))
        assert "test" in small_gpu_solver._probes

    def test_run_simulation(self, small_gpu_solver):
        """Running simulation should complete without errors."""
        small_gpu_solver.add_source(frequency=1000)
        small_gpu_solver.add_probe("test", position=(15, 10, 10))
        small_gpu_solver.run(steps=10)
        assert small_gpu_solver.step_count == 10

    def test_get_probe_data(self, small_gpu_solver):
        """Probe data should be retrievable."""
        small_gpu_solver.add_source(frequency=1000)
        small_gpu_solver.add_probe("test", position=(15, 10, 10))
        small_gpu_solver.run(steps=10)

        data = small_gpu_solver.get_probe_data()
        assert "test" in data
        assert len(data["test"]) == 10
        assert data["test"].dtype == np.float32

    def test_get_pressure_field(self, small_gpu_solver):
        """Pressure field should be retrievable as numpy array."""
        small_gpu_solver.run(steps=1)
        p = small_gpu_solver.get_pressure_field()

        assert isinstance(p, np.ndarray)
        assert p.shape == small_gpu_solver.shape
        assert p.dtype == np.float32

    def test_reset(self, small_gpu_solver):
        """Reset should clear simulation state."""
        small_gpu_solver.add_source(frequency=1000)
        small_gpu_solver.add_probe("test", position=(15, 10, 10))
        small_gpu_solver.run(steps=10)

        small_gpu_solver.reset()

        assert small_gpu_solver.step_count == 0
        assert len(small_gpu_solver._probe_data["test"]) == 0

    def test_cpu_fallback(self):
        """Solver should work with CPU device."""
        solver = GPUFDTDSolver(
            shape=(10, 10, 10), resolution=5e-3, device="cpu"
        )
        solver.add_source(frequency=1000)
        solver.run(steps=5)

        assert solver.device == "cpu"
        assert solver.using_gpu is False
        assert solver.step_count == 5


# =============================================================================
# BatchedGPUFDTDSolver Tests
# =============================================================================


class TestBatchedGPUFDTDSolver:
    """Tests for batched multi-band GPU FDTD solver."""

    @pytest.fixture
    def batched_solver(self):
        """Create a batched solver with 4 frequency bands."""
        return BatchedGPUFDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            bands=[(250, 100), (500, 200), (1000, 400), (2000, 800)],
        )

    def test_solver_creation(self, batched_solver):
        """Batched solver should initialize correctly."""
        assert batched_solver.shape == (30, 30, 30)
        assert batched_solver.n_bands == 4
        assert len(batched_solver.bands) == 4

    def test_bands_stored_correctly(self, batched_solver):
        """Band specifications should be stored."""
        expected_bands = [(250, 100), (500, 200), (1000, 400), (2000, 800)]
        assert batched_solver.bands == expected_bands

    def test_memory_usage_calculation(self, batched_solver):
        """Memory usage should be calculated correctly."""
        # 4 fields × 4 bands × 30³ × 4 bytes = 4 × 4 × 27000 × 4 = 1.728 MB
        expected_mb = 4 * 4 * 30 * 30 * 30 * 4 / (1024 * 1024)
        assert batched_solver.memory_usage_mb() == pytest.approx(
            expected_mb, rel=0.01
        )

    def test_run_returns_dict(self, batched_solver):
        """Run should return a dictionary of results."""
        results = batched_solver.run(steps=10)

        assert isinstance(results, dict)
        assert len(results) == 4
        assert "250Hz" in results
        assert "500Hz" in results
        assert "1000Hz" in results
        assert "2000Hz" in results

    def test_result_arrays_have_correct_length(self, batched_solver):
        """Result arrays should have correct number of samples."""
        results = batched_solver.run(steps=20)

        for _name, data in results.items():
            assert len(data) == 20
            assert data.dtype == np.float32

    def test_different_bands_produce_different_results(self, batched_solver):
        """Different frequency bands should produce different waveforms."""
        results = batched_solver.run(steps=100)

        # After enough steps, different frequencies should differ
        data_250 = results["250Hz"]
        data_2000 = results["2000Hz"]

        # They shouldn't be identical (different source frequencies)
        # Check after signals have developed (skip early zeros)
        if np.any(data_250 != 0) and np.any(data_2000 != 0):
            correlation = np.corrcoef(data_250, data_2000)[0, 1]
            # Low correlation expected for different frequencies
            assert abs(correlation) < 0.9

    def test_get_pressure_fields(self, batched_solver):
        """Pressure fields for all bands should be retrievable."""
        batched_solver.run(steps=10)
        fields = batched_solver.get_pressure_fields()

        assert isinstance(fields, dict)
        assert len(fields) == 4
        for _name, field in fields.items():
            assert field.shape == batched_solver.shape
            assert field.dtype == np.float32

    def test_reset_clears_state(self, batched_solver):
        """Reset should clear all band states."""
        batched_solver.run(steps=10)
        batched_solver.reset()

        assert batched_solver.step_count == 0
        results = batched_solver.get_probe_data()
        for data in results.values():
            assert len(data) == 0

    def test_empty_bands_raises_error(self):
        """Creating solver with no bands should raise ValueError."""
        with pytest.raises(ValueError, match="at least one frequency band"):
            BatchedGPUFDTDSolver(
                shape=(20, 20, 20),
                resolution=3e-3,
                bands=[],
            )

    def test_custom_source_position(self):
        """Custom source position should be used."""
        solver = BatchedGPUFDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            bands=[(500, 200)],
            source_position=(5, 15, 15),
        )
        assert solver.source_position == (5, 15, 15)

    def test_custom_probe_position(self):
        """Custom probe position should be used."""
        solver = BatchedGPUFDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            bands=[(500, 200)],
            probe_position=(25, 15, 15),
        )
        assert solver.probe_position == (25, 15, 15)


# =============================================================================
# Integration Tests
# =============================================================================


class TestGPUCPUConsistency:
    """Tests verifying GPU results are physically reasonable."""

    def test_wave_propagates(self):
        """Wave should propagate from source to probe."""
        solver = GPUFDTDSolver(shape=(60, 30, 30), resolution=2e-3)
        solver.add_source(position=(10, 15, 15), frequency=500, bandwidth=200)
        solver.add_probe("far", position=(50, 15, 15))

        # Run for enough time for wave to propagate
        # Distance: 40 cells × 2mm = 80mm
        # Time: 80mm / 343m/s ≈ 0.23ms
        # Plus pulse rise time ≈ 1ms
        solver.run(duration=0.002)

        data = solver.get_probe_data()["far"]

        # Should see non-zero signal after wave arrives
        assert np.max(np.abs(data)) > 0

    def test_batched_all_bands_produce_signal(self):
        """All frequency bands should produce detectable signals."""
        solver = BatchedGPUFDTDSolver(
            shape=(50, 30, 30),
            resolution=2e-3,
            bands=[(250, 100), (500, 200), (1000, 400)],
        )
        results = solver.run(duration=0.003)

        for name, data in results.items():
            # Each band should produce some signal
            assert np.max(np.abs(data)) > 0, f"Band {name} produced no signal"

    def test_energy_conservation_closed_cavity(self):
        """Energy should be approximately conserved in closed cavity."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=3e-3)

        # Initialize with pressure pulse at center
        solver._p[15, 15, 15] = 1.0

        # Record initial energy
        p_np = solver._p.cpu().numpy()
        initial_energy = np.sum(p_np**2)

        # Run for many steps
        solver.run(steps=500)

        # Check final energy
        p_np = solver._p.cpu().numpy()
        final_energy = np.sum(p_np**2)

        # Energy should be conserved within some tolerance
        # (numerical dispersion causes some drift)
        assert final_energy < 2 * initial_energy, "Energy grew too much"
        assert final_energy > 0.1 * initial_energy, "Energy decayed too much"


# =============================================================================
# Geometry Tests
# =============================================================================


class TestGeometry:
    """Tests for geometry mask support."""

    @pytest.fixture
    def solver_with_geometry(self):
        """Create solver with a geometry mask."""
        solver = GPUFDTDSolver(shape=(40, 20, 20), resolution=2e-3)
        return solver

    def test_set_geometry(self, solver_with_geometry):
        """Setting geometry mask should work."""
        mask = np.ones((40, 20, 20), dtype=bool)
        mask[20:, :, :] = False  # Solid wall at x=20
        solver_with_geometry.set_geometry(mask)

        # Should complete without error
        assert True

    def test_set_geometry_wrong_shape_raises_error(self, solver_with_geometry):
        """Geometry mask with wrong shape should raise ValueError."""
        mask = np.ones((30, 20, 20), dtype=bool)

        with pytest.raises(ValueError, match="doesn't match grid shape"):
            solver_with_geometry.set_geometry(mask)

    def test_geometry_blocks_propagation(self):
        """Solid geometry should block wave propagation."""
        solver = GPUFDTDSolver(shape=(60, 20, 20), resolution=2e-3)

        # Create wall at x=30
        mask = np.ones((60, 20, 20), dtype=bool)
        mask[30:, :, :] = False  # Solid from x=30 onwards
        solver.set_geometry(mask)

        # Source before wall, probe after wall
        solver.add_source(position=(10, 10, 10), frequency=1000, bandwidth=500)
        solver.add_probe("before_wall", position=(20, 10, 10))
        solver.add_probe("after_wall", position=(45, 10, 10))

        # Run simulation
        solver.run(duration=0.003)

        data = solver.get_probe_data()

        # Probe before wall should detect signal
        assert np.max(np.abs(data["before_wall"])) > 0.01

        # Probe after wall should detect very little (blocked)
        # Some numerical leakage is acceptable
        assert np.max(np.abs(data["after_wall"])) < 0.1 * np.max(
            np.abs(data["before_wall"])
        )

    def test_pressure_zero_in_solid_cells(self):
        """Pressure should remain zero in solid cells."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Make half the domain solid
        mask = np.ones((30, 30, 30), dtype=bool)
        mask[15:, :, :] = False
        solver.set_geometry(mask)

        # Add source in air region
        solver.add_source(position=(5, 15, 15), frequency=1000)

        # Run simulation
        solver.run(duration=0.002)

        # Get pressure field
        p = solver.get_pressure_field()

        # Solid region should have near-zero pressure
        solid_pressure = p[15:, :, :]
        assert np.max(np.abs(solid_pressure)) < 1e-6

    def test_batched_solver_geometry(self):
        """Batched solver should support geometry mask."""
        solver = BatchedGPUFDTDSolver(
            shape=(50, 20, 20),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
        )

        # Create wall
        mask = np.ones((50, 20, 20), dtype=bool)
        mask[25:, :, :] = False
        solver.set_geometry(mask)

        # Run simulation
        results = solver.run(duration=0.002)

        # Check that all bands ran successfully
        assert len(results) == 2
        for data in results.values():
            assert len(data) > 0

    def test_rigid_boundary_reflection(self):
        """Wave should reflect cleanly from rigid wall with rigid boundary conditions."""
        solver = GPUFDTDSolver(shape=(100, 30, 30), resolution=1e-3)

        # Solid wall at x=70
        mask = np.ones((100, 30, 30), dtype=bool)
        mask[70:, :, :] = False
        solver.set_geometry(mask)

        # Source before wall
        solver.add_source(position=(30, 15, 15), frequency=1000, bandwidth=500)

        # Probe near wall to observe reflection
        solver.add_probe("near_wall", position=(65, 15, 15))

        solver.run(duration=0.01)
        data = solver.get_probe_data()["near_wall"]

        # Should see interference pattern from incident + reflected waves
        # Check for presence of both positive and negative peaks (standing wave pattern)
        positive_peaks = np.sum(data > 0.3 * np.max(data))
        negative_peaks = np.sum(data < 0.3 * np.min(data))

        assert positive_peaks > 5, "Should have multiple positive peaks from interference"
        assert negative_peaks > 5, "Should have multiple negative peaks from interference"

        # Reflection should create higher amplitude than free-space propagation
        # (constructive interference near wall)
        max_amplitude = np.max(np.abs(data))
        assert max_amplitude > 0.5, "Reflected wave should have significant amplitude"

    def test_rigid_boundary_zero_velocity_at_wall(self):
        """Velocity normal to wall should be zero at rigid boundaries."""
        solver = GPUFDTDSolver(shape=(50, 30, 30), resolution=1e-3)

        # Solid wall at x=30
        mask = np.ones((50, 30, 30), dtype=bool)
        mask[30:, :, :] = False
        solver.set_geometry(mask)

        # Add source and run
        solver.add_source(position=(10, 15, 15), frequency=1000)
        solver.run(steps=50)

        # Get velocity field
        vx = solver._vx.cpu().numpy()

        # Velocity at face between x=29 (air) and x=30 (solid) should be zero
        # vx[29, :, :] is the velocity at this interface
        interface_vx = vx[29, :, :]
        assert np.max(np.abs(interface_vx)) < 1e-6, "Normal velocity at rigid wall should be zero"

    def test_closed_cavity_standing_waves(self):
        """Closed cavity with rigid boundaries should support standing wave modes."""
        # Small cavity for faster testing
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3, pml_layers=0)

        # Create closed cavity (solid walls on all sides except small air region)
        mask = np.ones((30, 30, 30), dtype=bool)
        # Set boundaries to solid
        mask[0:2, :, :] = False
        mask[-2:, :, :] = False
        mask[:, 0:2, :] = False
        mask[:, -2:, :] = False
        mask[:, :, 0:2] = False
        mask[:, :, -2:] = False
        solver.set_geometry(mask)

        # Excite cavity mode
        solver.add_source(position=(15, 15, 15), frequency=1000, bandwidth=300)
        solver.add_probe("center", position=(15, 15, 15))

        solver.run(duration=0.015)
        data = solver.get_probe_data()["center"]

        # In a closed cavity, energy should be sustained longer than free-space
        # Check that oscillations persist in latter half of simulation
        latter_half = data[len(data)//2:]
        latter_half_energy = np.mean(latter_half**2)
        total_energy = np.mean(data**2)

        # With rigid boundaries, energy should persist
        # (without rigid boundaries, wave would leak through walls)
        assert latter_half_energy > 0.3 * total_energy, \
            "Standing wave energy should persist in closed cavity"

        # Should have clear oscillations in the latter half
        peaks_in_latter_half = len(np.where(np.diff(np.sign(latter_half)))[0])
        assert peaks_in_latter_half > 10, "Should have sustained oscillations in closed cavity"

    def test_batched_solver_rigid_boundaries(self):
        """Batched solver should enforce rigid boundaries across all bands."""
        solver = BatchedGPUFDTDSolver(
            shape=(60, 30, 30),
            resolution=1e-3,
            bands=[(500, 200), (1000, 400)],
        )

        # Solid wall at x=40
        mask = np.ones((60, 30, 30), dtype=bool)
        mask[40:, :, :] = False
        solver.set_geometry(mask)

        # Run simulation
        solver.run(duration=0.008)

        # Get velocity fields for all bands
        vx = solver._vx.cpu().numpy()

        # For both bands, velocity at wall interface should be zero
        # vx[:, 39, :, :] is velocity at interface between x=39 (air) and x=40 (solid)
        for band_idx in range(solver.n_bands):
            interface_vx = vx[band_idx, 39, :, :]
            assert np.max(np.abs(interface_vx)) < 1e-6, \
                f"Band {band_idx}: Normal velocity at rigid wall should be zero"


# =============================================================================
# Material Tests
# =============================================================================


class TestMaterials:
    """Tests for frequency-dependent material support."""

    @pytest.fixture
    def debye_material(self):
        """Create a simple Debye material."""
        from strata_fdtd.materials.base import Pole, PoleType, SimpleMaterial

        # Create a material with one Debye pole for density
        class TestMaterial(SimpleMaterial):
            def __init__(self):
                super().__init__(name="test_debye", _rho=1.2, _c=343.0)
                self._poles = [
                    Pole(
                        pole_type=PoleType.DEBYE,
                        delta_chi=0.5,
                        tau=1e-4,
                        target="density",
                    )
                ]

            @property
            def poles(self):
                return self._poles

        return TestMaterial()

    def test_add_material(self, debye_material):
        """Adding a material should work."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Create region mask
        region_mask = np.zeros((30, 30, 30), dtype=bool)
        region_mask[10:20, 10:20, 10:20] = True

        # Add material
        mat_id = solver.add_material(debye_material, region_mask)

        assert mat_id == 1
        assert solver.has_materials is True

    def test_add_material_wrong_shape_raises_error(self, debye_material):
        """Material with wrong region mask shape should raise ValueError."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Wrong shape mask
        region_mask = np.zeros((20, 20, 20), dtype=bool)

        with pytest.raises(ValueError, match="doesn't match grid shape"):
            solver.add_material(debye_material, region_mask)

    @pytest.fixture
    def lorentz_material(self):
        """Create a material with a Lorentz pole."""
        from strata_fdtd.materials.base import Pole, PoleType, SimpleMaterial

        class LorentzMaterial(SimpleMaterial):
            def __init__(self):
                super().__init__(name="test_lorentz", _rho=1.2, _c=343.0)
                self._poles = [
                    Pole(
                        pole_type=PoleType.LORENTZ,
                        delta_chi=0.5,
                        omega_0=2 * np.pi * 1000,  # 1 kHz resonance
                        gamma=100,
                        target="modulus",
                    )
                ]

            @property
            def poles(self):
                return self._poles

        return LorentzMaterial()

    def test_add_lorentz_material(self, lorentz_material):
        """Adding a Lorentz pole material should work."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Create region mask
        region_mask = np.zeros((30, 30, 30), dtype=bool)
        region_mask[10:20, 10:20, 10:20] = True

        # Add material
        mat_id = solver.add_material(lorentz_material, region_mask)

        assert mat_id == 1
        assert solver.has_materials is True

    def test_lorentz_material_simulation_runs(self, lorentz_material):
        """Simulation with Lorentz pole materials should run without errors."""
        solver = GPUFDTDSolver(shape=(40, 40, 40), resolution=2e-3)

        # Add material in center region
        region_mask = np.zeros((40, 40, 40), dtype=bool)
        region_mask[15:25, 15:25, 15:25] = True
        solver.add_material(lorentz_material, region_mask)

        # Add source and probe
        solver.add_source(position=(10, 20, 20), frequency=1000)
        solver.add_probe("test", position=(30, 20, 20))

        # Run simulation
        solver.run(duration=0.002)

        # Should complete without error
        data = solver.get_probe_data()
        assert len(data["test"]) > 0

    def test_lorentz_ade_fields_have_j_prev(self, lorentz_material):
        """Lorentz poles should have both J and J_prev fields."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Add material
        region_mask = np.ones((30, 30, 30), dtype=bool)
        solver.add_material(lorentz_material, region_mask)

        # Run one step to initialize ADE fields
        solver.step()

        # Check that Lorentz poles have both J and J_prev
        assert solver._ade_initialized is True
        mat_id = 1
        pole_idx = 0
        pole_fields = solver._ade_fields[mat_id][pole_idx]
        assert "J" in pole_fields
        assert "J_prev" in pole_fields

    def test_mixed_debye_lorentz_material(self, debye_material):
        """Material with both Debye and Lorentz poles should work."""
        from strata_fdtd.materials.base import Pole, PoleType, SimpleMaterial

        class MixedMaterial(SimpleMaterial):
            def __init__(self):
                super().__init__(name="test_mixed", _rho=1.2, _c=343.0)
                self._poles = [
                    # Debye pole for density
                    Pole(
                        pole_type=PoleType.DEBYE,
                        delta_chi=0.3,
                        tau=1e-4,
                        target="density",
                    ),
                    # Lorentz pole for modulus
                    Pole(
                        pole_type=PoleType.LORENTZ,
                        delta_chi=0.5,
                        omega_0=2 * np.pi * 1000,
                        gamma=100,
                        target="modulus",
                    ),
                ]

            @property
            def poles(self):
                return self._poles

        material = MixedMaterial()
        solver = GPUFDTDSolver(shape=(40, 40, 40), resolution=2e-3)

        # Add material
        region_mask = np.zeros((40, 40, 40), dtype=bool)
        region_mask[15:25, 15:25, 15:25] = True
        solver.add_material(material, region_mask)

        # Add source and probe
        solver.add_source(position=(10, 20, 20), frequency=1000)
        solver.add_probe("test", position=(30, 20, 20))

        # Run simulation
        solver.run(duration=0.002)

        # Should complete without error
        data = solver.get_probe_data()
        assert len(data["test"]) > 0

    def test_lorentz_resonance_at_omega0(self, lorentz_material):
        """Lorentz poles should exhibit resonance at ω₀."""
        from strata_fdtd.materials.base import Pole, PoleType, SimpleMaterial

        # Create material with known resonance frequency
        f0 = 1000  # Hz
        omega_0 = 2 * np.pi * f0

        class ResonantMaterial(SimpleMaterial):
            def __init__(self):
                super().__init__(name="resonant", _rho=1.2, _c=343.0)
                self._poles = [
                    Pole(
                        pole_type=PoleType.LORENTZ,
                        delta_chi=0.8,  # Strong resonance
                        omega_0=omega_0,
                        gamma=50,  # Low damping for sharp peak
                        target="modulus",
                    )
                ]

            @property
            def poles(self):
                return self._poles

        material = ResonantMaterial()
        solver = GPUFDTDSolver(shape=(50, 50, 50), resolution=2e-3)

        # Add material in center region
        region_mask = np.zeros((50, 50, 50), dtype=bool)
        region_mask[20:30, 20:30, 20:30] = True
        solver.add_material(material, region_mask)

        # Test at three frequencies: below, at, and above resonance
        test_freqs = [800, 1000, 1200]  # Hz
        amplitudes = []

        for freq in test_freqs:
            solver.reset()
            solver.add_source(position=(15, 25, 25), frequency=freq, bandwidth=freq * 0.3)
            solver.add_probe("test", position=(35, 25, 25))
            solver.run(duration=0.005)

            data = solver.get_probe_data()["test"]
            # Measure peak amplitude after transient
            steady_state = data[len(data) // 2:]
            amplitude = np.max(np.abs(steady_state))
            amplitudes.append(amplitude)

        # Verify that amplitude is highest at resonance frequency
        assert amplitudes[1] > amplitudes[0], "Resonance peak should be higher than below-resonance"
        assert amplitudes[1] > amplitudes[2], "Resonance peak should be higher than above-resonance"

    def test_lorentz_damping_behavior(self):
        """Higher γ (damping) should reduce resonance quality factor."""
        from strata_fdtd.materials.base import Pole, PoleType, SimpleMaterial

        f0 = 1000  # Hz
        omega_0 = 2 * np.pi * f0

        # Test two materials with different damping
        damping_values = [50, 200]  # Low and high damping
        quality_factors = []

        for gamma in damping_values:

            class ResonantMaterial(SimpleMaterial):
                def __init__(self, g):
                    super().__init__(name=f"resonant_g{g}", _rho=1.2, _c=343.0)
                    self._poles = [
                        Pole(
                            pole_type=PoleType.LORENTZ,
                            delta_chi=0.8,
                            omega_0=omega_0,
                            gamma=g,
                            target="modulus",
                        )
                    ]

                @property
                def poles(self):
                    return self._poles

            material = ResonantMaterial(gamma)
            solver = GPUFDTDSolver(shape=(50, 50, 50), resolution=2e-3)

            # Add material
            region_mask = np.zeros((50, 50, 50), dtype=bool)
            region_mask[20:30, 20:30, 20:30] = True
            solver.add_material(material, region_mask)

            # Measure response at resonance frequency
            solver.add_source(position=(15, 25, 25), frequency=f0, bandwidth=f0 * 0.3)
            solver.add_probe("test", position=(35, 25, 25))
            solver.run(duration=0.005)

            data = solver.get_probe_data()["test"]
            steady_state = data[len(data) // 2:]
            amplitude = np.max(np.abs(steady_state))

            # Quality factor Q = ω₀/γ (theoretical)
            # Response amplitude should scale with Q
            quality_factors.append(amplitude)

        # Verify that higher damping (γ) reduces response amplitude
        assert quality_factors[0] > quality_factors[1], \
            f"Lower damping should give higher amplitude: {quality_factors[0]} vs {quality_factors[1]}"

    def test_material_simulation_runs(self, debye_material):
        """Simulation with materials should run without errors."""
        solver = GPUFDTDSolver(shape=(40, 40, 40), resolution=2e-3)

        # Add material in center region
        region_mask = np.zeros((40, 40, 40), dtype=bool)
        region_mask[15:25, 15:25, 15:25] = True
        solver.add_material(debye_material, region_mask)

        # Add source and probe
        solver.add_source(position=(10, 20, 20), frequency=1000)
        solver.add_probe("test", position=(30, 20, 20))

        # Run simulation
        solver.run(duration=0.002)

        # Should complete without error
        data = solver.get_probe_data()
        assert len(data["test"]) > 0

    def test_ade_fields_initialized(self, debye_material):
        """ADE fields should be initialized on first step."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Add material
        region_mask = np.ones((30, 30, 30), dtype=bool)
        solver.add_material(debye_material, region_mask)

        # Before first step
        assert solver._ade_initialized is False

        # Run one step
        solver.step()

        # After first step
        assert solver._ade_initialized is True
        assert len(solver._ade_fields) > 0

    def test_material_reset_clears_ade_fields(self, debye_material):
        """Reset should clear ADE auxiliary fields."""
        solver = GPUFDTDSolver(shape=(30, 30, 30), resolution=2e-3)

        # Add material and run
        region_mask = np.ones((30, 30, 30), dtype=bool)
        solver.add_material(debye_material, region_mask)
        solver.run(steps=10)

        # Reset
        solver.reset()

        # ADE fields should still exist but be zeroed
        assert solver._ade_initialized is True
        # Check that fields are zeros (after running through PyTorch)
        for mat_fields in solver._ade_fields.values():
            for pole_fields in mat_fields.values():
                for field in pole_fields.values():
                    assert field.abs().sum().item() == 0.0


# =============================================================================
# PML Boundary Tests
# =============================================================================


class TestGPUPML:
    """Tests for PML absorbing boundaries on GPU solvers."""

    def test_gpufdtd_accepts_pml_parameter(self):
        """GPUFDTDSolver should accept pml_layers parameter."""
        solver = GPUFDTDSolver(
            shape=(40, 40, 40), resolution=2e-3, pml_layers=10
        )
        assert solver.pml_layers == 10
        assert solver._pml_enabled is True

    def test_batched_accepts_pml_parameter(self):
        """BatchedGPUFDTDSolver should accept pml_layers parameter."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
            pml_layers=10,
        )
        assert solver.pml_layers == 10
        assert solver._pml_enabled is True

    def test_pml_reduces_reflections_single_band(self):
        """PML should reduce reflections compared to rigid boundaries."""
        # Solver WITHOUT PML (rigid boundaries)
        solver_rigid = GPUFDTDSolver(
            shape=(80, 30, 30), resolution=2e-3, pml_layers=0
        )
        solver_rigid.add_source(
            position=(20, 15, 15), frequency=1000, bandwidth=500
        )
        solver_rigid.add_probe("near_source", position=(25, 15, 15))
        solver_rigid.run(duration=0.01)
        data_rigid = solver_rigid.get_probe_data()["near_source"]

        # Solver WITH PML
        solver_pml = GPUFDTDSolver(
            shape=(80, 30, 30), resolution=2e-3, pml_layers=10
        )
        solver_pml.add_source(
            position=(20, 15, 15), frequency=1000, bandwidth=500
        )
        solver_pml.add_probe("near_source", position=(25, 15, 15))
        solver_pml.run(duration=0.01)
        data_pml = solver_pml.get_probe_data()["near_source"]

        # After initial pulse passes, PML should have much less ringing
        # Look at tail after main pulse (last 25% of signal)
        tail_start = int(0.75 * len(data_rigid))
        tail_rigid = data_rigid[tail_start:]
        tail_pml = data_pml[tail_start:]

        # RMS of tail should be much lower with PML
        rms_rigid = np.sqrt(np.mean(tail_rigid**2))
        rms_pml = np.sqrt(np.mean(tail_pml**2))

        # PML should reduce tail energy significantly
        assert rms_pml < 0.5 * rms_rigid, (
            f"PML tail RMS ({rms_pml:.2e}) not significantly less than "
            f"rigid ({rms_rigid:.2e})"
        )

    def test_pml_reflection_reduction_exceeds_40db(self):
        """PML should provide >40dB reflection reduction."""
        # Setup: source near one end, probe near other end
        # Measure reflection from far boundary
        solver_pml = GPUFDTDSolver(
            shape=(100, 30, 30), resolution=2e-3, pml_layers=15
        )
        solver_pml.add_source(
            position=(20, 15, 15), frequency=1000, bandwidth=500
        )
        # Probe near the far boundary to detect reflections
        solver_pml.add_probe("near_boundary", position=(80, 15, 15))
        solver_pml.run(duration=0.015)

        data = solver_pml.get_probe_data()["near_boundary"]

        # Identify direct wave arrival and reflection
        # Direct wave arrives first, reflection comes later
        # Split signal in half
        n = len(data)
        first_half = data[:n//2]
        second_half = data[n//2:]

        # Peak of direct wave
        peak_direct = np.max(np.abs(first_half))
        # Peak of reflection (should be much smaller with PML)
        peak_reflection = np.max(np.abs(second_half))

        # Compute reflection reduction in dB
        if peak_reflection > 0:
            reduction_db = 20 * np.log10(peak_direct / peak_reflection)
        else:
            reduction_db = np.inf

        # Should exceed 40dB reduction
        assert reduction_db > 40, (
            f"PML reflection reduction ({reduction_db:.1f}dB) < 40dB threshold"
        )

    def test_batched_pml_works_for_all_bands(self):
        """PML should work correctly for all bands in batched solver."""
        solver = BatchedGPUFDTDSolver(
            shape=(60, 30, 30),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400), (2000, 800)],
            pml_layers=10,
        )
        results = solver.run(duration=0.01)

        # All bands should produce signal that decays (PML absorbs energy)
        for name, data in results.items():
            # Signal should have some amplitude initially
            max_amplitude = np.max(np.abs(data))
            assert max_amplitude > 0, f"Band {name} has no signal"

            # Tail (last 20%) should be significantly quieter than peak
            tail_start = int(0.8 * len(data))
            tail_rms = np.sqrt(np.mean(data[tail_start:]**2))
            assert tail_rms < 0.3 * max_amplitude, (
                f"Band {name} tail not decaying with PML"
            )

    def test_pml_zero_layers_disables_pml(self):
        """pml_layers=0 should disable PML (rigid boundaries)."""
        solver = GPUFDTDSolver(
            shape=(40, 40, 40), resolution=2e-3, pml_layers=0
        )
        assert solver.pml_layers == 0
        assert solver._pml_enabled is False

    def test_pml_with_cpu_device(self):
        """PML should work with CPU device."""
        solver = GPUFDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            device="cpu",
            pml_layers=8,
        )
        solver.add_source(frequency=1000)
        solver.add_probe("test", position=(20, 15, 15))
        solver.run(steps=50)

        data = solver.get_probe_data()["test"]
        # Should complete without error and produce data
        assert len(data) == 50
        assert np.max(np.abs(data)) > 0

    def test_negative_pml_layers_raises_error(self):
        """Negative pml_layers should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=-5,
            )

    def test_pml_layers_too_large_raises_error(self):
        """pml_layers >= min(shape)//2 should raise ValueError."""
        # Shape (20, 20, 20) -> max_pml = 20//2 = 10
        # pml_layers must be < 10
        with pytest.raises(ValueError, match="too large for grid shape"):
            GPUFDTDSolver(
                shape=(20, 20, 20),
                resolution=2e-3,
                pml_layers=10,  # >= max_pml, should fail
            )

        # Also test with pml_layers > max_pml
        with pytest.raises(ValueError, match="too large for grid shape"):
            GPUFDTDSolver(
                shape=(20, 20, 20),
                resolution=2e-3,
                pml_layers=15,
            )

    def test_pml_at_maximum_allowed_works(self):
        """pml_layers at maximum allowed value should work."""
        # Shape (40, 40, 40) -> max_pml = 40//2 = 20
        # pml_layers = 19 should work (< 20)
        solver = GPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            pml_layers=19,
        )
        assert solver.pml_layers == 19
        assert solver._pml_enabled is True

    def test_batched_negative_pml_layers_raises_error(self):
        """Negative pml_layers should raise ValueError in batched solver."""
        with pytest.raises(ValueError, match="must be non-negative"):
            BatchedGPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                bands=[(500, 200)],
                pml_layers=-3,
            )

    def test_batched_pml_layers_too_large_raises_error(self):
        """pml_layers too large should raise ValueError in batched solver."""
        with pytest.raises(ValueError, match="too large for grid shape"):
            BatchedGPUFDTDSolver(
                shape=(20, 20, 20),
                resolution=2e-3,
                bands=[(500, 200)],
                pml_layers=10,  # >= 20//2, should fail
            )

    def test_batched_pml_at_maximum_allowed_works(self):
        """pml_layers at maximum should work in batched solver."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
            pml_layers=19,  # max is 20, so 19 should work
        )
        assert solver.pml_layers == 19
        assert solver._pml_enabled is True

    def test_gpufdtd_accepts_pml_r_target(self):
        """GPUFDTDSolver should accept pml_r_target parameter."""
        solver = GPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            pml_layers=10,
            pml_r_target=1e-8,
        )
        assert solver.pml_r_target == 1e-8
        assert solver.pml_layers == 10

    def test_gpufdtd_accepts_pml_order(self):
        """GPUFDTDSolver should accept pml_order parameter."""
        solver = GPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            pml_layers=10,
            pml_order=2,
        )
        assert solver.pml_order == 2

    def test_gpufdtd_accepts_pml_max_sigma(self):
        """GPUFDTDSolver should accept pml_max_sigma parameter."""
        solver = GPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            pml_layers=10,
            pml_max_sigma=500.0,
        )
        assert solver.pml_max_sigma == 500.0

    def test_batched_accepts_pml_r_target(self):
        """BatchedGPUFDTDSolver should accept pml_r_target parameter."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
            pml_layers=10,
            pml_r_target=1e-8,
        )
        assert solver.pml_r_target == 1e-8

    def test_batched_accepts_pml_order(self):
        """BatchedGPUFDTDSolver should accept pml_order parameter."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
            pml_layers=10,
            pml_order=2,
        )
        assert solver.pml_order == 2

    def test_batched_accepts_pml_max_sigma(self):
        """BatchedGPUFDTDSolver should accept pml_max_sigma parameter."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 40, 40),
            resolution=2e-3,
            bands=[(500, 200), (1000, 400)],
            pml_layers=10,
            pml_max_sigma=500.0,
        )
        assert solver.pml_max_sigma == 500.0


# =============================================================================
# Band Combination Tests
# =============================================================================


class TestBandCombination:
    """Tests for combine_bands() method."""

    @pytest.fixture
    def batched_solver_for_combination(self):
        """Create a batched solver for band combination tests."""
        return BatchedGPUFDTDSolver(
            shape=(50, 30, 30),
            resolution=2e-3,
            bands=[(250, 100), (500, 200), (1000, 400), (2000, 800)],
        )

    def test_combine_bands_returns_three_arrays(
        self, batched_solver_for_combination
    ):
        """combine_bands() should return (frequencies, magnitude, phase)."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results)

        assert isinstance(freqs, np.ndarray)
        assert isinstance(mag, np.ndarray)
        assert isinstance(phase, np.ndarray)

    def test_combine_bands_arrays_have_same_length(
        self, batched_solver_for_combination
    ):
        """All three returned arrays should have the same length."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results)

        assert len(freqs) == len(mag)
        assert len(freqs) == len(phase)

    def test_combine_bands_frequencies_are_sorted(
        self, batched_solver_for_combination
    ):
        """Frequency array should be monotonically increasing."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, _mag, _phase = solver.combine_bands(results)

        assert np.all(np.diff(freqs) >= 0)

    def test_combine_bands_magnitude_is_positive(
        self, batched_solver_for_combination
    ):
        """Magnitude should be non-negative."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        _freqs, mag, _phase = solver.combine_bands(results)

        assert np.all(mag >= 0)

    def test_combine_bands_phase_is_in_range(
        self, batched_solver_for_combination
    ):
        """Phase should be in [-π, π] range."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        _freqs, _mag, phase = solver.combine_bands(results)

        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_combine_bands_overlap_zero_works(
        self, batched_solver_for_combination
    ):
        """overlap=0.0 should work (minimal blending)."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results, overlap=0.0)

        assert len(freqs) > 0
        assert len(mag) > 0
        assert len(phase) > 0

    def test_combine_bands_overlap_one_works(
        self, batched_solver_for_combination
    ):
        """overlap=1.0 should work (maximum blending)."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results, overlap=1.0)

        assert len(freqs) > 0
        assert len(mag) > 0
        assert len(phase) > 0

    def test_combine_bands_invalid_overlap_raises_error(
        self, batched_solver_for_combination
    ):
        """Invalid overlap values should raise ValueError."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        with pytest.raises(ValueError, match="overlap must be in"):
            solver.combine_bands(results, overlap=-0.1)

        with pytest.raises(ValueError, match="overlap must be in"):
            solver.combine_bands(results, overlap=1.5)

    def test_combine_bands_no_data_raises_error(
        self, batched_solver_for_combination
    ):
        """Calling combine_bands() with no data should raise ValueError."""
        solver = batched_solver_for_combination

        with pytest.raises(ValueError, match="has no samples"):
            solver.combine_bands()

    def test_combine_bands_empty_dict_raises_error(
        self, batched_solver_for_combination
    ):
        """Calling combine_bands() with empty dict should raise ValueError."""
        solver = batched_solver_for_combination

        with pytest.raises(ValueError, match="empty"):
            solver.combine_bands({})

    def test_combine_bands_missing_band_raises_error(
        self, batched_solver_for_combination
    ):
        """Missing band in probe_data should raise ValueError."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        # Remove one band
        incomplete_results = {k: v for k, v in results.items() if k != "500Hz"}

        with pytest.raises(ValueError, match="not found in probe_data"):
            solver.combine_bands(incomplete_results)

    def test_combine_bands_different_overlaps_affect_smoothness(
        self, batched_solver_for_combination
    ):
        """Different overlap values should produce different results."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        _freqs1, mag1, _phase1 = solver.combine_bands(results, overlap=0.2)
        _freqs2, mag2, _phase2 = solver.combine_bands(results, overlap=0.8)

        # Results should differ (different weighting)
        assert not np.allclose(mag1, mag2)

    def test_combine_bands_produces_smooth_response(
        self, batched_solver_for_combination
    ):
        """Combined response should be smooth (no NaN or Inf)."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.01)

        freqs, mag, phase = solver.combine_bands(results, overlap=0.5)

        # Check for NaN or Inf
        assert not np.any(np.isnan(freqs))
        assert not np.any(np.isnan(mag))
        assert not np.any(np.isnan(phase))
        assert not np.any(np.isinf(freqs))
        assert not np.any(np.isinf(mag))
        assert not np.any(np.isinf(phase))

    def test_combine_bands_covers_all_band_centers(
        self, batched_solver_for_combination
    ):
        """Combined response should cover all band center frequencies."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.01)

        freqs, _mag, _phase = solver.combine_bands(results, overlap=0.5)

        # Check that all band centers are within the frequency range
        for center_freq, _bandwidth in solver.bands:
            assert freqs[0] <= center_freq <= freqs[-1]

    def test_combine_bands_single_band(self):
        """combine_bands() should work with a single band."""
        solver = BatchedGPUFDTDSolver(
            shape=(40, 30, 30),
            resolution=2e-3,
            bands=[(1000, 400)],
        )
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results, overlap=0.5)

        assert len(freqs) > 0
        assert len(mag) > 0
        assert len(phase) > 0
        assert np.all(mag >= 0)

    def test_combine_bands_magnitude_has_peaks_near_band_centers(
        self, batched_solver_for_combination
    ):
        """Magnitude response should have energy near band center frequencies."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.015)

        freqs, mag, _phase = solver.combine_bands(results, overlap=0.5)

        # For each band, check that there's significant energy near the center
        for center_freq, bandwidth in solver.bands:
            # Find frequencies within bandwidth/2 of center
            freq_mask = np.abs(freqs - center_freq) < bandwidth
            if np.any(freq_mask):
                band_mag = mag[freq_mask]
                # Should have some non-zero magnitude in this band
                assert np.max(band_mag) > 0

    def test_combine_bands_dtype_is_float32(
        self, batched_solver_for_combination
    ):
        """Returned arrays should be float32."""
        solver = batched_solver_for_combination
        results = solver.run(duration=0.005)

        freqs, mag, phase = solver.combine_bands(results, overlap=0.5)

        assert freqs.dtype == np.float32
        assert mag.dtype == np.float32
        assert phase.dtype == np.float32


# =============================================================================
# PML Tests (continued)
# =============================================================================


class TestGPUPMLContinued:
    """Additional PML tests."""

    def test_pml_with_different_r_target(self):
        """PML with different pml_r_target should still reduce reflections."""
        solver = GPUFDTDSolver(
            shape=(80, 30, 30),
            resolution=2e-3,
            pml_layers=15,
            pml_r_target=1e-8,
        )
        solver.add_source(position=(20, 15, 15), frequency=1000, bandwidth=500)
        solver.add_probe("probe", position=(25, 15, 15))
        solver.run(duration=0.01)
        data = solver.get_probe_data()["probe"]

        assert len(data) > 0
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))

    def test_pml_with_lower_order(self):
        """PML with lower order should still work."""
        solver = GPUFDTDSolver(
            shape=(80, 30, 30),
            resolution=2e-3,
            pml_layers=10,
            pml_order=2,
        )
        solver.add_source(position=(20, 15, 15), frequency=1000, bandwidth=500)
        solver.add_probe("probe", position=(25, 15, 15))
        solver.run(duration=0.01)
        data = solver.get_probe_data()["probe"]

        assert len(data) > 0
        assert not np.any(np.isnan(data))

    def test_pml_with_manual_max_sigma(self):
        """PML with manual pml_max_sigma should work."""
        solver = GPUFDTDSolver(
            shape=(80, 30, 30),
            resolution=2e-3,
            pml_layers=10,
            pml_max_sigma=300.0,
        )
        solver.add_source(position=(20, 15, 15), frequency=1000, bandwidth=500)
        solver.add_probe("probe", position=(25, 15, 15))
        solver.run(duration=0.01)
        data = solver.get_probe_data()["probe"]

        assert len(data) > 0
        assert not np.any(np.isnan(data))


# =============================================================================
# Batched Solver with Materials Tests
# =============================================================================


class TestBatchedSolverMaterials:
    """Tests for BatchedGPUFDTDSolver with frequency-dependent materials."""

    @pytest.fixture
    def batched_solver_with_material(self):
        """Create a batched solver with a simple material."""
        from strata_fdtd.materials import FIBERGLASS_48

        solver = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200), (2000, 800)],
        )

        # Add material in center region
        mask = np.zeros((50, 50, 50), dtype=bool)
        mask[20:30, 20:30, 20:30] = True
        solver.add_material(FIBERGLASS_48, mask)

        return solver

    def test_add_material_basic(self):
        """Should be able to add materials to batched solver."""
        from strata_fdtd.materials import FIBERGLASS_48

        solver = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200), (2000, 800)],
        )

        mask = np.zeros((50, 50, 50), dtype=bool)
        mask[20:30, 20:30, 20:30] = True

        mat_id = solver.add_material(FIBERGLASS_48, mask)
        assert mat_id == 1
        assert solver.has_materials
        assert len(solver._materials) == 1

    def test_add_multiple_materials(self):
        """Should be able to add multiple materials."""
        from strata_fdtd.materials import FIBERGLASS_48, MDF_MEDIUM

        solver = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200)],
        )

        mask1 = np.zeros((50, 50, 50), dtype=bool)
        mask1[10:20, 10:20, 10:20] = True

        mask2 = np.zeros((50, 50, 50), dtype=bool)
        mask2[30:40, 30:40, 30:40] = True

        mat_id1 = solver.add_material(FIBERGLASS_48, mask1)
        mat_id2 = solver.add_material(MDF_MEDIUM, mask2)

        assert mat_id1 == 1
        assert mat_id2 == 2
        assert len(solver._materials) == 2

    def test_material_mask_wrong_shape_raises(self):
        """Adding material with wrong mask shape should raise error."""
        from strata_fdtd.materials import FIBERGLASS_48

        solver = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200)],
        )

        wrong_mask = np.zeros((40, 40, 40), dtype=bool)

        with pytest.raises(ValueError, match="doesn't match grid shape"):
            solver.add_material(FIBERGLASS_48, wrong_mask)

    def test_ade_fields_initialized_on_first_step(self, batched_solver_with_material):
        """ADE fields should be initialized on first step."""
        assert not batched_solver_with_material._ade_initialized

        batched_solver_with_material.step()

        assert batched_solver_with_material._ade_initialized
        assert len(batched_solver_with_material._ade_fields) > 0

    def test_ade_fields_are_batched(self, batched_solver_with_material):
        """ADE auxiliary fields should have batch dimension."""
        batched_solver_with_material.step()

        # Check that J fields have shape (n_bands, nx, ny, nz)
        for poles_data in batched_solver_with_material._ade_fields.values():
            for pole_data in poles_data.values():
                J = pole_data["J"]
                assert J.shape[0] == batched_solver_with_material.n_bands
                assert J.shape[1:] == batched_solver_with_material.shape

    def test_materials_affect_different_bands_differently(self):
        """Different frequency bands should see different material effects."""
        from strata_fdtd.materials import FIBERGLASS_48

        solver = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200), (2000, 800)],  # Low and high frequency
            source_position=(10, 25, 25),
            probe_position=(40, 25, 25),
        )

        # Add absorbing material in the middle
        mask = np.zeros((50, 50, 50), dtype=bool)
        mask[20:30, :, :] = True  # Wall blocking propagation path
        solver.add_material(FIBERGLASS_48, mask)

        results = solver.run(duration=0.01)

        # Both bands should have recorded data
        assert "500Hz" in results
        assert "2000Hz" in results

        # Signals should be different due to frequency-dependent absorption
        # (FIBERGLASS_48 has multiple Debye poles affecting both density and modulus)
        data_low = results["500Hz"]
        data_high = results["2000Hz"]

        # Data should not be identical
        assert not np.allclose(data_low, data_high, rtol=0.1)

    def test_memory_usage_includes_ade_fields(self, batched_solver_with_material):
        """Memory usage calculation should include ADE fields."""
        # Memory without materials
        solver_no_mat = BatchedGPUFDTDSolver(
            shape=(50, 50, 50),
            resolution=1e-3,
            bands=[(500, 200), (2000, 800)],
        )
        mem_no_mat = solver_no_mat.memory_usage_mb()

        # Memory with materials
        mem_with_mat = batched_solver_with_material.memory_usage_mb()

        # Should have more memory with materials
        assert mem_with_mat > mem_no_mat

    def test_reset_clears_ade_fields(self, batched_solver_with_material):
        """Reset should clear ADE auxiliary fields."""
        # Run simulation to initialize ADE fields
        batched_solver_with_material.run(steps=10)

        # Check that ADE fields have non-zero values
        has_nonzero = False
        for poles_data in batched_solver_with_material._ade_fields.values():
            for pole_data in poles_data.values():
                if pole_data["J"].abs().max().item() > 0:
                    has_nonzero = True
                    break

        assert has_nonzero, "ADE fields should have non-zero values after simulation"

        # Reset
        batched_solver_with_material.reset()

        # All ADE fields should be zero
        for poles_data in batched_solver_with_material._ade_fields.values():
            for pole_data in poles_data.values():
                assert pole_data["J"].abs().max().item() == 0
                if "J_prev" in pole_data:
                    assert pole_data["J_prev"].abs().max().item() == 0

    def test_simulation_runs_with_materials(self, batched_solver_with_material):
        """Simulation should complete successfully with materials."""
        # Should not raise any exceptions
        results = batched_solver_with_material.run(duration=0.005)

        # Should produce valid results
        assert len(results) == batched_solver_with_material.n_bands
        for data in results.values():
            assert len(data) > 0
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))


# =============================================================================
# PML Parameter Validation Tests
# =============================================================================


class TestPMLParameterValidation:
    """Tests for PML parameter validation."""

    def test_pml_r_target_validation(self):
        """Invalid pml_r_target values should raise ValueError."""
        # pml_r_target <= 0
        with pytest.raises(ValueError, match="pml_r_target must be in range"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_r_target=0.0,
            )

        # pml_r_target >= 1
        with pytest.raises(ValueError, match="pml_r_target must be in range"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_r_target=1.0,
            )

        # Negative value
        with pytest.raises(ValueError, match="pml_r_target must be in range"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_r_target=-0.5,
            )

    def test_pml_order_validation(self):
        """Invalid pml_order values should raise ValueError."""
        # pml_order < 1
        with pytest.raises(ValueError, match="pml_order must be >= 1"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_order=0,
            )

        # Negative order
        with pytest.raises(ValueError, match="pml_order must be >= 1"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_order=-1,
            )

    def test_pml_max_sigma_validation(self):
        """Invalid pml_max_sigma values should raise ValueError."""
        # pml_max_sigma <= 0
        with pytest.raises(ValueError, match="pml_max_sigma must be positive"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_max_sigma=0.0,
            )

        # Negative value
        with pytest.raises(ValueError, match="pml_max_sigma must be positive"):
            GPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                pml_layers=10,
                pml_max_sigma=-100.0,
            )

    def test_batched_pml_r_target_validation(self):
        """Invalid pml_r_target in batched solver should raise ValueError."""
        with pytest.raises(ValueError, match="pml_r_target must be in range"):
            BatchedGPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                bands=[(500, 200), (1000, 400)],
                pml_layers=10,
                pml_r_target=1.5,
            )

    def test_batched_pml_order_validation(self):
        """Invalid pml_order in batched solver should raise ValueError."""
        with pytest.raises(ValueError, match="pml_order must be >= 1"):
            BatchedGPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                bands=[(500, 200), (1000, 400)],
                pml_layers=10,
                pml_order=0,
            )

    def test_batched_pml_max_sigma_validation(self):
        """Invalid pml_max_sigma in batched solver should raise ValueError."""
        with pytest.raises(ValueError, match="pml_max_sigma must be positive"):
            BatchedGPUFDTDSolver(
                shape=(40, 40, 40),
                resolution=2e-3,
                bands=[(500, 200), (1000, 400)],
                pml_layers=10,
                pml_max_sigma=-50.0,
            )
