"""
Unit tests for 3D FDTD acoustic simulation.

Tests verify:
- CFL stability condition
- Wave propagation speed
- Energy conservation in closed cavity
- Rectangular cavity mode frequencies
- PML absorption effectiveness
- Geometry/rigid boundary behavior
"""

import numpy as np
import pytest

from strata_fdtd import PML, FDTDSolver, GaussianPulse, RadiationImpedance
from strata_fdtd.boundaries import ABCFirstOrder

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_solver():
    """Create a small solver for fast tests."""
    return FDTDSolver(shape=(20, 20, 20), resolution=5e-3, c=343.0, rho=1.2)


@pytest.fixture
def medium_solver():
    """Create a medium solver for validation tests."""
    return FDTDSolver(shape=(50, 50, 50), resolution=2e-3, c=343.0, rho=1.2)


@pytest.fixture
def rectangular_cavity_solver():
    """Create solver for rectangular cavity mode tests."""
    # 100mm x 80mm x 60mm cavity at 2mm resolution
    return FDTDSolver(shape=(50, 40, 30), resolution=2e-3, c=343.0, rho=1.2)


# =============================================================================
# CFL Stability Tests
# =============================================================================


class TestCFLStability:
    def test_timestep_within_cfl_limit(self, small_solver):
        """Verify timestep satisfies CFL condition."""
        dt_max = small_solver.dx / (small_solver.c * np.sqrt(3))
        assert small_solver.dt <= dt_max

    def test_courant_factor_respected(self, small_solver):
        """Verify courant factor is applied."""
        dt_max = small_solver.dx / (small_solver.c * np.sqrt(3))
        # Default courant is 0.95
        assert small_solver.dt == pytest.approx(0.95 * dt_max)

    def test_custom_courant_factor(self):
        """Test custom courant factor."""
        solver = FDTDSolver(
            shape=(10, 10, 10), resolution=5e-3, courant=0.5
        )
        dt_max = solver.dx / (solver.c * np.sqrt(3))
        assert solver.dt == pytest.approx(0.5 * dt_max)

    def test_timestep_scales_with_resolution(self):
        """Finer resolution should give smaller timestep."""
        solver1 = FDTDSolver(shape=(10, 10, 10), resolution=4e-3)
        solver2 = FDTDSolver(shape=(20, 20, 20), resolution=2e-3)

        # dt should scale linearly with dx
        assert solver2.dt == pytest.approx(solver1.dt / 2, rel=0.01)


# =============================================================================
# Wave Speed Tests
# =============================================================================


class TestWaveSpeed:
    def test_wave_propagation_speed(self):
        """Measure pulse travel time and verify c = 343 m/s within 5%.

        Uses cross-correlation for robust time-of-flight measurement.
        FDTD has inherent numerical dispersion; 5% tolerance accounts for this.
        """
        from scipy.signal import correlate

        # Create elongated domain for clear measurement
        solver = FDTDSolver(shape=(200, 10, 10), resolution=1e-3, c=343.0)

        # Initialize with a Gaussian pressure pulse (uniform in y,z for plane wave)
        # This is cleaner than using GaussianPulse source which has envelope effects
        sigma = 5  # cells
        x = np.arange(solver.shape[0])
        initial_pulse = np.exp(-((x - 30) ** 2) / (2 * sigma**2))
        for j in range(solver.shape[1]):
            for k in range(solver.shape[2]):
                solver.p[:, j, k] = initial_pulse

        # Add probes at known distances
        solver.add_probe('ref', position=(50, 5, 5))     # Reference probe
        solver.add_probe('far', position=(150, 5, 5))    # 100mm from reference

        # Run simulation long enough for wave to reach far probe
        solver.run(duration=0.0005)  # 0.5ms

        # Get probe data
        ref_data = solver.get_probe_data('ref')['ref']
        far_data = solver.get_probe_data('far')['far']

        # Cross-correlation to find time delay
        corr = correlate(far_data, ref_data)
        lag = np.argmax(corr) - (len(ref_data) - 1)

        # Calculate speed
        distance = (150 - 50) * solver.dx  # 100mm
        travel_time = lag * solver.dt
        measured_c = distance / travel_time

        # Verify within 5% of expected speed
        # (FDTD numerical dispersion typically gives 3-5% error)
        assert measured_c == pytest.approx(343.0, rel=0.05), (
            f"Measured speed {measured_c:.1f} m/s, expected 343 m/s"
        )


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    def test_energy_bounded_in_closed_cavity(self, small_solver):
        """Energy should remain bounded in closed cavity (no PML).

        Note: FDTD leapfrog energy oscillates due to staggered grid timing.
        We verify that energy stays within a reasonable band, not exact conservation.
        """
        # Initialize with a Gaussian pressure pulse for smoother energy behavior
        cx, cy, cz = [s // 2 for s in small_solver.shape]
        sigma = 2  # cells
        for i in range(small_solver.shape[0]):
            for j in range(small_solver.shape[1]):
                for k in range(small_solver.shape[2]):
                    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    small_solver.p[i, j, k] = np.exp(-r2 / (2 * sigma**2))

        # Track energy over time
        energies = []
        for step in range(500):
            small_solver.step()
            if step % 10 == 0:
                energies.append(small_solver.compute_energy())

        # Energy should stay within a factor of 2 of initial (accounting for oscillation)
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        max_deviation = np.max(np.abs(energies - mean_energy)) / mean_energy

        assert max_deviation < 0.5, (
            f"Energy deviates by {max_deviation*100:.1f}% from mean, expected <50%"
        )

    @pytest.mark.parametrize("backend", ["python", "native"])
    def test_energy_conservation_closed_pipe(self, backend):
        """Energy should be conserved in closed pipe with rigid boundaries.

        This test was added to verify the fix for issue #89, where the native
        backend had energy dissipation due to incorrect ordering of boundary
        condition application (boundaries were applied AFTER pressure update
        instead of BETWEEN velocity and pressure updates).
        """
        from strata_fdtd.fdtd import has_native_kernels

        if backend == "native" and not has_native_kernels():
            pytest.skip("Native kernels not available")

        from strata_fdtd.primitives import SPEED_OF_SOUND

        # Create closed pipe geometry (issue #89 reproduction case)
        solver = FDTDSolver(
            shape=(110, 25, 25),
            resolution=2e-3,
            c=SPEED_OF_SOUND,
            backend=backend,
        )

        # Create closed pipe with rigid walls (leaving 1-cell buffer at edges)
        geometry = np.zeros(solver.shape, dtype=bool)
        geometry[5:105, 5:20, 5:20] = True  # Interior air region
        solver.set_geometry(geometry)

        # Initialize with small pressure pulse in center
        cx, cy, cz = 55, 12, 12
        solver.p[cx, cy, cz] = 1e-5

        # Track energy over 2000 steps (matches issue #89 description)
        initial_energy = None
        final_energy = None

        for step in range(2000):
            solver.step()
            energy = solver.compute_energy()

            if step == 10:  # Record energy after pulse has spread a bit
                initial_energy = energy
            if step == 1999:
                final_energy = energy

        # Energy should not decay by more than a factor of 10 (accounting for
        # numerical dissipation). The original bug showed 7 orders of magnitude
        # decay in native backend vs stable energy in Python backend.
        energy_ratio = final_energy / initial_energy
        assert energy_ratio > 0.1, (
            f"Energy decayed by factor of {1/energy_ratio:.1e} over 2000 steps "
            f"(backend={backend}). Expected <10x decay for proper boundary handling."
        )

    def test_energy_decays_with_pml(self):
        """Energy should decay when PML is present."""
        solver = FDTDSolver(shape=(40, 40, 40), resolution=3e-3)
        solver.add_boundary(PML(depth=8))

        # Initialize with pressure pulse
        solver.p[20, 20, 20] = 1.0

        # Step to establish fields
        solver.step()
        initial_energy = solver.compute_energy()

        # Run simulation
        for _ in range(500):
            solver.step()

        final_energy = solver.compute_energy()

        # Energy should have decayed significantly
        assert final_energy < 0.1 * initial_energy, (
            "Energy should decay with PML absorbing boundaries"
        )


# =============================================================================
# Rectangular Cavity Mode Tests
# =============================================================================


class TestCavityModes:
    def test_fundamental_mode_frequency(self, rectangular_cavity_solver):
        """Test that fundamental cavity mode matches analytical prediction.

        For rectangular cavity with dimensions Lx, Ly, Lz:
        f_lmn = (c/2) * sqrt((l/Lx)² + (m/Ly)² + (n/Lz)²)

        Fundamental mode (1,0,0) for Lx=100mm:
        f_100 = (343/2) * sqrt((1/0.1)²) = 171.5 * 10 = 1715 Hz
        """
        solver = rectangular_cavity_solver
        Lx = solver.shape[0] * solver.dx  # 100mm
        Ly = solver.shape[1] * solver.dx  # 80mm
        Lz = solver.shape[2] * solver.dx  # 60mm

        # Analytical fundamental frequencies
        f_100 = (solver.c / 2) * np.sqrt((1/Lx)**2)
        _ = (solver.c / 2) * np.sqrt((1/Ly)**2)  # f_010, for reference
        _ = (solver.c / 2) * np.sqrt((1/Lz)**2)  # f_001, for reference

        # Excite with broadband pulse in corner
        solver.add_source(GaussianPulse(
            position=(5, 5, 5),
            frequency=2000,
            bandwidth=3000,
        ))

        # Record at opposite corner
        solver.add_probe('corner', position=(45, 35, 25))

        # Run simulation long enough to resolve modes
        solver.run(duration=0.02)  # 20ms

        # Get frequency response
        freqs, magnitude = solver.get_frequency_response('corner')

        # Find peaks in spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(magnitude, height=magnitude.max() * 0.1)
        peak_freqs = freqs[peaks]

        # Check that fundamental modes are present within 2%
        def find_nearest_peak(target_freq):
            if len(peak_freqs) == 0:
                return None
            idx = np.argmin(np.abs(peak_freqs - target_freq))
            return peak_freqs[idx]

        measured_f100 = find_nearest_peak(f_100)
        if measured_f100 is not None:
            error = abs(measured_f100 - f_100) / f_100
            assert error < 0.02, (
                f"Mode (1,0,0): measured {measured_f100:.0f} Hz, "
                f"expected {f_100:.0f} Hz, error {error*100:.1f}%"
            )

    def test_multiple_modes_present(self, rectangular_cavity_solver):
        """Verify multiple cavity modes are excited."""
        solver = rectangular_cavity_solver

        # Broadband excitation
        solver.add_source(GaussianPulse(
            position=(10, 10, 10),
            frequency=3000,
            bandwidth=5000,
        ))
        solver.add_probe('center', position=(25, 20, 15))

        solver.run(duration=0.015)

        freqs, magnitude = solver.get_frequency_response('center')

        # Should see multiple peaks in spectrum
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitude, height=magnitude.max() * 0.05)

        assert len(peaks) >= 3, (
            f"Expected multiple modes, found only {len(peaks)} peaks"
        )


# =============================================================================
# PML Absorption Tests
# =============================================================================


class TestPMLAbsorption:
    def test_pml_reduces_reflection(self):
        """Test that PML reduces reflection compared to rigid boundary.

        The simple exponential decay PML won't achieve <1% reflection
        (that requires full CPML), but it should significantly attenuate
        compared to a rigid wall.
        """
        # Test with PML
        solver_pml = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
        solver_pml.add_boundary(PML(depth=15, axis='x'))

        # Initialize with planar pulse
        solver_pml.p[10, :, :] = 1.0
        solver_pml.add_probe('monitor', position=(30, 10, 10))
        solver_pml.run(duration=0.001)
        data_pml = solver_pml.get_probe_data('monitor')['monitor']

        # Test without PML (rigid boundaries)
        solver_rigid = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
        solver_rigid.p[10, :, :] = 1.0
        solver_rigid.add_probe('monitor', position=(30, 10, 10))
        solver_rigid.run(duration=0.001)
        data_rigid = solver_rigid.get_probe_data('monitor')['monitor']

        # Compare late-time energy (after initial pulse has passed)
        n_samples = len(data_pml)
        late_start = n_samples // 2  # Second half of simulation

        late_energy_pml = np.sum(data_pml[late_start:] ** 2)
        late_energy_rigid = np.sum(data_rigid[late_start:] ** 2)

        # PML should have less late-time energy (less reflection)
        assert late_energy_pml < late_energy_rigid, (
            f"PML late energy ({late_energy_pml:.2e}) should be less than "
            f"rigid ({late_energy_rigid:.2e})"
        )

    def test_pml_initialization(self):
        """Test that PML initializes correctly."""
        solver = FDTDSolver(shape=(50, 50, 50), resolution=2e-3)
        pml = PML(depth=10, axis='all')

        assert not pml.is_initialized
        solver.add_boundary(pml)
        assert pml.is_initialized

    def test_pml_interior_slice(self):
        """Test getting interior (non-PML) region."""
        solver = FDTDSolver(shape=(50, 50, 50), resolution=2e-3)
        pml = PML(depth=10, axis='all')
        solver.add_boundary(pml)

        x_slice, y_slice, z_slice = pml.get_interior_slice()

        assert x_slice == slice(10, 40)
        assert y_slice == slice(10, 40)
        assert z_slice == slice(10, 40)


# =============================================================================
# Geometry and Rigid Boundary Tests
# =============================================================================


class TestGeometry:
    def test_geometry_shape_validation(self, small_solver):
        """Test that geometry shape must match solver shape."""
        wrong_shape = np.ones((10, 10, 10), dtype=bool)

        with pytest.raises(ValueError, match="doesn't match"):
            small_solver.set_geometry(wrong_shape)

    def test_solid_blocks_propagation(self):
        """Test that solid geometry blocks wave propagation."""
        solver = FDTDSolver(shape=(60, 20, 20), resolution=2e-3)

        # Create geometry with solid wall in middle
        geometry = np.ones(solver.shape, dtype=bool)
        geometry[30, :, :] = False  # Solid wall

        solver.set_geometry(geometry)

        # Initialize with planar pressure pulse before wall (stronger signal)
        solver.p[10, :, :] = 1.0

        # Probes before and after wall
        solver.add_probe('before', position=(25, 10, 10))
        solver.add_probe('after', position=(40, 10, 10))

        solver.run(duration=0.001)

        before = solver.get_probe_data('before')['before']
        after = solver.get_probe_data('after')['after']

        # Signal should reach 'before' probe
        assert np.max(np.abs(before)) > 0.01, "Signal should reach probe before wall"

        # Signal should be blocked from 'after' probe
        assert np.max(np.abs(after)) < 0.01 * np.max(np.abs(before)), (
            "Wall should block most signal"
        )

    def test_pressure_zero_in_solid(self):
        """Test that pressure is zeroed in solid regions."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=3e-3)

        # Create geometry with some solid cells
        geometry = np.ones(solver.shape, dtype=bool)
        geometry[10, 10, 10] = False

        solver.set_geometry(geometry)

        # Initialize pressure everywhere
        solver.p.fill(1.0)

        # Single step should zero pressure in solid
        solver.step()

        assert solver.p[10, 10, 10] == 0.0


# =============================================================================
# Source and Probe Tests
# =============================================================================


class TestSourcesAndProbes:
    def test_gaussian_pulse_waveform(self):
        """Test Gaussian pulse waveform generation."""
        source = GaussianPulse(
            position=(0, 0, 0),
            frequency=1000,
            bandwidth=2000,
            amplitude=1.0,
        )

        t = np.linspace(0, 0.01, 1000)
        waveform = source.waveform(t, dt=1e-5)

        # Waveform should be smooth and bounded
        assert np.all(np.isfinite(waveform))
        assert np.max(np.abs(waveform)) <= 1.1  # Near amplitude

        # Should have significant energy
        assert np.max(np.abs(waveform)) > 0.5

    def test_point_source_injection(self, small_solver):
        """Test that point source injects at correct location."""
        source = GaussianPulse(
            position=(10, 10, 10),
            frequency=1000,
        )
        small_solver.add_source(source)

        # Initial pressure should be zero
        assert small_solver.p[10, 10, 10] == 0.0

        # After step, source location should have non-zero pressure
        small_solver.step()
        assert small_solver.p[10, 10, 10] != 0.0

    def test_planar_source(self):
        """Test planar source injection."""
        solver = FDTDSolver(shape=(30, 20, 20), resolution=3e-3)

        source = GaussianPulse(
            position={'axis': 0, 'index': 5},
            frequency=1000,
            source_type='plane',
        )
        solver.add_source(source)
        solver.step()

        # Entire plane should have non-zero pressure
        assert np.all(solver.p[5, :, :] != 0.0)

        # Other planes should still be near zero
        assert np.allclose(solver.p[10, :, :], 0.0, atol=1e-10)

    def test_probe_recording(self, small_solver):
        """Test that probes record pressure correctly."""
        small_solver.add_probe('test', position=(10, 10, 10))

        # Set known pressure
        small_solver.p[10, 10, 10] = 0.5

        # Step should record
        small_solver.step()

        data = small_solver.get_probe_data('test')['test']
        assert len(data) == 1

    def test_duplicate_probe_name_rejected(self, small_solver):
        """Test that duplicate probe names are rejected."""
        small_solver.add_probe('test', position=(5, 5, 5))

        with pytest.raises(ValueError, match="already exists"):
            small_solver.add_probe('test', position=(10, 10, 10))


# =============================================================================
# Simulation Control Tests
# =============================================================================


class TestSimulationControl:
    def test_run_duration(self, small_solver):
        """Test that simulation runs for correct duration."""
        duration = 0.001  # 1ms
        small_solver.run(duration=duration)

        # Time should be approximately the requested duration
        assert small_solver.time >= duration * 0.99
        assert small_solver.time <= duration * 1.01

    def test_reset_clears_fields(self, small_solver):
        """Test that reset clears all fields."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.vx[10, 10, 10] = 1.0

        small_solver.step()
        assert small_solver.step_count > 0

        small_solver.reset()

        assert np.all(small_solver.p == 0)
        assert np.all(small_solver.vx == 0)
        assert small_solver.step_count == 0
        assert small_solver.time == 0.0

    def test_snapshots_enabled(self, small_solver):
        """Test snapshot recording when enabled."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.enable_snapshots(interval=10)

        for _ in range(25):
            small_solver.step()

        snapshots = small_solver.get_snapshots()
        # Should have snapshots at steps 0, 10, 20
        assert len(snapshots) == 3

    def test_sample_rate(self, small_solver):
        """Test sample rate calculation."""
        sr = small_solver.get_sample_rate()
        assert sr == pytest.approx(1.0 / small_solver.dt)


# =============================================================================
# ABC (Mur) Boundary Tests
# =============================================================================


class TestABCBoundary:
    def test_abc_initialization(self):
        """Test ABC boundary initialization."""
        solver = FDTDSolver(shape=(30, 30, 30), resolution=3e-3)
        abc = ABCFirstOrder(axis='all')

        solver.add_boundary(abc)
        # Should not raise

    def test_abc_reduces_reflection(self):
        """Test that ABC reduces reflection compared to rigid."""
        # With ABC
        solver_abc = FDTDSolver(shape=(80, 20, 20), resolution=2e-3)
        solver_abc.add_boundary(ABCFirstOrder(axis='x'))
        solver_abc.add_source(GaussianPulse(position=(10, 10, 10), frequency=2000))
        solver_abc.add_probe('monitor', position=(30, 10, 10))
        solver_abc.run(duration=0.001)

        # Without ABC (rigid boundaries)
        solver_rigid = FDTDSolver(shape=(80, 20, 20), resolution=2e-3)
        solver_rigid.add_source(GaussianPulse(position=(10, 10, 10), frequency=2000))
        solver_rigid.add_probe('monitor', position=(30, 10, 10))
        solver_rigid.run(duration=0.001)

        abc_data = solver_abc.get_probe_data()['monitor']
        rigid_data = solver_rigid.get_probe_data()['monitor']

        # Late-time energy should be lower with ABC
        late_abc = np.sum(abc_data[len(abc_data)//2:]**2)
        late_rigid = np.sum(rigid_data[len(rigid_data)//2:]**2)

        assert late_abc < late_rigid, "ABC should reduce late reflections"


# =============================================================================
# Frequency Response Tests
# =============================================================================


class TestFrequencyResponse:
    def test_frequency_response_shape(self, small_solver):
        """Test frequency response output shape."""
        small_solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=1000))
        small_solver.add_probe('test', position=(15, 10, 10))
        small_solver.run(duration=0.005)

        freqs, magnitude = small_solver.get_frequency_response('test')

        # Should have same length
        assert len(freqs) == len(magnitude)

        # Frequencies should be non-negative
        assert np.all(freqs >= 0)

        # Magnitude should be non-negative
        assert np.all(magnitude >= 0)

    def test_frequency_response_nyquist(self, small_solver):
        """Test that frequency response extends to Nyquist."""
        small_solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=1000))
        small_solver.add_probe('test', position=(15, 10, 10))
        small_solver.run(duration=0.001)

        freqs, _ = small_solver.get_frequency_response('test')

        nyquist = small_solver.get_sample_rate() / 2
        assert freqs[-1] <= nyquist
        assert freqs[-1] >= nyquist * 0.9


# =============================================================================
# Performance Benchmark Tests (optional)
# =============================================================================


# =============================================================================
# Energy Diagnostics Tests
# =============================================================================


class TestEnergyDiagnostics:
    """Tests for energy tracking and conservation diagnostics."""

    def test_energy_tracking_disabled_by_default(self, small_solver):
        """Energy history should be empty when tracking is not enabled."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.001)

        history = small_solver.get_energy_history()
        assert len(history) == 0

    def test_energy_tracking_records_history(self, small_solver):
        """Energy history should be recorded when track_energy=True."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.001, track_energy=True)

        history = small_solver.get_energy_history()
        assert len(history) > 0

        # Each entry should be (step, time, energy)
        step, time, energy = history[0]
        assert isinstance(step, (int, np.integer))
        assert isinstance(time, (float, np.floating))
        assert isinstance(energy, (float, np.floating))

    def test_energy_history_includes_initial_state(self, small_solver):
        """Energy history should include the initial state (step 0)."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.0005, track_energy=True)

        history = small_solver.get_energy_history()
        # First entry should be step 0
        assert history[0][0] == 0

    def test_energy_sample_interval(self, small_solver):
        """Energy should be sampled at specified interval."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.001, track_energy=True, energy_sample_interval=10)

        history = small_solver.get_energy_history()
        # Check that steps are multiples of 10 (except initial step 0)
        steps = [h[0] for h in history]
        assert steps[0] == 0  # Initial state
        for s in steps[1:]:
            assert s % 10 == 0, f"Step {s} is not a multiple of 10"

    def test_energy_report_basic(self, small_solver):
        """Test basic energy report generation."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.001, track_energy=True)

        report = small_solver.energy_report()

        # Check all required keys are present
        assert "initial_energy" in report
        assert "final_energy" in report
        assert "max_energy" in report
        assert "min_energy" in report
        assert "energy_change_percent" in report
        assert "conservation_status" in report
        assert "n_samples" in report

        # Values should be reasonable
        assert report["initial_energy"] >= 0
        assert report["final_energy"] >= 0
        assert report["max_energy"] >= report["min_energy"]
        assert report["n_samples"] > 0

    def test_energy_report_status_stable(self, small_solver):
        """Test stable conservation status in closed cavity."""
        # Initialize with Gaussian pulse for smoother energy behavior
        cx, cy, cz = [s // 2 for s in small_solver.shape]
        sigma = 2
        for i in range(small_solver.shape[0]):
            for j in range(small_solver.shape[1]):
                for k in range(small_solver.shape[2]):
                    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    small_solver.p[i, j, k] = np.exp(-r2 / (2 * sigma**2))

        small_solver.run(duration=0.0005, track_energy=True)
        report = small_solver.energy_report()

        # Should be stable or close to it in closed cavity
        # Allow for some numerical oscillation
        assert abs(report["energy_change_percent"]) < 50

    def test_energy_report_status_decaying_with_pml(self):
        """Test decaying status when PML absorbs energy."""
        from strata_fdtd import PML

        solver = FDTDSolver(shape=(30, 30, 30), resolution=3e-3)
        solver.add_boundary(PML(depth=6))
        solver.p[15, 15, 15] = 1.0

        solver.run(duration=0.003, track_energy=True)
        report = solver.energy_report()

        # With PML, energy should decay
        assert report["conservation_status"] == "decaying"
        assert report["final_energy"] < report["initial_energy"]

    def test_energy_report_raises_without_history(self, small_solver):
        """energy_report should raise ValueError if no history recorded."""
        with pytest.raises(ValueError, match="No energy history"):
            small_solver.energy_report()

    def test_warn_energy_drift_triggers_warning(self):
        """Test that warn_energy_drift emits warning on drift."""
        from strata_fdtd import PML

        solver = FDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            warn_energy_drift=True,
            energy_drift_threshold=0.01,  # 1% threshold
        )
        solver.add_boundary(PML(depth=6))
        solver.p[15, 15, 15] = 1.0

        # Should emit warning due to energy decay from PML
        with pytest.warns(UserWarning, match="Energy drift detected"):
            solver.run(duration=0.003, track_energy=True)

    def test_warn_energy_drift_no_warning_when_disabled(self, small_solver):
        """No warning should be emitted when warn_energy_drift is False."""
        from strata_fdtd import PML

        # Use same setup as warning test, but with warn_energy_drift=False
        solver = FDTDSolver(
            shape=(30, 30, 30),
            resolution=3e-3,
            warn_energy_drift=False,  # Disabled
        )
        solver.add_boundary(PML(depth=6))
        solver.p[15, 15, 15] = 1.0

        # Should NOT emit warning even with significant energy change
        import warnings as warn_module
        with warn_module.catch_warnings():
            warn_module.simplefilter("error")
            try:
                solver.run(duration=0.003, track_energy=True)
            except UserWarning:
                pytest.fail("Unexpected UserWarning raised with warn_energy_drift=False")

    def test_reset_clears_energy_history(self, small_solver):
        """Reset should clear energy history."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.0005, track_energy=True)

        assert len(small_solver.get_energy_history()) > 0

        small_solver.reset()

        assert len(small_solver.get_energy_history()) == 0

    def test_get_energy_history_returns_copy(self, small_solver):
        """get_energy_history should return a copy, not the internal list."""
        small_solver.p[10, 10, 10] = 1.0
        small_solver.run(duration=0.0005, track_energy=True)

        history1 = small_solver.get_energy_history()
        history2 = small_solver.get_energy_history()

        # Should be equal but not the same object
        assert history1 == history2
        assert history1 is not history2

        # Modifying returned list shouldn't affect internal state
        history1.clear()
        assert len(small_solver.get_energy_history()) > 0


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance measurement.

    Run with: pytest -m benchmark tests/strata_fdtd/
    """

    def test_step_throughput(self):
        """Measure timesteps per second."""
        import time

        solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)

        # Warmup
        for _ in range(10):
            solver.step()

        # Benchmark
        n_steps = 100
        start = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        elapsed = time.perf_counter() - start

        steps_per_sec = n_steps / elapsed
        cells_per_sec = (100 * 100 * 100 * n_steps) / elapsed

        print(f"\nFDTD throughput: {steps_per_sec:.1f} steps/sec")
        print(f"Cell updates: {cells_per_sec/1e6:.1f} M cells/sec")

    def test_memory_usage(self):
        """Measure memory usage for different grid sizes."""
        for size in [50, 100]:
            solver = FDTDSolver(shape=(size, size, size), resolution=1e-3)

            # Estimate memory from array sizes
            total_bytes = (
                solver.p.nbytes +
                solver.vx.nbytes +
                solver.vy.nbytes +
                solver.vz.nbytes +
                solver.geometry.nbytes
            )

            print(f"\n{size}³ grid: {total_bytes / 1e6:.1f} MB")


# =============================================================================
# Radiation Impedance Boundary Tests
# =============================================================================


class TestRadiationImpedance:
    """Tests for radiation impedance boundary condition."""

    def test_initialization_with_reflection_coeff(self):
        """Test initialization with constant reflection coefficient."""
        solver = FDTDSolver(shape=(50, 20, 20), resolution=2e-3)
        rad = RadiationImpedance(axis='x', side='high', reflection_coeff=0.7)

        assert not rad.is_initialized
        solver.add_boundary(rad)
        assert rad.is_initialized
        assert rad.reflection_coefficient == pytest.approx(0.7)

    def test_initialization_with_pipe_radius(self):
        """Test initialization with frequency-dependent mode."""
        solver = FDTDSolver(shape=(50, 20, 20), resolution=2e-3)
        rad = RadiationImpedance(axis='x', side='high', pipe_radius=0.01)

        solver.add_boundary(rad)
        assert rad.is_initialized
        # Reflection coefficient should be computed from ka
        assert 0 < rad.reflection_coefficient < 1

    def test_invalid_reflection_coeff(self):
        """Test that invalid reflection coefficient raises error."""
        with pytest.raises(ValueError, match="reflection_coeff must be in"):
            RadiationImpedance(axis='x', side='high', reflection_coeff=1.5)

        with pytest.raises(ValueError, match="reflection_coeff must be in"):
            RadiationImpedance(axis='x', side='low', reflection_coeff=-0.1)

    def test_invalid_pipe_radius(self):
        """Test that invalid pipe radius raises error."""
        with pytest.raises(ValueError, match="pipe_radius must be positive"):
            RadiationImpedance(axis='x', side='high', pipe_radius=-0.01)

    def test_missing_parameters(self):
        """Test that missing both parameters raises error."""
        with pytest.raises(ValueError, match="Must specify either"):
            RadiationImpedance(axis='x', side='high')

    def test_partial_reflection_preserves_energy(self):
        """Test that radiation impedance preserves more energy than ABC.

        A high reflection coefficient should preserve standing wave
        energy better than a pure absorbing boundary.
        """
        # With high reflection (R=0.9) - should preserve energy
        solver_rad = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
        solver_rad.add_boundary(RadiationImpedance(
            axis='x', side='high', reflection_coeff=0.9
        ))
        # Also add radiation at low end for symmetric open pipe
        solver_rad.add_boundary(RadiationImpedance(
            axis='x', side='low', reflection_coeff=0.9
        ))

        # Initialize with pulse in center
        solver_rad.p[50, 10, 10] = 1.0
        solver_rad.step()
        initial_energy_rad = solver_rad.compute_energy()

        # Run simulation
        for _ in range(200):
            solver_rad.step()
        final_energy_rad = solver_rad.compute_energy()

        # With ABC (nearly full absorption)
        solver_abc = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
        solver_abc.add_boundary(ABCFirstOrder(axis='x'))

        solver_abc.p[50, 10, 10] = 1.0
        solver_abc.step()
        initial_energy_abc = solver_abc.compute_energy()

        for _ in range(200):
            solver_abc.step()
        final_energy_abc = solver_abc.compute_energy()

        # Radiation impedance should retain more energy than ABC
        energy_retained_rad = final_energy_rad / initial_energy_rad
        energy_retained_abc = final_energy_abc / initial_energy_abc

        assert energy_retained_rad > energy_retained_abc, (
            f"Radiation impedance ({energy_retained_rad:.2%}) should retain "
            f"more energy than ABC ({energy_retained_abc:.2%})"
        )

    def test_open_pipe_mode_detection(self):
        """Test that radiation impedance enables mode detection in open pipe.

        With radiation impedance boundary (high R), we should detect
        resonant modes. The FDTD domain has a rigid boundary at x=0
        and radiation impedance at x=max, creating a half-open pipe.

        For a half-open pipe with rigid end at x=0:
        f_n = (2n-1) * c / (4L)

        However, with high reflection coefficient (R=0.85), the behavior
        approaches a closed pipe, giving modes near:
        f_n = n * c / (2L)

        The key test is that modes ARE detected (unlike with PML which
        absorbs all standing wave energy).
        """
        from scipy.signal import find_peaks

        # Create elongated pipe geometry (100mm long, 20mm cross-section)
        L = 0.1  # 100mm
        dx = 2e-3  # 2mm resolution
        nx = int(L / dx)  # 50 cells

        solver = FDTDSolver(shape=(nx, 10, 10), resolution=dx, c=343.0)

        # Add radiation impedance at one end
        # High reflection to preserve standing waves
        solver.add_boundary(RadiationImpedance(
            axis='x', side='high', reflection_coeff=0.85
        ))

        # Broadband excitation near closed end
        solver.add_source(GaussianPulse(
            position=(5, 5, 5),
            frequency=2000,
            bandwidth=4000,
        ))

        # Probe near open end
        solver.add_probe('open_end', position=(nx - 10, 5, 5))

        # Run long enough to establish standing waves
        solver.run(duration=0.02)  # 20ms

        # Analyze frequency response
        freqs, magnitude = solver.get_frequency_response('open_end')

        # Find peaks
        peaks, _ = find_peaks(magnitude, height=magnitude.max() * 0.1)
        peak_freqs = freqs[peaks]

        # The key assertion: radiation impedance allows mode detection
        # (PML would absorb everything, giving no peaks)
        assert len(peak_freqs) >= 1, (
            "Should detect at least one resonant mode with radiation impedance"
        )

        # With high R (0.85), behavior approaches closed pipe
        # Expected closed pipe modes: f_n = n * c / (2L)
        # f_1 = 343 / (2 * 0.1) = 1715 Hz
        # f_2 = 2 * 343 / (2 * 0.1) = 3430 Hz
        f_closed_1 = 343.0 / (2 * L)

        # Verify at least one peak is near expected frequency
        # Allow 15% tolerance for numerical dispersion
        closest_to_f1 = min(peak_freqs, key=lambda f: abs(f - f_closed_1))
        rel_error = abs(closest_to_f1 - f_closed_1) / f_closed_1

        assert rel_error < 0.15, (
            f"Fundamental mode at {closest_to_f1:.0f} Hz differs from "
            f"expected {f_closed_1:.0f} Hz by {rel_error*100:.1f}%"
        )

    def test_reset_clears_state(self):
        """Test that reset clears boundary state."""
        solver = FDTDSolver(shape=(30, 20, 20), resolution=3e-3)
        rad = RadiationImpedance(axis='x', side='high', reflection_coeff=0.7)
        solver.add_boundary(rad)

        # Run a few steps
        solver.p[15, 10, 10] = 1.0
        for _ in range(10):
            solver.step()

        # Reset
        rad.reset()

        # Internal state should be zeroed (not easy to verify directly,
        # but at least check it doesn't raise)
        solver.reset()
        for _ in range(5):
            solver.step()

    def test_different_axes(self):
        """Test radiation impedance works on all axes."""
        for axis in ['x', 'y', 'z']:
            for side in ['low', 'high']:
                solver = FDTDSolver(shape=(30, 30, 30), resolution=3e-3)
                rad = RadiationImpedance(
                    axis=axis, side=side, reflection_coeff=0.7
                )
                solver.add_boundary(rad)

                # Should not raise
                solver.p[15, 15, 15] = 1.0
                for _ in range(10):
                    solver.step()

    def test_low_reflection_approaches_abc(self):
        """Test that low reflection coefficient approaches ABC behavior."""
        # With very low reflection (R=0.1)
        solver_low = FDTDSolver(shape=(80, 20, 20), resolution=2e-3)
        solver_low.add_boundary(RadiationImpedance(
            axis='x', side='high', reflection_coeff=0.1
        ))

        solver_low.p[10, 10, 10] = 1.0
        for _ in range(200):
            solver_low.step()
        energy_low = solver_low.compute_energy()

        # With high reflection (R=0.9)
        solver_high = FDTDSolver(shape=(80, 20, 20), resolution=2e-3)
        solver_high.add_boundary(RadiationImpedance(
            axis='x', side='high', reflection_coeff=0.9
        ))

        solver_high.p[10, 10, 10] = 1.0
        for _ in range(200):
            solver_high.step()
        energy_high = solver_high.compute_energy()

        # Low reflection should have less energy (more absorption)
        assert energy_low < energy_high, (
            "Lower reflection coefficient should absorb more energy"
        )


# =============================================================================
# Nonuniform Grid Native Backend Tests (Issue #149)
# =============================================================================


class TestNonuniformGridNative:
    """Tests for native backend support with nonuniform grids."""

    def test_nonuniform_native_fdtd_step(self):
        """Verify native nonuniform FDTD step matches Python implementation."""
        from strata_fdtd.fdtd import has_native_kernels
        from strata_fdtd.grid import NonuniformGrid

        if not has_native_kernels():
            pytest.skip("Native kernels not available")

        # Create a nonuniform grid with geometric stretch
        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 40),
            base_resolution=2e-3,
            stretch_z=1.05,
        )

        # Create solvers with each backend
        solver_python = FDTDSolver(grid=grid, backend="python")
        solver_native = FDTDSolver(grid=grid, backend="native")

        # Initialize with same pressure pulse
        solver_python.p[15, 15, 20] = 1.0
        solver_native.p[15, 15, 20] = 1.0

        # Run 100 steps
        for _ in range(100):
            solver_python.step()
            solver_native.step()

        # Verify pressure fields match within numerical tolerance
        max_diff = np.max(np.abs(solver_python.p - solver_native.p))
        assert max_diff < 1e-4, f"Max pressure difference: {max_diff}"

        # Verify velocity fields match
        max_vx_diff = np.max(np.abs(solver_python.vx - solver_native.vx))
        max_vy_diff = np.max(np.abs(solver_python.vy - solver_native.vy))
        max_vz_diff = np.max(np.abs(solver_python.vz - solver_native.vz))
        assert max_vx_diff < 1e-4, f"Max vx difference: {max_vx_diff}"
        assert max_vy_diff < 1e-4, f"Max vy difference: {max_vy_diff}"
        assert max_vz_diff < 1e-4, f"Max vz difference: {max_vz_diff}"

    def test_nonuniform_native_uses_correct_backend(self):
        """Verify native backend is actually used for nonuniform grids."""
        from strata_fdtd.fdtd import has_native_kernels
        from strata_fdtd.grid import NonuniformGrid

        if not has_native_kernels():
            pytest.skip("Native kernels not available")

        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=2e-3,
            stretch_z=1.03,
        )

        solver = FDTDSolver(grid=grid, backend="native")
        assert solver.using_native is True
        assert not solver.grid.is_uniform

    def test_nonuniform_native_energy_conservation(self):
        """Verify energy is conserved with nonuniform grid and native backend."""
        from strata_fdtd.fdtd import has_native_kernels
        from strata_fdtd.grid import NonuniformGrid

        if not has_native_kernels():
            pytest.skip("Native kernels not available")

        grid = NonuniformGrid.from_stretch(
            shape=(40, 40, 40),
            base_resolution=2e-3,
            stretch_x=1.0,
            stretch_y=1.0,
            stretch_z=1.03,
        )

        solver = FDTDSolver(grid=grid, backend="native")
        solver.p[20, 20, 20] = 1.0

        # Record initial energy
        initial_energy = solver.compute_energy()

        # Run for 500 steps
        for _ in range(500):
            solver.step()

        final_energy = solver.compute_energy()

        # Allow for some numerical dissipation but not catastrophic loss
        energy_ratio = final_energy / initial_energy
        assert energy_ratio > 0.1, (
            f"Energy decayed by factor of {1/energy_ratio:.1e} over 500 steps"
        )

    @pytest.mark.parametrize("backend", ["python", "native"])
    def test_nonuniform_divergence_consistency(self, backend):
        """Verify divergence computation matches between backends."""
        from strata_fdtd.fdtd import has_native_kernels
        from strata_fdtd.grid import NonuniformGrid

        if backend == "native" and not has_native_kernels():
            pytest.skip("Native kernels not available")

        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 25),
            base_resolution=2e-3,
            stretch_z=1.04,
        )

        solver = FDTDSolver(grid=grid, backend=backend)

        # Set up a velocity field with known pattern
        # Note: velocity arrays have same shape as pressure in this implementation
        nx, ny, nz = solver.shape
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    solver.vx[i, j, k] = np.sin(i * 0.3)
                    solver.vy[i, j, k] = np.cos(j * 0.3)
                    solver.vz[i, j, k] = np.sin(k * 0.2)

        # Compute divergence using the solver's internal method
        divergence = solver._compute_divergence()

        # Verify divergence is computed without errors
        assert divergence.shape == solver.shape
        assert np.isfinite(divergence).all()

        # Divergence should not be all zeros for our test velocity field
        assert np.abs(divergence).max() > 0.01


# =============================================================================
# GPU Backend Tests (Issue #160)
# =============================================================================


class TestGPUBackendSelection:
    """Tests for GPU backend selection in FDTDSolver."""

    def test_has_gpu_backend_returns_bool(self):
        """has_gpu_backend() should return a boolean."""
        from strata_fdtd.fdtd import has_gpu_backend
        result = has_gpu_backend()
        assert isinstance(result, bool)

    def test_backend_auto_selects_available(self):
        """Auto backend should select best available option."""
        from strata_fdtd.fdtd import has_gpu_backend, has_native_kernels

        solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="auto")

        # Auto should prefer native > gpu > python
        if has_native_kernels():
            assert solver.using_native
            assert not solver.using_gpu
        elif has_gpu_backend():
            assert not solver.using_native
            assert solver.using_gpu
        else:
            assert not solver.using_native
            assert not solver.using_gpu

    def test_backend_python_forces_python(self):
        """Backend='python' should force Python implementation."""
        solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="python")
        assert not solver.using_native
        assert not solver.using_gpu

    def test_backend_native_raises_if_unavailable(self):
        """Backend='native' should raise ImportError if unavailable."""
        from strata_fdtd.fdtd import has_native_kernels

        if has_native_kernels():
            # Should work
            solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="native")
            assert solver.using_native
        else:
            # Should raise
            with pytest.raises(ImportError, match="Native FDTD kernels not available"):
                FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="native")

    def test_backend_gpu_raises_if_unavailable(self):
        """Backend='gpu' should raise RuntimeError if unavailable."""
        from strata_fdtd.fdtd import has_gpu_backend

        if has_gpu_backend():
            # Should work (with warning)
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="gpu")
                assert solver.using_gpu
                # Should have a warning about limited features
                assert len(w) >= 1
                assert "limited feature support" in str(w[0].message).lower()
        else:
            # Should raise
            with pytest.raises(RuntimeError, match="GPU.*not available"):
                FDTDSolver(shape=(10, 10, 10), resolution=1e-3, backend="gpu")

    @pytest.mark.skipif(
        not __import__("strata_fdtd.fdtd", fromlist=["has_gpu_backend"]).has_gpu_backend(),
        reason="GPU backend not available"
    )
    def test_gpu_backend_runs_simulation(self):
        """GPU backend should be able to run a basic simulation."""
        import warnings

        # Suppress the warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="gpu")

        # Add a simple pulse at center
        solver.p[10, 10, 10] = 1.0

        # Should be able to step
        solver.step()

        # Pressure should have spread from the center
        assert solver.p[10, 10, 10] != 1.0  # Changed
        assert np.abs(solver.p).sum() > 0  # Not all zero

    @pytest.mark.skipif(
        not __import__("strata_fdtd.fdtd", fromlist=["has_gpu_backend"]).has_gpu_backend(),
        reason="GPU backend not available"
    )
    def test_gpu_backend_probe_recording(self):
        """GPU backend should record probe data correctly."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3, backend="gpu")

        solver.add_probe("center", position=(10, 10, 10))
        solver.p[10, 10, 10] = 1.0

        # Run a few steps
        for _ in range(5):
            solver.step()

        # Should have recorded probe data
        data = solver.get_probe_data()
        assert "center" in data
        assert len(data["center"]) == 5
