"""
Unit tests for grid specifications (UniformGrid, NonuniformGrid).

Tests verify:
- Grid construction with different methods
- Spacing array computation
- CFL condition with variable spacing
- Backward compatibility with uniform grids
"""

import numpy as np
import pytest

from strata_fdtd import NonuniformGrid, UniformGrid

# =============================================================================
# UniformGrid Tests
# =============================================================================


class TestUniformGrid:
    def test_basic_construction(self):
        """Test basic uniform grid construction."""
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        assert grid.shape == (100, 100, 100)
        assert grid.resolution == 1e-3
        assert grid.min_spacing == 1e-3
        assert grid.max_spacing == 1e-3
        assert grid.is_uniform is True

    def test_spacing_arrays(self):
        """Test that spacing arrays are uniform."""
        grid = UniformGrid(shape=(50, 40, 30), resolution=2e-3)

        assert len(grid.dx) == 50
        assert len(grid.dy) == 40
        assert len(grid.dz) == 30
        assert np.all(grid.dx == 2e-3)
        assert np.all(grid.dy == 2e-3)
        assert np.all(grid.dz == 2e-3)

    def test_coordinate_arrays(self):
        """Test cell center coordinate computation."""
        grid = UniformGrid(shape=(10, 10, 10), resolution=1e-3)

        # Cell centers should be at 0.5*dx, 1.5*dx, 2.5*dx, ...
        expected_x = np.arange(10) * 1e-3 + 0.5e-3
        np.testing.assert_allclose(grid.x_coords, expected_x)

    def test_physical_extent(self):
        """Test physical domain size computation."""
        grid = UniformGrid(shape=(100, 80, 60), resolution=1e-3)

        Lx, Ly, Lz = grid.physical_extent()
        assert Lx == pytest.approx(0.1)
        assert Ly == pytest.approx(0.08)
        assert Lz == pytest.approx(0.06)


# =============================================================================
# NonuniformGrid Tests
# =============================================================================


class TestNonuniformGrid:
    def test_basic_construction(self):
        """Test basic nonuniform grid construction from coordinate arrays."""
        x = np.linspace(0, 0.1, 50)
        y = np.linspace(0, 0.1, 50)
        z = np.geomspace(0.001, 0.1, 100)

        grid = NonuniformGrid(x_coords=x, y_coords=y, z_coords=z)

        assert grid.shape == (50, 50, 100)
        assert grid.is_uniform is False

    def test_min_spacing(self):
        """Test minimum spacing detection."""
        # Create grid with varying spacing
        x = np.array([0, 0.001, 0.003, 0.006])  # dx = [0.001, 0.002, 0.003]
        y = np.linspace(0, 0.1, 10)
        z = np.linspace(0, 0.1, 10)

        grid = NonuniformGrid(x_coords=x, y_coords=y, z_coords=z)

        # Min spacing should come from the smallest cell
        assert grid.min_spacing < grid.max_spacing

    def test_from_stretch_uniform(self):
        """Test that stretch=1.0 produces uniform grid."""
        grid = NonuniformGrid.from_stretch(
            shape=(50, 50, 50),
            base_resolution=1e-3,
            stretch_x=1.0,
            stretch_y=1.0,
            stretch_z=1.0,
        )

        # Should be nearly uniform (small numerical differences allowed)
        assert grid.stretch_ratio[0] == pytest.approx(1.0, rel=0.01)
        assert grid.stretch_ratio[1] == pytest.approx(1.0, rel=0.01)
        assert grid.stretch_ratio[2] == pytest.approx(1.0, rel=0.01)

    def test_from_stretch_variable(self):
        """Test that stretch>1.0 produces variable spacing."""
        grid = NonuniformGrid.from_stretch(
            shape=(100, 100, 100),
            base_resolution=1e-3,
            stretch_x=1.0,
            stretch_y=1.0,
            stretch_z=1.05,  # 5% growth in z
        )

        # x and y should be nearly uniform
        assert grid.stretch_ratio[0] == pytest.approx(1.0, rel=0.05)
        assert grid.stretch_ratio[1] == pytest.approx(1.0, rel=0.05)

        # z should have significant stretch
        assert grid.stretch_ratio[2] > 1.5

    def test_from_stretch_center_fine(self):
        """Test center-fine stretch pattern."""
        grid = NonuniformGrid.from_stretch(
            shape=(100, 10, 10),
            base_resolution=1e-3,
            stretch_x=1.1,  # 10% growth from center
            center_fine=True,
        )

        # Center cells should be smaller than edge cells
        center_dx = grid.dx[50]
        edge_dx_left = grid.dx[0]
        edge_dx_right = grid.dx[-1]

        assert center_dx < edge_dx_left
        assert center_dx < edge_dx_right

    def test_from_regions_single(self):
        """Test grid from single region."""
        grid = NonuniformGrid.from_regions(
            x_regions=[(0, 0.1, 1e-3)],
            y_regions=[(0, 0.1, 1e-3)],
            z_regions=[(0, 0.1, 1e-3)],
        )

        assert grid.shape[0] == 100  # 0.1m / 1e-3 = 100 cells

    def test_from_regions_multiple(self):
        """Test grid from multiple piecewise regions."""
        grid = NonuniformGrid.from_regions(
            x_regions=[
                (0, 0.05, 2e-3),     # Coarse: 25 cells
                (0.05, 0.15, 1e-3),  # Fine: 100 cells
                (0.15, 0.2, 2e-3),   # Coarse: 25 cells
            ],
            y_regions=[(0, 0.1, 1e-3)],
            z_regions=[(0, 0.1, 1e-3)],
        )

        # Total cells should be ~150 in x (some boundary adjustment)
        assert 140 < grid.shape[0] < 160

        # Fine region should have smaller spacing
        fine_idx = grid.shape[0] // 2  # Middle of grid
        coarse_idx_left = 10  # Near start
        coarse_idx_right = grid.shape[0] - 10  # Near end

        assert grid.dx[fine_idx] < grid.dx[coarse_idx_left]
        assert grid.dx[fine_idx] < grid.dx[coarse_idx_right]

    def test_from_regions_contiguity_check(self):
        """Test that non-contiguous regions raise error."""
        with pytest.raises(ValueError, match="contiguous"):
            NonuniformGrid.from_regions(
                x_regions=[
                    (0, 0.05, 1e-3),
                    (0.06, 0.1, 1e-3),  # Gap at 0.05-0.06
                ],
            )

    def test_invalid_coordinates(self):
        """Test that invalid coordinate arrays raise errors."""
        # Non-monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            NonuniformGrid(
                x_coords=np.array([0, 0.02, 0.01, 0.03]),  # Not increasing
                y_coords=np.linspace(0, 0.1, 10),
                z_coords=np.linspace(0, 0.1, 10),
            )

        # Too few points
        with pytest.raises(ValueError, match="at least 2"):
            NonuniformGrid(
                x_coords=np.array([0.05]),  # Only 1 point
                y_coords=np.linspace(0, 0.1, 10),
                z_coords=np.linspace(0, 0.1, 10),
            )

    def test_spacing_arrays_for_stencil(self):
        """Test precomputed stencil arrays."""
        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=1e-3,
            stretch_z=1.05,
        )

        arrays = grid.get_spacing_arrays_for_stencil()

        assert "inv_dx_face" in arrays
        assert "inv_dx_cell" in arrays
        assert "inv_dy_face" in arrays
        assert "inv_dy_cell" in arrays
        assert "inv_dz_face" in arrays
        assert "inv_dz_cell" in arrays

        # Face arrays have n-1 elements
        assert len(arrays["inv_dx_face"]) == 19
        # Cell arrays have n elements
        assert len(arrays["inv_dx_cell"]) == 20

        # Inverse spacing should be positive
        assert np.all(arrays["inv_dx_face"] > 0)
        assert np.all(arrays["inv_dx_cell"] > 0)


# =============================================================================
# FDTDSolver with NonuniformGrid Tests
# =============================================================================


class TestFDTDWithNonuniformGrid:
    def test_solver_with_uniform_grid(self):
        """Test that solver works with explicit UniformGrid."""
        from strata_fdtd import FDTDSolver

        grid = UniformGrid(shape=(20, 20, 20), resolution=5e-3)
        solver = FDTDSolver(grid=grid)

        assert solver.shape == (20, 20, 20)
        assert solver.dx == 5e-3
        assert solver.grid.is_uniform

    def test_solver_with_nonuniform_grid(self):
        """Test that solver works with NonuniformGrid."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=2e-3,
            stretch_z=1.05,
        )
        solver = FDTDSolver(grid=grid)

        assert solver.shape == (30, 30, 30)
        # dx should be minimum spacing
        assert solver.dx == grid.min_spacing
        assert not solver.grid.is_uniform

    def test_backward_compatibility(self):
        """Test that shape+resolution still works."""
        from strata_fdtd import FDTDSolver

        solver = FDTDSolver(shape=(20, 20, 20), resolution=3e-3)

        assert solver.shape == (20, 20, 20)
        assert solver.dx == 3e-3
        assert solver.grid.is_uniform

    def test_cfl_with_nonuniform(self):
        """Test CFL condition uses minimum spacing."""
        from strata_fdtd import FDTDSolver

        # Create grid with varying spacing
        grid = NonuniformGrid.from_stretch(
            shape=(50, 50, 50),
            base_resolution=1e-3,  # Finest resolution
            stretch_z=1.1,
        )

        solver = FDTDSolver(grid=grid)

        # CFL should be based on minimum spacing
        dt_max = grid.min_spacing / (solver.c * np.sqrt(3))
        assert solver.dt <= dt_max

    def test_step_with_nonuniform_grid(self):
        """Test that simulation steps work with nonuniform grid."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=2e-3,
            stretch_z=1.03,
        )
        solver = FDTDSolver(grid=grid)

        # Initialize with pulse
        solver.p[15, 15, 15] = 1.0

        # Step should not raise
        for _ in range(10):
            solver.step()

        # Fields should have evolved
        assert solver.p[15, 15, 15] != 1.0

    def test_energy_conservation_nonuniform(self):
        """Test energy is bounded with nonuniform grid (no PML)."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=3e-3,
            stretch_z=1.02,
        )
        solver = FDTDSolver(grid=grid)

        # Initialize with Gaussian pulse
        cx, cy, cz = 15, 15, 15
        sigma = 2
        for i in range(solver.shape[0]):
            for j in range(solver.shape[1]):
                for k in range(solver.shape[2]):
                    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    solver.p[i, j, k] = np.exp(-r2 / (2 * sigma**2))

        # Track energy
        energies = []
        for _ in range(100):
            solver.step()
            if solver.step_count % 10 == 0:
                energies.append(solver.compute_energy())

        # Energy should stay within factor of 2 (accounting for numerical oscillation)
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        max_deviation = np.max(np.abs(energies - mean_energy)) / mean_energy

        assert max_deviation < 1.0, (
            f"Energy deviates by {max_deviation*100:.1f}% from mean"
        )

    def test_pml_with_nonuniform_grid(self):
        """Test PML works with nonuniform grid."""
        from strata_fdtd import PML, FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(40, 40, 40),
            base_resolution=2e-3,
            stretch_x=1.02,
        )
        solver = FDTDSolver(grid=grid)
        solver.add_boundary(PML(depth=8))

        # Initialize with pulse
        solver.p[20, 20, 20] = 1.0
        solver.step()
        initial_energy = solver.compute_energy()

        # Run simulation
        for _ in range(200):
            solver.step()

        final_energy = solver.compute_energy()

        # Energy should decay with PML
        assert final_energy < 0.5 * initial_energy

    def test_native_backend_with_nonuniform(self):
        """Test that native backend works with nonuniform grid (Issue #131)."""
        from strata_fdtd import FDTDSolver
        from strata_fdtd.core.solver import has_native_kernels

        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=3e-3,
            stretch_z=1.05,
        )

        if has_native_kernels():
            # Native backend should be used without warning
            solver = FDTDSolver(grid=grid, backend="native")
            assert solver.using_native
        else:
            # If native not available, should raise ImportError
            import pytest
            with pytest.raises(ImportError):
                FDTDSolver(grid=grid, backend="native")


# =============================================================================
# Wave Propagation Tests with Nonuniform Grid
# =============================================================================


class TestWavePropagationNonuniform:
    def test_wave_reaches_far_probe(self):
        """Test that waves propagate correctly in nonuniform grid."""
        from strata_fdtd import FDTDSolver, GaussianPulse

        # Create stretched grid
        grid = NonuniformGrid.from_stretch(
            shape=(100, 20, 20),
            base_resolution=2e-3,
            stretch_x=1.01,
            center_fine=True,
        )
        solver = FDTDSolver(grid=grid)

        # Add source and probes
        solver.add_source(GaussianPulse(position=(10, 10, 10), frequency=2000))
        solver.add_probe('near', position=(20, 10, 10))
        solver.add_probe('far', position=(80, 10, 10))

        # Run simulation
        solver.run(duration=0.001)

        near_data = solver.get_probe_data('near')['near']
        far_data = solver.get_probe_data('far')['far']

        # Both probes should have received signal
        assert np.max(np.abs(near_data)) > 0
        assert np.max(np.abs(far_data)) > 0

        # Near probe should have larger signal (less dispersion)
        assert np.max(np.abs(near_data)) > np.max(np.abs(far_data))


# =============================================================================
# Native Nonuniform Kernel Tests (Issue #131)
# =============================================================================


class TestNativeNonuniformKernels:
    """Tests for native C++ nonuniform grid kernels.

    These tests verify that native kernels produce results consistent with
    the Python implementation for nonuniform grids.
    """

    @pytest.fixture
    def skip_without_native(self):
        """Skip test if native kernels are not available."""
        from strata_fdtd.core.solver import has_native_kernels

        if not has_native_kernels():
            pytest.skip("Native kernels not available")

    def test_native_python_consistency(self, skip_without_native):
        """Test native and Python backends produce consistent results."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=2e-3,
            stretch_z=1.05,
        )

        # Create two solvers with different backends
        solver_native = FDTDSolver(grid=grid, backend="native")
        solver_python = FDTDSolver(grid=grid, backend="python")

        # Initialize with same pulse
        solver_native.p[15, 15, 15] = 1.0
        solver_python.p[15, 15, 15] = 1.0

        # Run same number of steps
        for _ in range(50):
            solver_native.step()
            solver_python.step()

        # Results should be very close (within float32 precision after many steps)
        # Note: Float32 accumulation across 50 steps can lead to ~3% relative error
        np.testing.assert_allclose(
            solver_native.p, solver_python.p,
            rtol=0.05, atol=1e-7,
            err_msg="Pressure fields differ between native and Python"
        )
        np.testing.assert_allclose(
            solver_native.vx, solver_python.vx,
            rtol=0.05, atol=1e-7,
            err_msg="Velocity vx fields differ between native and Python"
        )

    def test_native_energy_conservation(self, skip_without_native):
        """Test energy is conserved with native nonuniform kernels."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=3e-3,
            stretch_z=1.02,
        )
        solver = FDTDSolver(grid=grid, backend="native")

        # Initialize with Gaussian pulse
        cx, cy, cz = 15, 15, 15
        sigma = 2
        for i in range(solver.shape[0]):
            for j in range(solver.shape[1]):
                for k in range(solver.shape[2]):
                    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                    solver.p[i, j, k] = np.exp(-r2 / (2 * sigma**2))

        # Track energy
        solver.run(duration=0.002, track_energy=True, energy_sample_interval=10)
        report = solver.energy_report()

        # Energy should be roughly conserved (within 50%)
        assert abs(report["energy_change_percent"]) < 50, (
            f"Energy changed by {report['energy_change_percent']:.1f}%"
        )

    def test_native_with_geometry(self, skip_without_native):
        """Test native nonuniform kernels with solid geometry."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(40, 40, 40),
            base_resolution=2e-3,
            stretch_x=1.03,
        )
        solver = FDTDSolver(grid=grid, backend="native")

        # Create a box-shaped solid region
        geometry = np.ones(solver.shape, dtype=bool)
        geometry[10:30, 10:30, 10:30] = False  # Solid box
        solver.set_geometry(geometry)

        # Initialize with pulse outside solid
        solver.p[5, 20, 20] = 1.0

        # Run simulation
        for _ in range(50):
            solver.step()

        # Pressure inside solid should remain zero
        assert np.allclose(solver.p[15:25, 15:25, 15:25], 0.0)

    def test_native_with_pml(self, skip_without_native):
        """Test native nonuniform kernels with PML boundaries."""
        from strata_fdtd import PML, FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(50, 50, 50),
            base_resolution=2e-3,
            stretch_z=1.02,
        )
        solver = FDTDSolver(grid=grid, backend="native")
        solver.add_boundary(PML(depth=10))

        # Initialize with pulse
        solver.p[25, 25, 25] = 1.0
        solver.step()
        initial_energy = solver.compute_energy()

        # Run simulation
        for _ in range(200):
            solver.step()

        final_energy = solver.compute_energy()

        # PML should absorb energy
        assert final_energy < 0.3 * initial_energy, (
            f"Energy not absorbed: {final_energy / initial_energy * 100:.1f}% remaining"
        )

    def test_native_with_stretched_all_axes(self, skip_without_native):
        """Test native kernels with stretch on all axes."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_stretch(
            shape=(30, 30, 30),
            base_resolution=2e-3,
            stretch_x=1.03,
            stretch_y=1.02,
            stretch_z=1.04,
            center_fine=True,
        )

        solver_native = FDTDSolver(grid=grid, backend="native")
        solver_python = FDTDSolver(grid=grid, backend="python")

        # Initialize
        solver_native.p[15, 15, 15] = 1.0
        solver_python.p[15, 15, 15] = 1.0

        # Run
        for _ in range(30):
            solver_native.step()
            solver_python.step()

        # Should match (within float32 precision after multiple steps)
        np.testing.assert_allclose(
            solver_native.p, solver_python.p,
            rtol=0.01, atol=1e-7
        )

    def test_native_from_regions_grid(self, skip_without_native):
        """Test native kernels with region-based grid."""
        from strata_fdtd import FDTDSolver

        grid = NonuniformGrid.from_regions(
            x_regions=[
                (0, 0.03, 2e-3),   # Coarse
                (0.03, 0.06, 1e-3),  # Fine
                (0.06, 0.09, 2e-3),  # Coarse
            ],
            y_regions=[(0, 0.06, 2e-3)],
            z_regions=[(0, 0.06, 2e-3)],
        )

        solver_native = FDTDSolver(grid=grid, backend="native")
        solver_python = FDTDSolver(grid=grid, backend="python")

        # Initialize in fine region
        center = solver_native.shape[0] // 2
        solver_native.p[center, 15, 15] = 1.0
        solver_python.p[center, 15, 15] = 1.0

        # Run
        for _ in range(30):
            solver_native.step()
            solver_python.step()

        # Results should match (within float32 precision after multiple steps)
        np.testing.assert_allclose(
            solver_native.p, solver_python.p,
            rtol=0.01, atol=1e-7
        )
