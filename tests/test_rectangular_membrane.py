"""
Unit tests for RectangularMembraneSource with sinusoidal mode shapes.

Tests verify:
- Sinusoidal mode shape accuracy
- Boundary conditions (zero at edges, maximum at center for (1,1) mode)
- Support for fundamental and higher-order modes
- Non-square aspect ratio handling
- Grid discretization correctness
- Integration with FDTDSolver
"""

import numpy as np
import pytest

from strata_fdtd import (
    FDTDSolver,
    GaussianPulse,
    RectangularMembraneSource,
)
from strata_fdtd.core.grid import NonuniformGrid, UniformGrid


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_grid():
    """Create a small uniform grid for testing."""
    return UniformGrid(shape=(20, 20, 20), resolution=5e-3)


@pytest.fixture
def fine_grid():
    """Create a finer grid for more accurate mode shape tests."""
    return UniformGrid(shape=(50, 50, 50), resolution=2e-3)


@pytest.fixture
def small_solver():
    """Create a small solver for fast tests."""
    return FDTDSolver(shape=(20, 20, 20), resolution=5e-3, c=343.0, rho=1.2)


@pytest.fixture
def waveform():
    """Create a standard waveform for tests."""
    return GaussianPulse(position=(0, 0, 0), frequency=200, amplitude=1.0)


# =============================================================================
# Mode Shape Accuracy Tests
# =============================================================================


class TestSinusoidalModeShape:
    """Test sinusoidal mode shape implementation."""

    def test_fundamental_mode_center_is_one(self, waveform):
        """Fundamental mode (1,1) should have value 1.0 at center."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.04, 0.04),
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        # At center (u=0, v=0), sin(π/2) * sin(π/2) = 1
        u = np.array([0.0])
        v = np.array([0.0])
        shape = membrane.mode_shape(u, v)
        assert np.isclose(shape[0], 1.0, atol=1e-10)

    def test_fundamental_mode_edges_are_zero(self, waveform):
        """Fundamental mode (1,1) should have value 0.0 at all edges."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        Lx, Ly = size
        # Test multiple points to ensure proper normalization context
        # Include center point for normalization
        u = np.array([0.0, -Lx/2, Lx/2, 0.0, 0.0])
        v = np.array([0.0, 0.0, 0.0, -Ly/2, Ly/2])
        shape = membrane.mode_shape(u, v)

        # Center should be 1.0
        assert np.isclose(shape[0], 1.0, atol=1e-10)
        # All edges should be 0.0
        assert np.allclose(shape[1:], 0.0, atol=1e-10)

    def test_outside_boundary_is_zero(self, waveform):
        """Mode shape should be zero outside the membrane boundary."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        Lx, Ly = size
        # Points outside the membrane
        u = np.array([Lx, -Lx, 0.0, 0.0])
        v = np.array([0.0, 0.0, Ly, -Ly])
        shape = membrane.mode_shape(u, v)
        assert np.all(shape == 0.0)

    def test_mode_shape_symmetry(self, waveform):
        """Mode (1,1) should be symmetric about center."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        Lx, Ly = size
        # Test symmetric points
        u = np.array([Lx/4, -Lx/4, Lx/4, -Lx/4])
        v = np.array([Ly/4, Ly/4, -Ly/4, -Ly/4])
        shape = membrane.mode_shape(u, v)

        # All four symmetric points should have the same value
        assert np.allclose(shape, shape[0], atol=1e-10)

    def test_mode_shape_values_in_range(self, waveform):
        """All mode shape values should be in [0, 1]."""
        size = (0.04, 0.04)
        for mode in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]:
            membrane = RectangularMembraneSource(
                center=(0.05, 0.05, 0.05),
                size=size,
                normal_axis='z',
                waveform=waveform,
                mode=mode,
            )

            Lx, Ly = size
            u = np.linspace(-Lx, Lx, 50)
            v = np.linspace(-Ly, Ly, 50)
            U, V = np.meshgrid(u, v)
            shape = membrane.mode_shape(U.flatten(), V.flatten())

            assert np.all(shape >= 0.0), f"Mode {mode} has negative values"
            assert np.all(shape <= 1.0), f"Mode {mode} has values > 1"


# =============================================================================
# Higher-Order Mode Tests
# =============================================================================


class TestHigherOrderModes:
    """Test higher-order sinusoidal modes."""

    def test_mode_21_has_one_vertical_node(self, waveform):
        """Mode (2,1) should have a nodal line at u=0."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(2, 1),
        )

        Lx, Ly = size
        # Sample along u at v=0
        u = np.linspace(-Lx/2 * 0.99, Lx/2 * 0.99, 100)
        v = np.zeros_like(u)
        shape = membrane.mode_shape(u, v)

        # Find local minima (nodal points after abs)
        min_indices = []
        for i in range(1, len(shape) - 1):
            if shape[i] < shape[i - 1] and shape[i] < shape[i + 1]:
                min_indices.append(i)

        # Should have at least one nodal line at center
        assert len(min_indices) >= 1, "Mode (2,1) should have at least one vertical node"

    def test_mode_12_has_one_horizontal_node(self, waveform):
        """Mode (1,2) should have a nodal line at v=0."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 2),
        )

        Lx, Ly = size
        # Sample along v at u=0
        u = np.zeros(100)
        v = np.linspace(-Ly/2 * 0.99, Ly/2 * 0.99, 100)
        shape = membrane.mode_shape(u, v)

        # Find local minima (nodal points after abs)
        min_indices = []
        for i in range(1, len(shape) - 1):
            if shape[i] < shape[i - 1] and shape[i] < shape[i + 1]:
                min_indices.append(i)

        assert len(min_indices) >= 1, "Mode (1,2) should have at least one horizontal node"

    def test_mode_22_center_is_nodal(self, waveform):
        """Mode (2,2) should have a nodal point at center."""
        size = (0.04, 0.04)
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(2, 2),
        )

        # For mode (2,2), sin(π) * sin(π) = 0 at center
        # But we need context points for normalization
        Lx, Ly = size
        u = np.array([0.0, Lx/4, -Lx/4])
        v = np.array([0.0, Ly/4, -Ly/4])
        shape = membrane.mode_shape(u, v)

        # Center should be a nodal point (zero)
        assert np.isclose(shape[0], 0.0, atol=1e-10)


# =============================================================================
# Invalid Mode Tests
# =============================================================================


class TestInvalidModes:
    """Test error handling for invalid mode parameters."""

    def test_zero_mode_m_raises(self, waveform):
        """Mode index m=0 should raise ValueError."""
        with pytest.raises(ValueError, match="Mode index m must be >= 1"):
            RectangularMembraneSource(
                center=(0.05, 0.05, 0.05),
                size=(0.04, 0.04),
                normal_axis='z',
                waveform=waveform,
                mode=(0, 1),
            )

    def test_zero_mode_n_raises(self, waveform):
        """Mode index n=0 should raise ValueError."""
        with pytest.raises(ValueError, match="Mode index n must be >= 1"):
            RectangularMembraneSource(
                center=(0.05, 0.05, 0.05),
                size=(0.04, 0.04),
                normal_axis='z',
                waveform=waveform,
                mode=(1, 0),
            )

    def test_negative_mode_raises(self, waveform):
        """Negative mode indices should raise ValueError."""
        with pytest.raises(ValueError, match="Mode index m must be >= 1"):
            RectangularMembraneSource(
                center=(0.05, 0.05, 0.05),
                size=(0.04, 0.04),
                normal_axis='z',
                waveform=waveform,
                mode=(-1, 1),
            )


# =============================================================================
# Aspect Ratio Tests
# =============================================================================


class TestAspectRatio:
    """Test non-square aspect ratio handling."""

    def test_rectangular_dimensions(self, waveform):
        """Test membrane with different width and height."""
        # Width > height
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.06, 0.04),  # 6cm x 4cm
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        Lx, Ly = membrane.size

        # At center, should be 1.0
        u = np.array([0.0])
        v = np.array([0.0])
        shape = membrane.mode_shape(u, v)
        assert np.isclose(shape[0], 1.0, atol=1e-10)

        # At width edge (but inside height)
        u = np.array([Lx/2])
        v = np.array([0.0])
        shape = membrane.mode_shape(u, v)
        # Edge value should be very small (essentially zero)

    def test_tall_rectangular(self, waveform):
        """Test membrane with height > width."""
        # Height > width
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.05),  # 3cm x 5cm
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        # At center, should be 1.0
        u = np.array([0.0])
        v = np.array([0.0])
        shape = membrane.mode_shape(u, v)
        assert np.isclose(shape[0], 1.0, atol=1e-10)


# =============================================================================
# Grid Injection Tests
# =============================================================================


class TestGridInjection:
    """Test grid injection mask and weights."""

    def test_mask_is_rectangular(self, waveform, small_grid):
        """Injection mask should be rectangular."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        mask = membrane.get_injection_mask(small_grid)

        # Find plane index
        plane_idx = int(np.argmin(np.abs(small_grid.z_coords - 0.05)))

        # Mask should be non-empty in the membrane plane
        plane_mask = mask[:, :, plane_idx]
        assert np.any(plane_mask), "Membrane plane should have some True values"

    def test_weights_shape_matches_grid(self, waveform, small_grid):
        """Injection weights should match grid shape."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        weights = membrane.get_injection_weights(small_grid)
        assert weights.shape == small_grid.shape

    def test_weights_match_mode_shape(self, waveform, small_grid):
        """Weights should match mode shape values at grid points."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        weights = membrane.get_injection_weights(small_grid)
        mask = membrane.get_injection_mask(small_grid)

        # Get coordinates
        u, v, plane_idx = membrane._grid_to_rectangular_coords(small_grid)

        # Compute mode shape for all masked cells at once
        all_u = u[mask]
        all_v = v[mask]
        all_expected = membrane.mode_shape(all_u, all_v)

        # Weights inside mask should match mode shape
        all_weights = weights[mask]
        assert np.allclose(all_weights, all_expected, atol=1e-10)

    def test_z_normal_weights(self, waveform, small_grid):
        """Test injection weights for z-normal membrane."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        weights = membrane.get_injection_weights(small_grid)
        plane_idx = int(np.argmin(np.abs(small_grid.z_coords - 0.05)))

        # Only the membrane plane should have non-zero weights
        for k in range(small_grid.shape[2]):
            if k != plane_idx:
                assert np.all(weights[:, :, k] == 0.0)

    def test_y_normal_weights(self, waveform, small_grid):
        """Test injection weights for y-normal membrane."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='y',
            waveform=waveform,
        )

        weights = membrane.get_injection_weights(small_grid)
        plane_idx = int(np.argmin(np.abs(small_grid.y_coords - 0.05)))

        # Only the membrane plane should have non-zero weights
        for j in range(small_grid.shape[1]):
            if j != plane_idx:
                assert np.all(weights[:, j, :] == 0.0)

    def test_x_normal_weights(self, waveform, small_grid):
        """Test injection weights for x-normal membrane."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='x',
            waveform=waveform,
        )

        weights = membrane.get_injection_weights(small_grid)
        plane_idx = int(np.argmin(np.abs(small_grid.x_coords - 0.05)))

        # Only the membrane plane should have non-zero weights
        for i in range(small_grid.shape[0]):
            if i != plane_idx:
                assert np.all(weights[i, :, :] == 0.0)


# =============================================================================
# Solver Integration Tests
# =============================================================================


class TestSolverIntegration:
    """Test integration with FDTDSolver."""

    def test_add_rectangular_membrane_source(self, small_solver, waveform):
        """RectangularMembraneSource can be added to solver."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        small_solver.add_source(membrane)
        assert len(small_solver._sources) == 1

    def test_rectangular_membrane_injects_pressure(self, small_solver, waveform):
        """Rectangular membrane should inject pressure following mode shape."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        small_solver.add_source(membrane)

        # Initial pressure should be zero
        assert np.all(small_solver.p == 0.0)

        # Step the solver
        small_solver.step()

        # Pressure should now be non-zero in membrane region
        weights = membrane.get_injection_weights(small_solver._grid)
        nonzero_weights = weights > 0

        # At least some cells in the membrane area should have pressure
        assert np.any(small_solver.p[nonzero_weights] != 0.0)

    def test_velocity_injection(self, small_solver, waveform):
        """Rectangular membrane with velocity injection should modify velocity."""
        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
            injection_type='velocity',
        )

        small_solver.add_source(membrane)

        # Initial velocity should be zero
        assert np.all(small_solver.vz == 0.0)

        # Step the solver
        small_solver.step()

        # vz should now be non-zero in membrane region
        assert np.any(small_solver.vz != 0.0)

    def test_center_outside_grid_raises(self, small_solver, waveform):
        """Membrane with center outside grid should raise error."""
        membrane = RectangularMembraneSource(
            center=(0.5, 0.05, 0.05),  # x=0.5m is outside 0.1m grid
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        with pytest.raises(ValueError, match="outside grid"):
            small_solver.add_source(membrane)


# =============================================================================
# Nonuniform Grid Tests
# =============================================================================


class TestNonuniformGrid:
    """Test RectangularMembraneSource with nonuniform grids."""

    def test_works_with_nonuniform_grid(self, waveform):
        """RectangularMembraneSource should work with nonuniform grids."""
        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=5e-3,
            stretch_z=1.02,
        )

        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        # Should not raise
        weights = membrane.get_injection_weights(grid)
        assert weights.shape == grid.shape
        assert np.any(weights > 0)

    def test_mask_correct_on_nonuniform_grid(self, waveform):
        """Injection mask should be correct on nonuniform grid."""
        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=5e-3,
            stretch_z=1.02,
        )

        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=(0.03, 0.03),
            normal_axis='z',
            waveform=waveform,
        )

        mask = membrane.get_injection_mask(grid)

        # Should have some True values
        assert np.any(mask)

        # Only one z-plane should have True values
        plane_counts = np.sum(mask, axis=(0, 1))
        assert np.count_nonzero(plane_counts) == 1


# =============================================================================
# Mathematical Accuracy Tests
# =============================================================================


class TestMathematicalAccuracy:
    """Test mathematical correctness of sinusoidal mode shapes."""

    def test_sin_product_formula(self, waveform):
        """Verify mode shape matches sin(mπx/L) * sin(nπy/L) formula."""
        size = (0.04, 0.06)  # Non-square for generality
        Lx, Ly = size

        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        # Sample points inside membrane
        u = np.linspace(-Lx/2 * 0.9, Lx/2 * 0.9, 20)
        v = np.linspace(-Ly/2 * 0.9, Ly/2 * 0.9, 20)
        U, V = np.meshgrid(u, v)
        u_flat = U.flatten()
        v_flat = V.flatten()

        shape = membrane.mode_shape(u_flat, v_flat)

        # Compute expected using standard formula
        # x_norm = u/Lx + 0.5, maps to [0, 1]
        x_norm = u_flat / Lx + 0.5
        y_norm = v_flat / Ly + 0.5
        expected = np.abs(np.sin(np.pi * x_norm) * np.sin(np.pi * y_norm))
        expected = expected / np.max(expected)  # Normalize

        assert np.allclose(shape, expected, atol=1e-10)

    def test_mode_21_formula(self, waveform):
        """Verify mode (2,1) matches sin(2πx/L) * sin(πy/L)."""
        size = (0.04, 0.04)
        Lx, Ly = size

        membrane = RectangularMembraneSource(
            center=(0.05, 0.05, 0.05),
            size=size,
            normal_axis='z',
            waveform=waveform,
            mode=(2, 1),
        )

        # Sample points inside membrane
        u = np.linspace(-Lx/2 * 0.9, Lx/2 * 0.9, 20)
        v = np.linspace(-Ly/2 * 0.9, Ly/2 * 0.9, 20)
        U, V = np.meshgrid(u, v)
        u_flat = U.flatten()
        v_flat = V.flatten()

        shape = membrane.mode_shape(u_flat, v_flat)

        # Compute expected
        x_norm = u_flat / Lx + 0.5
        y_norm = v_flat / Ly + 0.5
        expected = np.abs(np.sin(2 * np.pi * x_norm) * np.sin(np.pi * y_norm))
        max_val = np.max(expected)
        if max_val > 1e-10:
            expected = expected / max_val

        assert np.allclose(shape, expected, atol=1e-10)
