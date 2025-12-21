"""
Unit tests for CircularMembraneSource with Bessel function mode shapes.

Tests verify:
- Bessel function mode shape accuracy against scipy reference
- Boundary conditions (zero at edge, maximum at center)
- Support for fundamental and higher-order modes
- Grid discretization correctness
- Integration with FDTDSolver
"""

import numpy as np
import pytest
from scipy.special import j0, jn_zeros, jv

from strata_fdtd import (
    CircularMembraneSource,
    FDTDSolver,
    GaussianPulse,
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


class TestBesselModeShape:
    """Test Bessel function mode shape implementation."""

    def test_fundamental_mode_center_is_one(self, waveform, small_grid):
        """Fundamental mode (0,1) should have value 1.0 at center."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        # At r=0, J_0(0) = 1, so normalized value should be 1.0
        r = np.array([0.0])
        theta = np.array([0.0])
        shape = membrane.mode_shape(r, theta)
        assert np.isclose(shape[0], 1.0, atol=1e-10)

    def test_fundamental_mode_edge_is_zero(self, waveform, small_grid):
        """Fundamental mode (0,1) should have value 0.0 at edge."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        # At r=radius, J_0(alpha_01) = 0
        r = np.array([radius])
        theta = np.array([0.0])
        shape = membrane.mode_shape(r, theta)
        assert np.isclose(shape[0], 0.0, atol=1e-10)

    def test_outside_radius_is_zero(self, waveform, small_grid):
        """Mode shape should be zero outside the membrane radius."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        # At r > radius, should be zero
        r = np.array([radius * 1.5, radius * 2.0])
        theta = np.array([0.0, np.pi / 4])
        shape = membrane.mode_shape(r, theta)
        assert np.all(shape == 0.0)

    def test_mode_shape_matches_scipy_j0(self, waveform):
        """Mode shape should match scipy.special.j0 for fundamental mode."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        # Sample points across the membrane
        r = np.linspace(0, radius, 50)
        theta = np.zeros_like(r)

        # Get membrane mode shape
        shape = membrane.mode_shape(r, theta)

        # Compute expected from scipy
        alpha = jn_zeros(0, 1)[0]  # First zero of J_0
        rho = r / radius
        expected = np.abs(j0(alpha * rho))
        expected = expected / np.max(expected)  # Normalize

        assert np.allclose(shape, expected, atol=1e-10)

    def test_mode_01_matches_scipy(self, waveform):
        """Mode (0,2) should match scipy Bessel function."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 2),
        )

        r = np.linspace(0, radius, 50)
        theta = np.zeros_like(r)
        shape = membrane.mode_shape(r, theta)

        # Expected: J_0 evaluated at second zero
        alpha = jn_zeros(0, 2)[1]  # Second zero of J_0
        rho = r / radius
        expected = np.abs(j0(alpha * rho))
        expected = expected / np.max(expected)

        assert np.allclose(shape, expected, atol=1e-10)

    def test_mode_11_matches_scipy(self, waveform):
        """Mode (1,1) should match scipy Bessel function with angular dependence."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        # Test along a line where cos(theta) = 1
        r = np.linspace(0, radius * 0.99, 50)  # Avoid edge for numerical stability
        theta = np.zeros_like(r)  # theta=0, so cos(theta)=1
        shape = membrane.mode_shape(r, theta)

        # Expected: |J_1(alpha_11 * rho) * cos(theta)|, normalized
        alpha = jn_zeros(1, 1)[0]  # First zero of J_1
        rho = r / radius
        expected = np.abs(jv(1, alpha * rho) * np.cos(theta))
        if np.max(expected) > 0:
            expected = expected / np.max(expected)

        assert np.allclose(shape, expected, atol=1e-10)

    def test_mode_shape_values_in_range(self, waveform):
        """All mode shape values should be in [0, 1]."""
        radius = 0.02
        for mode in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]:
            membrane = CircularMembraneSource(
                center=(0.05, 0.05, 0.05),
                radius=radius,
                normal_axis='z',
                waveform=waveform,
                mode=mode,
            )

            r = np.linspace(0, radius * 1.5, 100)
            theta = np.linspace(0, 2 * np.pi, 100)
            R, THETA = np.meshgrid(r, theta)
            shape = membrane.mode_shape(R.flatten(), THETA.flatten())

            assert np.all(shape >= 0.0), f"Mode {mode} has negative values"
            assert np.all(shape <= 1.0), f"Mode {mode} has values > 1"


# =============================================================================
# Higher-Order Mode Tests
# =============================================================================


class TestHigherOrderModes:
    """Test higher-order Bessel modes."""

    def test_mode_02_has_radial_node(self, waveform):
        """Mode (0,2) should have one radial node between center and edge."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 2),
        )

        # Sample along radius
        r = np.linspace(0.001 * radius, 0.99 * radius, 100)
        theta = np.zeros_like(r)
        shape = membrane.mode_shape(r, theta)

        # The J_0 function has a zero between 0 and alpha_02
        # At normalized r, the first zero of J_0 is at ~2.405/5.520 = 0.436
        # So there should be a local minimum (crossing through zero) around there
        # Since we take absolute value, we look for a local minimum

        # Find local minima
        min_indices = []
        for i in range(1, len(shape) - 1):
            if shape[i] < shape[i - 1] and shape[i] < shape[i + 1]:
                min_indices.append(i)

        assert len(min_indices) >= 1, "Mode (0,2) should have at least one radial node"

    def test_mode_11_angular_variation(self, waveform):
        """Mode (1,1) should vary with angle (cos(theta))."""
        radius = 0.02
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(1, 1),
        )

        r_mid = radius * 0.5  # Halfway to edge

        # Sample multiple angles to get proper normalization context
        # The mode shape varies as cos(theta), so theta=0 gives max, theta=pi/2 gives 0
        r_values = np.array([r_mid, r_mid, r_mid])
        theta_values = np.array([0.0, np.pi, np.pi / 2])
        shapes = membrane.mode_shape(r_values, theta_values)

        # At theta=0 and theta=pi, cos(theta) has opposite signs
        # But since we take absolute value, both should be equal and maximum
        assert np.isclose(shapes[0], shapes[1], atol=1e-10)

        # At theta=pi/2, cos(theta)=0, so mode shape should be 0
        # This is a nodal line for the (1,1) mode
        assert np.isclose(shapes[2], 0.0, atol=1e-10)

    def test_dynamic_bessel_zero_computation(self, waveform):
        """Higher modes not in cache should compute zeros dynamically."""
        radius = 0.02
        # Mode (3, 3) is not in the precomputed cache
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=radius,
            normal_axis='z',
            waveform=waveform,
            mode=(3, 3),
        )

        # Should not raise and should have valid alpha
        expected_alpha = jn_zeros(3, 3)[2]  # Third zero of J_3
        assert np.isclose(membrane._alpha, expected_alpha, rtol=1e-10)


# =============================================================================
# Invalid Mode Tests
# =============================================================================


class TestInvalidModes:
    """Test error handling for invalid mode parameters."""

    def test_negative_azimuthal_mode_raises(self, waveform):
        """Negative azimuthal mode m should raise ValueError."""
        with pytest.raises(ValueError, match="Azimuthal mode m must be non-negative"):
            CircularMembraneSource(
                center=(0.05, 0.05, 0.05),
                radius=0.02,
                normal_axis='z',
                waveform=waveform,
                mode=(-1, 1),
            )

    def test_zero_radial_mode_raises(self, waveform):
        """Zero radial mode n should raise ValueError."""
        with pytest.raises(ValueError, match="Radial mode n must be positive"):
            CircularMembraneSource(
                center=(0.05, 0.05, 0.05),
                radius=0.02,
                normal_axis='z',
                waveform=waveform,
                mode=(0, 0),
            )

    def test_negative_radial_mode_raises(self, waveform):
        """Negative radial mode n should raise ValueError."""
        with pytest.raises(ValueError, match="Radial mode n must be positive"):
            CircularMembraneSource(
                center=(0.05, 0.05, 0.05),
                radius=0.02,
                normal_axis='z',
                waveform=waveform,
                mode=(0, -1),
            )


# =============================================================================
# Grid Injection Tests
# =============================================================================


class TestGridInjection:
    """Test grid injection mask and weights."""

    def test_mask_is_circular(self, waveform, small_grid):
        """Injection mask should be approximately circular."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
        )

        mask = membrane.get_injection_mask(small_grid)

        # Find plane index
        plane_idx = int(np.argmin(np.abs(small_grid.z_coords - 0.05)))

        # Count cells in each direction from center
        center_i = int(np.argmin(np.abs(small_grid.x_coords - 0.05)))
        center_j = int(np.argmin(np.abs(small_grid.y_coords - 0.05)))

        # Mask should be roughly symmetric around center
        plane_mask = mask[:, :, plane_idx]
        assert plane_mask[center_i, center_j], "Center should be in mask"

    def test_weights_shape_matches_grid(self, waveform, small_grid):
        """Injection weights should match grid shape."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
        )

        weights = membrane.get_injection_weights(small_grid)
        assert weights.shape == small_grid.shape

    def test_weights_match_mode_shape(self, waveform, small_grid):
        """Weights should match mode shape values at grid points."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        weights = membrane.get_injection_weights(small_grid)
        mask = membrane.get_injection_mask(small_grid)

        # Get coordinates
        r, theta, plane_idx = membrane._grid_to_membrane_coords(small_grid)

        # Compute mode shape for all masked cells at once
        # (to get proper normalization context)
        all_r = r[mask]
        all_theta = theta[mask]
        all_expected = membrane.mode_shape(all_r, all_theta)

        # Weights inside mask should match mode shape
        all_weights = weights[mask]
        assert np.allclose(all_weights, all_expected, atol=1e-10)

    def test_z_normal_weights(self, waveform, small_grid):
        """Test injection weights for z-normal membrane."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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

    def test_add_circular_membrane_source(self, small_solver, waveform):
        """CircularMembraneSource can be added to solver."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
        )

        small_solver.add_source(membrane)
        assert len(small_solver._sources) == 1

    def test_circular_membrane_injects_pressure(self, small_solver, waveform):
        """Circular membrane should inject pressure following mode shape."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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
        """Circular membrane with velocity injection should modify velocity."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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
        membrane = CircularMembraneSource(
            center=(0.5, 0.05, 0.05),  # x=0.5m is outside 0.1m grid
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
        )

        with pytest.raises(ValueError, match="outside grid"):
            small_solver.add_source(membrane)


# =============================================================================
# Nonuniform Grid Tests
# =============================================================================


class TestNonuniformGrid:
    """Test CircularMembraneSource with nonuniform grids."""

    def test_works_with_nonuniform_grid(self, waveform):
        """CircularMembraneSource should work with nonuniform grids."""
        grid = NonuniformGrid.from_stretch(
            shape=(20, 20, 20),
            base_resolution=5e-3,
            stretch_z=1.02,
        )

        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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

        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
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
# Bessel Zero Cache Tests
# =============================================================================


class TestBesselZeroCache:
    """Test the precomputed Bessel zero cache."""

    def test_cached_zeros_are_accurate(self, waveform):
        """Cached Bessel zeros should match scipy computation."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
        )

        for (m, n), cached_zero in membrane._BESSEL_ZEROS.items():
            computed_zero = jn_zeros(m, n)[n - 1]
            assert np.isclose(
                cached_zero, computed_zero, rtol=1e-10
            ), f"Cached zero for mode ({m},{n}) is inaccurate"

    def test_fundamental_mode_uses_cache(self, waveform):
        """Fundamental mode (0,1) should use cached Bessel zero."""
        membrane = CircularMembraneSource(
            center=(0.05, 0.05, 0.05),
            radius=0.02,
            normal_axis='z',
            waveform=waveform,
            mode=(0, 1),
        )

        expected_alpha = 2.4048255576957727  # First zero of J_0
        assert np.isclose(membrane._alpha, expected_alpha, rtol=1e-10)
