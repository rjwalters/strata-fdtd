"""
Unit tests for MembraneSource base class.

Tests verify:
- Coordinate transformation correctness
- Grid alignment warnings
- Abstract method enforcement
- Integration with FDTDSolver
- Weight caching behavior
"""

import numpy as np
import pytest
import warnings

from strata_fdtd import FDTDSolver, GaussianPulse, MembraneSource
from strata_fdtd.core.grid import UniformGrid


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_grid():
    """Create a small uniform grid for testing."""
    return UniformGrid(shape=(20, 20, 20), resolution=5e-3)


@pytest.fixture
def small_solver():
    """Create a small solver for fast tests."""
    return FDTDSolver(shape=(20, 20, 20), resolution=5e-3, c=343.0, rho=1.2)


class SimpleCircularMembrane(MembraneSource):
    """Simple circular membrane for testing purposes.

    Uses a uniform mode shape (all cells within radius have weight 1.0).
    """

    radius: float = 0.02  # 2cm radius

    def __init__(self, center, normal_axis, waveform, radius=0.02, **kwargs):
        super().__init__(center=center, normal_axis=normal_axis, waveform=waveform, **kwargs)
        object.__setattr__(self, 'radius', radius)

    def mode_shape(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Uniform mode shape - constant 1.0 within boundary."""
        return np.ones_like(r)

    def get_injection_mask(self, grid) -> np.ndarray:
        """Return mask of cells within radius."""
        r, theta, plane_idx = self._grid_to_membrane_coords(grid)
        mask = np.zeros(grid.shape, dtype=bool)

        if self.normal_axis == 'z':
            mask[:, :, plane_idx] = r[:, :, plane_idx] <= self.radius
        elif self.normal_axis == 'y':
            mask[:, plane_idx, :] = r[:, plane_idx, :] <= self.radius
        else:  # x
            mask[plane_idx, :, :] = r[plane_idx, :, :] <= self.radius

        return mask

    def get_injection_weights(self, grid) -> np.ndarray:
        """Return uniform weights within radius, zero outside."""
        r, theta, plane_idx = self._grid_to_membrane_coords(grid)
        weights = np.zeros(grid.shape, dtype=np.float64)

        mask = self.get_injection_mask(grid)
        weights[mask] = self.mode_shape(r[mask], theta[mask])

        return weights


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class TestMembraneSourceAbstract:
    """Test that abstract methods raise NotImplementedError."""

    def test_mode_shape_not_implemented(self):
        """mode_shape() should raise NotImplementedError on base class."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = MembraneSource(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
        )

        r = np.array([0.01, 0.02])
        theta = np.array([0.0, np.pi/4])

        with pytest.raises(NotImplementedError, match="Subclasses must implement mode_shape"):
            membrane.mode_shape(r, theta)

    def test_get_injection_mask_not_implemented(self, small_grid):
        """get_injection_mask() should raise NotImplementedError on base class."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = MembraneSource(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
        )

        with pytest.raises(NotImplementedError, match="Subclasses must implement get_injection_mask"):
            membrane.get_injection_mask(small_grid)

    def test_get_injection_weights_not_implemented(self, small_grid):
        """get_injection_weights() should raise NotImplementedError on base class."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = MembraneSource(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
        )

        with pytest.raises(NotImplementedError, match="Subclasses must implement get_injection_weights"):
            membrane.get_injection_weights(small_grid)


# =============================================================================
# Coordinate Transformation Tests
# =============================================================================


class TestCoordinateTransformation:
    """Test membrane coordinate transformation methods."""

    def test_z_normal_axis_transformation(self, small_grid):
        """Test coordinate transformation for z-normal membrane."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        r, theta, plane_idx = membrane._grid_to_membrane_coords(small_grid)

        # Check shape matches grid
        assert r.shape == small_grid.shape
        assert theta.shape == small_grid.shape

        # Check plane index is correct (0.05m / 0.005m = 10, but cell centers are at 0.5*dx, 1.5*dx, etc.)
        # So center at 0.05m is closest to index 9 or 10 depending on grid.z_coords
        expected_plane_idx = int(np.argmin(np.abs(small_grid.z_coords - 0.05)))
        assert plane_idx == expected_plane_idx

        # At the center, r should be close to 0
        center_i = int(np.argmin(np.abs(small_grid.x_coords - 0.05)))
        center_j = int(np.argmin(np.abs(small_grid.y_coords - 0.05)))
        assert r[center_i, center_j, plane_idx] < small_grid.resolution

    def test_y_normal_axis_transformation(self, small_grid):
        """Test coordinate transformation for y-normal membrane."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='y',
            waveform=waveform,
            radius=0.02,
        )

        r, theta, plane_idx = membrane._grid_to_membrane_coords(small_grid)

        expected_plane_idx = int(np.argmin(np.abs(small_grid.y_coords - 0.05)))
        assert plane_idx == expected_plane_idx

    def test_x_normal_axis_transformation(self, small_grid):
        """Test coordinate transformation for x-normal membrane."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='x',
            waveform=waveform,
            radius=0.02,
        )

        r, theta, plane_idx = membrane._grid_to_membrane_coords(small_grid)

        expected_plane_idx = int(np.argmin(np.abs(small_grid.x_coords - 0.05)))
        assert plane_idx == expected_plane_idx


# =============================================================================
# Grid Alignment Tests
# =============================================================================


class TestGridAlignment:
    """Test grid alignment warning behavior."""

    def test_no_warning_when_on_grid(self, small_grid):
        """No warning when membrane center is on-grid."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        # Place membrane at a grid cell center
        membrane = SimpleCircularMembrane(
            center=(0.0025, 0.0025, 0.0025),  # First cell center at 0.5*dx
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            membrane._check_grid_alignment(small_grid)
            # Filter for our specific warning
            relevant_warnings = [x for x in w if "off-grid" in str(x.message)]
            assert len(relevant_warnings) == 0

    def test_no_warning_at_boundary(self, small_grid):
        """No warning at exactly 0.5 cells offset (boundary case).

        With a uniform grid, the maximum distance from any point to the
        nearest cell center is exactly 0.5 cells (midway between cells).
        The warning threshold is > 0.5 cells, so this boundary case should
        not trigger a warning.
        """
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        # Grid resolution is 5e-3m (5mm), cell centers at 0.0025, 0.0075, etc.
        # z=0.005 is exactly between cells, 0.0025m from nearest = 0.5 cells
        membrane = SimpleCircularMembrane(
            center=(0.0025, 0.0025, 0.005),  # z exactly between cells
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            membrane._check_grid_alignment(small_grid)
            relevant_warnings = [x for x in w if "off-grid" in str(x.message)]
            # At exactly 0.5 cells (max distance), no warning since threshold is >
            assert len(relevant_warnings) == 0


# =============================================================================
# Injection Mask and Weights Tests
# =============================================================================


class TestInjectionMaskAndWeights:
    """Test mask and weight generation."""

    def test_mask_shape_matches_grid(self, small_grid):
        """Injection mask should have same shape as grid."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        mask = membrane.get_injection_mask(small_grid)
        assert mask.shape == small_grid.shape
        assert mask.dtype == bool

    def test_weights_shape_matches_grid(self, small_grid):
        """Injection weights should have same shape as grid."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        weights = membrane.get_injection_weights(small_grid)
        assert weights.shape == small_grid.shape
        assert weights.dtype == np.float64

    def test_mask_is_only_in_membrane_plane(self, small_grid):
        """Mask should only be True in the membrane plane."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        mask = membrane.get_injection_mask(small_grid)
        r, theta, plane_idx = membrane._grid_to_membrane_coords(small_grid)

        # All True values should be in the plane
        for k in range(small_grid.shape[2]):
            if k != plane_idx:
                assert not mask[:, :, k].any(), f"Mask has True values outside plane at k={k}"

    def test_weights_are_zero_outside_radius(self, small_grid):
        """Weights should be zero outside the membrane radius."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.01,  # Small radius
        )

        weights = membrane.get_injection_weights(small_grid)
        mask = membrane.get_injection_mask(small_grid)

        # Weights outside mask should be zero
        assert np.all(weights[~mask] == 0.0)

        # Weights inside mask should be non-zero
        if mask.any():
            assert np.all(weights[mask] > 0.0)


# =============================================================================
# Solver Integration Tests
# =============================================================================


class TestSolverIntegration:
    """Test integration with FDTDSolver."""

    def test_add_membrane_source(self, small_solver):
        """Membrane source can be added to solver."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        # Should not raise
        small_solver.add_source(membrane)
        assert len(small_solver._sources) == 1

    def test_membrane_source_outside_grid_raises(self, small_solver):
        """Adding membrane with center outside grid should raise."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = SimpleCircularMembrane(
            center=(0.5, 0.05, 0.05),  # x=0.5m is outside 0.1m grid
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        with pytest.raises(ValueError, match="outside grid"):
            small_solver.add_source(membrane)

    def test_membrane_injects_pressure(self, small_solver):
        """Membrane source should inject pressure into the field."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200, amplitude=1.0)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        small_solver.add_source(membrane)

        # Initial pressure should be zero
        assert np.all(small_solver.p == 0.0)

        # Step the solver
        small_solver.step()

        # Pressure should now be non-zero in membrane region
        assert np.any(small_solver.p != 0.0)

    def test_velocity_injection(self, small_solver):
        """Membrane with velocity injection should modify velocity field."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200, amplitude=1.0)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
            injection_type='velocity',
        )

        small_solver.add_source(membrane)

        # Initial velocity should be zero
        assert np.all(small_solver.vz == 0.0)

        # Step the solver
        small_solver.step()

        # vz should now be non-zero in membrane region
        assert np.any(small_solver.vz != 0.0)

    def test_weights_are_cached(self, small_solver):
        """Weights should be computed once and cached."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200, amplitude=1.0)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        small_solver.add_source(membrane)

        # Before first step, cache should be None
        assert membrane._cached_weights is None
        assert membrane._cached_mask is None

        # After first step, cache should be populated
        small_solver.step()
        assert membrane._cached_weights is not None
        assert membrane._cached_mask is not None

        # Store reference to cached arrays
        cached_weights = membrane._cached_weights
        cached_mask = membrane._cached_mask

        # After second step, should use same cached arrays
        small_solver.step()
        assert membrane._cached_weights is cached_weights
        assert membrane._cached_mask is cached_mask

    def test_respects_geometry_mask(self, small_solver):
        """Membrane injection should respect geometry mask (only inject in air)."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200, amplitude=1.0)
        membrane = SimpleCircularMembrane(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            radius=0.02,
        )

        # Block part of the membrane area
        small_solver.geometry[9:12, 9:12, :] = False  # Solid block

        small_solver.add_source(membrane)
        small_solver.step()

        # Pressure in solid region should be zero
        assert np.all(small_solver.p[9:12, 9:12, :] == 0.0)


# =============================================================================
# Source Type Tests
# =============================================================================


class TestSourceType:
    """Test source_type attribute behavior."""

    def test_source_type_is_membrane(self):
        """MembraneSource should have source_type='membrane'."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)
        membrane = MembraneSource(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
        )

        assert membrane.source_type == 'membrane'

    def test_source_type_not_settable(self):
        """source_type should not be settable via init."""
        waveform = GaussianPulse(position=(0, 0, 0), frequency=200)

        # source_type has init=False, so it can't be passed
        # This test verifies it's always 'membrane'
        membrane = MembraneSource(
            center=(0.05, 0.05, 0.05),
            normal_axis='z',
            waveform=waveform,
            mode=(1, 2),  # Test other params still work
        )

        assert membrane.source_type == 'membrane'
        assert membrane.mode == (1, 2)
