"""
Unit tests for spiral and helical path generators.

Tests verify:
- Path construction and parameter validation
- Point sampling along paths
- Path length computation
- Distance-to-curve utilities
"""

import numpy as np
import pytest

from strata_fdtd.paths import (
    HelixPath,
    SpiralPath,
    distance_to_helix,
    distance_to_polyline,
)

# =============================================================================
# SpiralPath Tests
# =============================================================================


class TestSpiralPath:
    def test_construction_logarithmic(self):
        """Test logarithmic spiral construction."""
        spiral = SpiralPath(
            center=(0.15, 0.15),
            inner_radius=0.04,
            outer_radius=0.12,
            turns=2.5,
            spiral_type="logarithmic",
        )

        np.testing.assert_allclose(spiral.center, [0.15, 0.15])
        assert spiral.inner_radius == 0.04
        assert spiral.outer_radius == 0.12
        assert spiral.turns == 2.5
        assert spiral.spiral_type == "logarithmic"

    def test_construction_archimedean(self):
        """Test Archimedean spiral construction."""
        spiral = SpiralPath(
            center=(0.1, 0.1),
            inner_radius=0.02,
            outer_radius=0.08,
            turns=3.0,
            spiral_type="archimedean",
        )

        assert spiral.spiral_type == "archimedean"

    def test_construction_validates_parameters(self):
        """Test parameter validation during construction."""
        # Invalid inner radius
        with pytest.raises(ValueError, match="inner_radius must be positive"):
            SpiralPath(
                center=(0.1, 0.1),
                inner_radius=0.0,
                outer_radius=0.1,
                turns=2.0,
            )

        # Outer radius not greater than inner
        with pytest.raises(ValueError, match="outer_radius.*must be >"):
            SpiralPath(
                center=(0.1, 0.1),
                inner_radius=0.1,
                outer_radius=0.05,
                turns=2.0,
            )

        # Invalid turns
        with pytest.raises(ValueError, match="turns must be positive"):
            SpiralPath(
                center=(0.1, 0.1),
                inner_radius=0.04,
                outer_radius=0.08,
                turns=0.0,
            )

        # Invalid spiral type
        with pytest.raises(ValueError, match="spiral_type must be"):
            SpiralPath(
                center=(0.1, 0.1),
                inner_radius=0.04,
                outer_radius=0.08,
                turns=2.0,
                spiral_type="invalid",  # type: ignore
            )

    def test_sample_points_shape(self):
        """Test that sample_points returns correct shape."""
        spiral = SpiralPath(
            center=(0.1, 0.1),
            inner_radius=0.04,
            outer_radius=0.08,
            turns=2.0,
        )

        points = spiral.sample_points(100)

        assert points.shape == (100, 2)

    def test_sample_points_endpoints_logarithmic(self):
        """Test that logarithmic spiral starts and ends at correct radii."""
        spiral = SpiralPath(
            center=(0.0, 0.0),
            inner_radius=0.04,
            outer_radius=0.12,
            turns=2.0,
            spiral_type="logarithmic",
        )

        points = spiral.sample_points(1000)

        # First point should be at inner radius
        r_start = np.linalg.norm(points[0])
        assert abs(r_start - 0.04) < 1e-10

        # Last point should be at outer radius
        r_end = np.linalg.norm(points[-1])
        assert abs(r_end - 0.12) < 1e-10

    def test_sample_points_endpoints_archimedean(self):
        """Test that Archimedean spiral starts and ends at correct radii."""
        spiral = SpiralPath(
            center=(0.0, 0.0),
            inner_radius=0.03,
            outer_radius=0.09,
            turns=3.0,
            spiral_type="archimedean",
        )

        points = spiral.sample_points(1000)

        # First point should be at inner radius
        r_start = np.linalg.norm(points[0])
        assert abs(r_start - 0.03) < 1e-10

        # Last point should be at outer radius
        r_end = np.linalg.norm(points[-1])
        assert abs(r_end - 0.09) < 1e-10

    def test_total_length_positive(self):
        """Test that total length is positive."""
        spiral = SpiralPath(
            center=(0.1, 0.1),
            inner_radius=0.04,
            outer_radius=0.12,
            turns=2.5,
        )

        length = spiral.total_length

        assert length > 0
        # For this spiral, length should be reasonable (rough estimate)
        # 2.5 turns with avg radius ~0.08 gives circumference ~0.5
        # Total ~1.25m is reasonable
        assert 0.5 < length < 3.0

    def test_total_length_increases_with_turns(self):
        """Test that length increases with more turns."""
        spiral1 = SpiralPath(
            center=(0.1, 0.1),
            inner_radius=0.04,
            outer_radius=0.08,
            turns=2.0,
        )
        spiral2 = SpiralPath(
            center=(0.1, 0.1),
            inner_radius=0.04,
            outer_radius=0.08,
            turns=4.0,
        )

        assert spiral2.total_length > spiral1.total_length


# =============================================================================
# HelixPath Tests
# =============================================================================


class TestHelixPath:
    def test_construction(self):
        """Test helix construction."""
        helix = HelixPath(
            center=(0.1, 0.1, 0.0),
            radius=0.06,
            pitch=0.04,
            turns=3.0,
        )

        np.testing.assert_allclose(helix.center, [0.1, 0.1, 0.0])
        assert helix.radius == 0.06
        assert helix.pitch == 0.04
        assert helix.turns == 3.0

    def test_construction_validates_parameters(self):
        """Test parameter validation during construction."""
        # Invalid radius
        with pytest.raises(ValueError, match="radius must be positive"):
            HelixPath(
                center=(0.1, 0.1, 0.0),
                radius=-0.05,
                pitch=0.04,
                turns=3.0,
            )

        # Invalid pitch
        with pytest.raises(ValueError, match="pitch must be positive"):
            HelixPath(
                center=(0.1, 0.1, 0.0),
                radius=0.06,
                pitch=0.0,
                turns=3.0,
            )

        # Invalid turns
        with pytest.raises(ValueError, match="turns must be positive"):
            HelixPath(
                center=(0.1, 0.1, 0.0),
                radius=0.06,
                pitch=0.04,
                turns=-1.0,
            )

    def test_sample_points_shape(self):
        """Test that sample_points returns correct shape."""
        helix = HelixPath(
            center=(0.1, 0.1, 0.0),
            radius=0.06,
            pitch=0.04,
            turns=3.0,
        )

        points = helix.sample_points(150)

        assert points.shape == (150, 3)

    def test_sample_points_z_progression(self):
        """Test that helix progresses in Z correctly."""
        helix = HelixPath(
            center=(0.0, 0.0, 0.0),
            radius=0.05,
            pitch=0.04,
            turns=3.0,
        )

        points = helix.sample_points(1000)

        # First point should be at z=0
        assert abs(points[0, 2]) < 1e-10

        # Last point should be at z = turns * pitch
        expected_z = 3.0 * 0.04
        assert abs(points[-1, 2] - expected_z) < 1e-10

        # Z should be monotonically increasing
        z_values = points[:, 2]
        assert np.all(np.diff(z_values) >= 0)

    def test_sample_points_radial_distance(self):
        """Test that all points are at constant radial distance from axis."""
        helix = HelixPath(
            center=(0.0, 0.0, 0.0),
            radius=0.05,
            pitch=0.04,
            turns=2.0,
        )

        points = helix.sample_points(1000)

        # Radial distance in XY plane should be constant
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        np.testing.assert_allclose(r, 0.05, rtol=1e-10)

    def test_total_length_analytical(self):
        """Test total length matches analytical formula."""
        helix = HelixPath(
            center=(0.1, 0.1, 0.0),
            radius=0.06,
            pitch=0.04,
            turns=3.0,
        )

        # Analytical: L = turns * sqrt((2*pi*r)^2 + pitch^2)
        circumference = 2 * np.pi * 0.06
        expected = 3.0 * np.sqrt(circumference**2 + 0.04**2)

        assert abs(helix.total_length - expected) < 1e-10

    def test_total_height(self):
        """Test total height calculation."""
        helix = HelixPath(
            center=(0.1, 0.1, 0.0),
            radius=0.06,
            pitch=0.04,
            turns=3.0,
        )

        assert abs(helix.total_height - 0.12) < 1e-10


# =============================================================================
# Distance Utility Tests
# =============================================================================


class TestDistanceToPolyline:
    def test_distance_to_straight_line_2d(self):
        """Test distance to a straight line in 2D."""
        # Line from (0,0) to (1,0)
        polyline = np.array([[0.0, 0.0], [1.0, 0.0]])

        # Point directly above midpoint
        points = np.array([[0.5, 0.5]])

        distances = distance_to_polyline(points, polyline)

        # Distance should be 0.5
        assert abs(distances[0] - 0.5) < 1e-10

    def test_distance_to_straight_line_3d(self):
        """Test distance to a straight line in 3D."""
        # Line from (0,0,0) to (1,0,0)
        polyline = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Point off to the side
        points = np.array([[0.5, 0.3, 0.4]])

        distances = distance_to_polyline(points, polyline)

        # Distance should be sqrt(0.3^2 + 0.4^2) = 0.5
        expected = np.sqrt(0.3**2 + 0.4**2)
        assert abs(distances[0] - expected) < 1e-10

    def test_distance_to_bent_path(self):
        """Test distance to a bent path."""
        # L-shaped path
        polyline = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

        # Point near corner
        points = np.array([[1.1, 0.5]])

        distances = distance_to_polyline(points, polyline)

        # Distance should be 0.1 (to the vertical segment)
        assert abs(distances[0] - 0.1) < 1e-10

    def test_multiple_points(self):
        """Test distance for multiple query points."""
        polyline = np.array([[0.0, 0.0], [1.0, 0.0]])
        points = np.array(
            [
                [0.5, 0.5],  # Distance 0.5
                [0.5, 1.0],  # Distance 1.0
                [0.5, 0.0],  # Distance 0.0 (on line)
            ]
        )

        distances = distance_to_polyline(points, polyline)

        np.testing.assert_allclose(distances, [0.5, 1.0, 0.0], atol=1e-10)

    def test_validates_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        polyline_2d = np.array([[0.0, 0.0], [1.0, 0.0]])
        points_3d = np.array([[0.5, 0.5, 0.5]])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            distance_to_polyline(points_3d, polyline_2d)


class TestDistanceToHelix:
    def test_distance_to_helix_on_curve(self):
        """Test distance for point on helix curve."""
        # Create helix
        center = (0.0, 0.0, 0.0)
        radius = 0.05
        pitch = 0.04
        turns = 2.0

        # Point on helix at theta = pi/2 (first quarter turn)
        theta = np.pi / 2
        point = np.array(
            [
                [
                    0.0 + radius * np.cos(theta),
                    0.0 + radius * np.sin(theta),
                    0.0 + (theta / (2 * np.pi)) * pitch,
                ]
            ]
        )

        distances = distance_to_helix(point, center, radius, pitch, turns)

        # Should be very close to zero (on the curve)
        assert distances[0] < 1e-3  # Discretization tolerance

    def test_distance_to_helix_center(self):
        """Test distance from helix center."""
        center = (0.0, 0.0, 0.0)
        radius = 0.05
        pitch = 0.04
        turns = 2.0

        # Point at center, mid-height
        points = np.array([[0.0, 0.0, 0.04]])

        distances = distance_to_helix(points, center, radius, pitch, turns)

        # Distance should be approximately the radius
        assert abs(distances[0] - radius) < 1e-3

    def test_multiple_points(self):
        """Test distance for multiple query points."""
        center = (0.0, 0.0, 0.0)
        radius = 0.05
        pitch = 0.04
        turns = 2.0

        points = np.array(
            [
                [0.0, 0.0, 0.0],  # At center, bottom
                [0.05, 0.0, 0.0],  # On helix start
                [0.1, 0.0, 0.04],  # Far from helix
            ]
        )

        distances = distance_to_helix(points, center, radius, pitch, turns)

        # All distances should be non-negative
        assert np.all(distances >= 0)
        # Second point should be closest (on curve, distance ~0)
        assert distances[1] < 1e-3  # Within discretization tolerance
        assert distances[1] < distances[0]
        assert distances[1] < distances[2]
