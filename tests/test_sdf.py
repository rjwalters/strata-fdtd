"""
Unit tests for SDF (Signed Distance Function) primitives.

Tests verify:
- SDF correctness for known points inside/outside shapes
- Bounding box computation
- Voxelization with UniformGrid and NonuniformGrid
- Edge cases and parameter validation
"""

import numpy as np
import pytest

from strata_fdtd import (
    Box,
    Cone,
    Cylinder,
    HelicalTube,
    Horn,
    NonuniformGrid,
    Sphere,
    SpiralHorn,
    UniformGrid,
)
from strata_fdtd.sdf import conical_horn, exponential_horn, hyperbolic_horn, tractrix_horn

# =============================================================================
# Box Tests
# =============================================================================


class TestBox:
    def test_construction_center_size(self):
        """Test box construction with center and size."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        np.testing.assert_allclose(box.center, [0.05, 0.05, 0.05])
        np.testing.assert_allclose(box.size, [0.02, 0.02, 0.02])

    def test_construction_corners(self):
        """Test box construction with min/max corners."""
        box = Box(min_corner=(0.04, 0.04, 0.04), max_corner=(0.06, 0.06, 0.06))

        np.testing.assert_allclose(box.center, [0.05, 0.05, 0.05])
        np.testing.assert_allclose(box.size, [0.02, 0.02, 0.02])

    def test_construction_requires_parameters(self):
        """Test that construction fails without proper parameters."""
        with pytest.raises(ValueError, match="Must provide either"):
            Box(center=(0, 0, 0))  # Missing size

        with pytest.raises(ValueError, match="Must provide either"):
            Box(min_corner=(0, 0, 0))  # Missing max_corner

    def test_construction_rejects_negative_size(self):
        """Test that negative sizes are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            Box(center=(0, 0, 0), size=(-1, 1, 1))

    def test_sdf_center_point(self):
        """Test SDF at box center (inside)."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        points = np.array([[0.05, 0.05, 0.05]])

        distances = box.sdf(points)

        # Center point should be inside (negative distance)
        assert distances[0] < 0
        # Distance should be -half_size = -0.01
        assert abs(distances[0] + 0.01) < 1e-10

    def test_sdf_surface_point(self):
        """Test SDF at box surface (should be ~0)."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        # Point on surface at x = 0.06
        points = np.array([[0.06, 0.05, 0.05]])

        distances = box.sdf(points)

        # Surface point should have distance ~0
        assert abs(distances[0]) < 1e-10

    def test_sdf_outside_point(self):
        """Test SDF for point outside box."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        # Point 0.01m outside in x direction
        points = np.array([[0.07, 0.05, 0.05]])

        distances = box.sdf(points)

        # Should be outside (positive distance of 0.01)
        assert abs(distances[0] - 0.01) < 1e-10

    def test_sdf_corner_point(self):
        """Test SDF at box corner."""
        box = Box(center=(0, 0, 0), size=(2, 2, 2))
        # Corner at (1, 1, 1)
        points = np.array([[1, 1, 1]])

        distances = box.sdf(points)

        # Corner should be on surface
        assert abs(distances[0]) < 1e-10

    def test_contains(self):
        """Test contains() method."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        points = np.array([
            [0.05, 0.05, 0.05],  # Inside (center)
            [0.06, 0.05, 0.05],  # On surface
            [0.07, 0.05, 0.05],  # Outside
        ])

        inside = box.contains(points)

        assert inside[0]  # Center is inside
        assert inside[1]  # Surface is considered inside (<=0)
        assert not inside[2]  # Outside

    def test_bounding_box(self):
        """Test bounding box computation."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        min_corner, max_corner = box.bounding_box

        np.testing.assert_allclose(min_corner, [0.04, 0.04, 0.04])
        np.testing.assert_allclose(max_corner, [0.06, 0.06, 0.06])

    def test_voxelize_uniform_grid(self):
        """Test voxelization with uniform grid."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        mask = box.voxelize(grid)

        # Check shape
        assert mask.shape == (100, 100, 100)

        # Center cell should be inside
        assert mask[50, 50, 50]

        # Corner cells should be outside
        assert not mask[0, 0, 0]
        assert not mask[99, 99, 99]

    def test_voxelize_nonuniform_grid(self):
        """Test voxelization with nonuniform grid."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        grid = NonuniformGrid.from_stretch(
            shape=(50, 50, 50),
            base_resolution=1e-3,
            stretch_z=1.05,
        )

        mask = box.voxelize(grid)

        # Check shape
        assert mask.shape == (50, 50, 50)

        # Some cells should be inside, some outside
        assert mask.any()  # At least one True
        assert not mask.all()  # At least one False


# =============================================================================
# Sphere Tests
# =============================================================================


class TestSphere:
    def test_construction(self):
        """Test sphere construction."""
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.01)

        np.testing.assert_allclose(sphere.center, [0.05, 0.05, 0.05])
        assert sphere.radius == 0.01

    def test_construction_rejects_negative_radius(self):
        """Test that negative radius is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            Sphere(center=(0, 0, 0), radius=-1)

    def test_sdf_center_point(self):
        """Test SDF at sphere center."""
        sphere = Sphere(center=(0, 0, 0), radius=0.01)
        points = np.array([[0, 0, 0]])

        distances = sphere.sdf(points)

        # Center should be inside with distance = -radius
        assert abs(distances[0] + 0.01) < 1e-10

    def test_sdf_surface_point(self):
        """Test SDF at sphere surface."""
        sphere = Sphere(center=(0, 0, 0), radius=0.01)
        # Point on surface
        points = np.array([[0.01, 0, 0]])

        distances = sphere.sdf(points)

        # Surface point should have distance ~0
        assert abs(distances[0]) < 1e-10

    def test_sdf_outside_point(self):
        """Test SDF for point outside sphere."""
        sphere = Sphere(center=(0, 0, 0), radius=0.01)
        # Point at distance 0.02 from center
        points = np.array([[0.02, 0, 0]])

        distances = sphere.sdf(points)

        # Distance should be 0.02 - 0.01 = 0.01
        assert abs(distances[0] - 0.01) < 1e-10

    def test_contains(self):
        """Test contains() method."""
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.01)
        points = np.array([
            [0.05, 0.05, 0.05],  # Center
            [0.06, 0.05, 0.05],  # On surface
            [0.07, 0.05, 0.05],  # Outside
        ])

        inside = sphere.contains(points)

        assert inside[0]
        assert inside[1]  # Surface is inside (<=0)
        assert not inside[2]

    def test_bounding_box(self):
        """Test bounding box computation."""
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.01)

        min_corner, max_corner = sphere.bounding_box

        np.testing.assert_allclose(min_corner, [0.04, 0.04, 0.04])
        np.testing.assert_allclose(max_corner, [0.06, 0.06, 0.06])

    def test_voxelize_uniform_grid(self):
        """Test voxelization with uniform grid."""
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.01)
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        mask = sphere.voxelize(grid)

        assert mask.shape == (100, 100, 100)
        assert mask[50, 50, 50]  # Center
        assert not mask[0, 0, 0]  # Corner


# =============================================================================
# Cylinder Tests
# =============================================================================


class TestCylinder:
    def test_construction(self):
        """Test cylinder construction."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)

        np.testing.assert_allclose(cyl.p1, [0, 0, 0])
        np.testing.assert_allclose(cyl.p2, [0, 0, 0.1])
        assert cyl.radius == 0.01
        assert abs(cyl.length - 0.1) < 1e-10

    def test_construction_rejects_negative_radius(self):
        """Test that negative radius is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            Cylinder(p1=(0, 0, 0), p2=(0, 0, 1), radius=-1)

    def test_construction_rejects_coincident_points(self):
        """Test that coincident endpoints are rejected."""
        with pytest.raises(ValueError, match="must be distinct"):
            Cylinder(p1=(0, 0, 0), p2=(0, 0, 0), radius=0.01)

    def test_sdf_axis_point(self):
        """Test SDF for point on cylinder axis."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)
        # Point on axis, midway
        points = np.array([[0, 0, 0.05]])

        distances = cyl.sdf(points)

        # Distance should be -radius (inside)
        assert abs(distances[0] + 0.01) < 1e-10

    def test_sdf_surface_point(self):
        """Test SDF at cylinder surface."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)
        # Point on surface, midway along axis
        points = np.array([[0.01, 0, 0.05]])

        distances = cyl.sdf(points)

        # Should be on surface (~0)
        assert abs(distances[0]) < 1e-10

    def test_sdf_outside_radial(self):
        """Test SDF for point outside cylinder radially."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)
        # Point 0.02 from axis (0.01 outside)
        points = np.array([[0.02, 0, 0.05]])

        distances = cyl.sdf(points)

        # Distance should be 0.01 (0.02 - 0.01)
        assert abs(distances[0] - 0.01) < 1e-10

    def test_sdf_outside_cap(self):
        """Test SDF for point beyond cylinder cap."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)
        # Point on axis but beyond cap at z=0.15
        points = np.array([[0, 0, 0.15]])

        distances = cyl.sdf(points)

        # Should be outside (positive distance)
        assert distances[0] > 0

    def test_arbitrary_orientation(self):
        """Test cylinder with arbitrary orientation."""
        # Horizontal cylinder along x-axis
        cyl = Cylinder(p1=(0, 0.05, 0.05), p2=(0.1, 0.05, 0.05), radius=0.01)

        # Point on axis
        points = np.array([[0.05, 0.05, 0.05]])
        distances = cyl.sdf(points)

        # Should be inside with distance -radius
        assert abs(distances[0] + 0.01) < 1e-10

    def test_bounding_box(self):
        """Test bounding box computation."""
        cyl = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)

        min_corner, max_corner = cyl.bounding_box

        # Bounding box is conservative (includes radius in all directions)
        np.testing.assert_allclose(min_corner, [-0.01, -0.01, -0.01])
        np.testing.assert_allclose(max_corner, [0.01, 0.01, 0.11])

    def test_voxelize_uniform_grid(self):
        """Test voxelization with uniform grid."""
        cyl = Cylinder(p1=(0.05, 0.05, 0.04), p2=(0.05, 0.05, 0.06), radius=0.005)
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        mask = cyl.voxelize(grid)

        assert mask.shape == (100, 100, 100)
        assert mask[50, 50, 50]  # Center should be inside
        assert not mask[0, 0, 0]  # Corner should be outside


# =============================================================================
# Cone Tests
# =============================================================================


class TestCone:
    def test_construction(self):
        """Test cone construction."""
        cone = Cone(p1=(0, 0, 0), p2=(0, 0, 0.1), r1=0.02, r2=0.01)

        np.testing.assert_allclose(cone.p1, [0, 0, 0])
        np.testing.assert_allclose(cone.p2, [0, 0, 0.1])
        assert cone.r1 == 0.02
        assert cone.r2 == 0.01

    def test_construction_rejects_negative_radii(self):
        """Test that negative radii are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Cone(p1=(0, 0, 0), p2=(0, 0, 1), r1=-1, r2=0.5)

    def test_construction_rejects_both_radii_zero(self):
        """Test that both radii zero is rejected."""
        with pytest.raises(ValueError, match="cannot have both radii zero"):
            Cone(p1=(0, 0, 0), p2=(0, 0, 1), r1=0, r2=0)

    def test_construction_rejects_coincident_points(self):
        """Test that coincident endpoints are rejected."""
        with pytest.raises(ValueError, match="must be distinct"):
            Cone(p1=(0, 0, 0), p2=(0, 0, 0), r1=1, r2=0.5)

    def test_cone_as_cylinder(self):
        """Test cone with equal radii (cylinder)."""
        cone = Cone(p1=(0, 0, 0), p2=(0, 0, 0.1), r1=0.01, r2=0.01)

        # Point on axis should be at distance -r
        points = np.array([[0, 0, 0.05]])
        distances = cone.sdf(points)

        assert abs(distances[0] + 0.01) < 1e-10

    def test_full_cone(self):
        """Test full cone (r2 = 0)."""
        cone = Cone(p1=(0, 0, 0), p2=(0, 0, 0.1), r1=0.02, r2=0.0)

        # Point at tip should be on surface
        points = np.array([[0, 0, 0.1]])
        distances = cone.sdf(points)

        assert abs(distances[0]) < 1e-9

    def test_sdf_axis_point(self):
        """Test SDF for point on cone axis."""
        cone = Cone(p1=(0, 0, 0), p2=(0, 0, 0.1), r1=0.02, r2=0.01)

        # Point on axis at midpoint
        points = np.array([[0, 0, 0.05]])
        distances = cone.sdf(points)

        # Should be inside (negative distance)
        assert distances[0] < 0

    def test_bounding_box(self):
        """Test bounding box computation."""
        cone = Cone(p1=(0, 0, 0), p2=(0, 0, 0.1), r1=0.02, r2=0.01)

        min_corner, max_corner = cone.bounding_box

        # Bounding box is conservative (uses max radius in all directions)
        np.testing.assert_allclose(min_corner, [-0.02, -0.02, -0.02])
        np.testing.assert_allclose(max_corner, [0.02, 0.02, 0.12])

    def test_voxelize_uniform_grid(self):
        """Test voxelization with uniform grid."""
        cone = Cone(p1=(0.05, 0.05, 0.04), p2=(0.05, 0.05, 0.06), r1=0.01, r2=0.005)
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        mask = cone.voxelize(grid)

        assert mask.shape == (100, 100, 100)
        assert mask.any()  # Some cells should be inside
        assert not mask.all()  # Some cells should be outside


# =============================================================================
# Integration Tests
# =============================================================================


class TestSDFIntegration:
    def test_multiple_shapes_voxelize(self):
        """Test voxelizing multiple shapes to the same grid."""
        grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)

        box = Box(center=(0.03, 0.03, 0.03), size=(0.02, 0.02, 0.02))
        sphere = Sphere(center=(0.07, 0.07, 0.07), radius=0.01)

        box_mask = box.voxelize(grid)
        sphere_mask = sphere.voxelize(grid)

        # Masks should be different
        assert not np.array_equal(box_mask, sphere_mask)

        # Union: either shape
        union_mask = box_mask | sphere_mask
        assert union_mask.any()

        # Intersection: both shapes (should be empty since they don't overlap)
        intersection_mask = box_mask & sphere_mask
        assert not intersection_mask.any()

    def test_csg_operations_via_sdf(self):
        """Test CSG operations using SDF min/max."""
        # Create two overlapping spheres
        s1 = Sphere(center=(0.05, 0.05, 0.05), radius=0.015)
        s2 = Sphere(center=(0.06, 0.05, 0.05), radius=0.015)

        # Test point in overlap region
        points = np.array([[0.055, 0.05, 0.05]])

        d1 = s1.sdf(points)
        d2 = s2.sdf(points)

        # Union: min distance (closest surface)
        union_dist = np.minimum(d1, d2)
        assert union_dist[0] < 0  # Inside union

        # Intersection: max distance (furthest surface)
        intersection_dist = np.maximum(d1, d2)
        assert intersection_dist[0] < 0  # Inside intersection

    def test_batch_evaluation(self):
        """Test SDF evaluation with large batch of points."""
        sphere = Sphere(center=(0, 0, 0), radius=1.0)

        # Generate random points
        np.random.seed(42)
        points = np.random.randn(1000, 3)

        distances = sphere.sdf(points)

        # Check shape
        assert distances.shape == (1000,)

        # Points with distance < 0 should satisfy contains()
        inside = sphere.contains(points)
        np.testing.assert_array_equal(inside, distances <= 0)

    def test_nonuniform_grid_voxelization(self):
        """Test that all primitives work with nonuniform grids."""
        grid = NonuniformGrid.from_regions(
            x_regions=[(0, 0.05, 2e-3), (0.05, 0.1, 1e-3)],
            y_regions=[(0, 0.1, 1e-3)],
            z_regions=[(0, 0.1, 1e-3)],
        )

        box = Box(center=(0.075, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        sphere = Sphere(center=(0.075, 0.05, 0.05), radius=0.01)
        cyl = Cylinder(p1=(0.075, 0.05, 0.04), p2=(0.075, 0.05, 0.06), radius=0.005)
        cone = Cone(p1=(0.075, 0.05, 0.04), p2=(0.075, 0.05, 0.06), r1=0.01, r2=0.005)

        # All should voxelize without error
        box_mask = box.voxelize(grid)
        sphere_mask = sphere.voxelize(grid)
        cyl_mask = cyl.voxelize(grid)
        cone_mask = cone.voxelize(grid)

        # All should have some True and some False
        for mask in [box_mask, sphere_mask, cyl_mask, cone_mask]:
            assert mask.any()
            assert not mask.all()


# =============================================================================
# SpiralHorn Tests
# =============================================================================


class TestSpiralHorn:
    def test_construction(self):
        """Test spiral horn construction."""
        horn = SpiralHorn(
            center=(0.15, 0.15),
            throat_radius=0.025,
            mouth_radius=0.075,
            inner_spiral_radius=0.04,
            outer_spiral_radius=0.12,
            turns=2.5,
            height=0.15,
            profile="exponential",
        )

        assert horn.throat_radius == 0.025
        assert horn.mouth_radius == 0.075
        assert horn.turns == 2.5

    def test_construction_validates_parameters(self):
        """Test parameter validation."""
        # Invalid throat radius
        with pytest.raises(ValueError, match="radii must be positive"):
            SpiralHorn(
                center=(0.15, 0.15),
                throat_radius=-0.01,
                mouth_radius=0.075,
                inner_spiral_radius=0.04,
                outer_spiral_radius=0.12,
                turns=2.5,
                height=0.15,
            )

        # Invalid height
        with pytest.raises(ValueError, match="Height must be positive"):
            SpiralHorn(
                center=(0.15, 0.15),
                throat_radius=0.025,
                mouth_radius=0.075,
                inner_spiral_radius=0.04,
                outer_spiral_radius=0.12,
                turns=2.5,
                height=0.0,
            )

        # Invalid profile
        with pytest.raises(ValueError, match="profile must be"):
            SpiralHorn(
                center=(0.15, 0.15),
                throat_radius=0.025,
                mouth_radius=0.075,
                inner_spiral_radius=0.04,
                outer_spiral_radius=0.12,
                turns=2.5,
                height=0.15,
                profile="invalid",
            )

    def test_bounding_box(self):
        """Test bounding box computation."""
        horn = SpiralHorn(
            center=(0.15, 0.15),
            throat_radius=0.025,
            mouth_radius=0.075,
            inner_spiral_radius=0.04,
            outer_spiral_radius=0.12,
            turns=2.5,
            height=0.15,
        )

        bb_min, bb_max = horn.bounding_box

        # Box should contain spiral extents plus max radius
        max_r = 0.075
        expected_min = np.array([0.15 - 0.12 - max_r, 0.15 - 0.12 - max_r, 0.0])
        expected_max = np.array([0.15 + 0.12 + max_r, 0.15 + 0.12 + max_r, 0.15])

        np.testing.assert_allclose(bb_min, expected_min)
        np.testing.assert_allclose(bb_max, expected_max)

    def test_sdf_evaluation(self):
        """Test SDF evaluation doesn't crash."""
        horn = SpiralHorn(
            center=(0.15, 0.15),
            throat_radius=0.025,
            mouth_radius=0.075,
            inner_spiral_radius=0.04,
            outer_spiral_radius=0.12,
            turns=2.5,
            height=0.15,
        )

        # Test with a few points
        points = np.array([
            [0.15, 0.15, 0.075],  # Near center
            [0.2, 0.2, 0.075],  # Off to side
            [0.15, 0.19, 0.075],  # On path
        ])

        distances = horn.sdf(points)

        # Should return valid distances
        assert distances.shape == (3,)
        assert np.all(np.isfinite(distances))

    def test_voxelize(self):
        """Test voxelization to grid."""
        horn = SpiralHorn(
            center=(0.05, 0.05),
            throat_radius=0.01,
            mouth_radius=0.02,
            inner_spiral_radius=0.015,
            outer_spiral_radius=0.04,
            turns=1.5,
            height=0.08,
        )

        grid = UniformGrid(shape=(100, 100, 80), resolution=1e-3)
        mask = horn.voxelize(grid)

        # Should have some inside and some outside
        assert mask.any()
        assert not mask.all()

    def test_different_profiles(self):
        """Test different expansion profiles."""
        for profile in ["exponential", "linear", "hyperbolic"]:
            horn = SpiralHorn(
                center=(0.05, 0.05),
                throat_radius=0.01,
                mouth_radius=0.03,
                inner_spiral_radius=0.02,
                outer_spiral_radius=0.04,
                turns=2.0,
                height=0.08,
                profile=profile,
            )

            # Check that radii interpolation works
            assert len(horn.radii) > 0
            assert horn.radii[0] == pytest.approx(0.01, abs=1e-6)
            assert horn.radii[-1] == pytest.approx(0.03, abs=1e-6)


# =============================================================================
# HelicalTube Tests
# =============================================================================


class TestHelicalTube:
    def test_construction(self):
        """Test helical tube construction."""
        tube = HelicalTube(
            center=(0.1, 0.1, 0.0),
            helix_radius=0.06,
            tube_radius_start=0.02,
            tube_radius_end=0.02,
            pitch=0.04,
            turns=3.0,
        )

        assert tube.helix_radius == 0.06
        assert tube.tube_radius_start == 0.02
        assert tube.tube_radius_end == 0.02
        assert tube.pitch == 0.04
        assert tube.turns == 3.0

    def test_construction_validates_parameters(self):
        """Test parameter validation."""
        # Invalid helix radius
        with pytest.raises(ValueError, match="helix_radius must be positive"):
            HelicalTube(
                center=(0.1, 0.1, 0.0),
                helix_radius=0.0,
                tube_radius_start=0.02,
                tube_radius_end=0.02,
                pitch=0.04,
                turns=3.0,
            )

        # Invalid tube radii
        with pytest.raises(ValueError, match="tube radii must be positive"):
            HelicalTube(
                center=(0.1, 0.1, 0.0),
                helix_radius=0.06,
                tube_radius_start=-0.01,
                tube_radius_end=0.02,
                pitch=0.04,
                turns=3.0,
            )

        # Invalid pitch
        with pytest.raises(ValueError, match="pitch must be positive"):
            HelicalTube(
                center=(0.1, 0.1, 0.0),
                helix_radius=0.06,
                tube_radius_start=0.02,
                tube_radius_end=0.02,
                pitch=0.0,
                turns=3.0,
            )

    def test_bounding_box(self):
        """Test bounding box computation."""
        tube = HelicalTube(
            center=(0.1, 0.1, 0.0),
            helix_radius=0.06,
            tube_radius_start=0.02,
            tube_radius_end=0.02,
            pitch=0.04,
            turns=3.0,
        )

        bb_min, bb_max = tube.bounding_box

        # Box should contain helix + tube radius
        total_r = 0.06 + 0.02
        expected_min = np.array([0.1 - total_r, 0.1 - total_r, 0.0])
        expected_max = np.array([0.1 + total_r, 0.1 + total_r, 0.12])  # 3 * 0.04

        np.testing.assert_allclose(bb_min, expected_min)
        np.testing.assert_allclose(bb_max, expected_max)

    def test_sdf_evaluation(self):
        """Test SDF evaluation doesn't crash."""
        tube = HelicalTube(
            center=(0.1, 0.1, 0.0),
            helix_radius=0.06,
            tube_radius_start=0.02,
            tube_radius_end=0.02,
            pitch=0.04,
            turns=3.0,
        )

        # Test with a few points
        points = np.array([
            [0.1, 0.1, 0.06],  # At center
            [0.16, 0.1, 0.0],  # On helix path (approximately)
            [0.2, 0.2, 0.06],  # Far from helix
        ])

        distances = tube.sdf(points)

        # Should return valid distances
        assert distances.shape == (3,)
        assert np.all(np.isfinite(distances))

    def test_voxelize(self):
        """Test voxelization to grid."""
        tube = HelicalTube(
            center=(0.05, 0.05, 0.0),
            helix_radius=0.03,
            tube_radius_start=0.01,
            tube_radius_end=0.01,
            pitch=0.02,
            turns=2.0,
        )

        grid = UniformGrid(shape=(100, 100, 40), resolution=1e-3)
        mask = tube.voxelize(grid)

        # Should have some inside and some outside
        assert mask.any()
        assert not mask.all()

    def test_tapered_tube(self):
        """Test tapered helical tube."""
        tube = HelicalTube(
            center=(0.05, 0.05, 0.0),
            helix_radius=0.03,
            tube_radius_start=0.005,
            tube_radius_end=0.015,
            pitch=0.02,
            turns=2.0,
        )

        # Check that tube radii taper correctly
        assert len(tube.tube_radii) > 0
        assert tube.tube_radii[0] == pytest.approx(0.005, abs=1e-6)
        assert tube.tube_radii[-1] == pytest.approx(0.015, abs=1e-6)

        # Radii should increase monotonically
        assert np.all(np.diff(tube.tube_radii) >= 0)


# =============================================================================
# Horn Tests
# =============================================================================


class TestHorn:
    """Tests for Horn primitive with various flare profiles."""

    def test_construction_conical(self):
        """Test construction of conical horn."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.02,
            profile="conical",
        )

        np.testing.assert_allclose(horn.throat_position, [0, 0, 0.1])
        np.testing.assert_allclose(horn.mouth_position, [0, 0, 0])
        assert horn.throat_radius == 0.01
        assert horn.mouth_radius == 0.02
        assert horn.profile == "conical"

    def test_construction_exponential(self):
        """Test construction of exponential horn."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.025,
            mouth_radius=0.075,
            profile="exponential",
        )

        assert horn.profile == "exponential"
        assert hasattr(horn, "_alpha")

    def test_construction_hyperbolic(self):
        """Test construction of hyperbolic horn."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.02,
            mouth_radius=0.06,
            profile="hyperbolic",
        )

        assert horn.profile == "hyperbolic"
        assert hasattr(horn, "_alpha")

    def test_construction_tractrix(self):
        """Test construction of tractrix horn."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.025,
            mouth_radius=0.08,
            profile="tractrix",
        )

        assert horn.profile == "tractrix"
        assert hasattr(horn, "_R")

    def test_invalid_profile(self):
        """Test that invalid profile raises error."""
        with pytest.raises(ValueError, match="profile must be one of"):
            Horn(
                throat_position=(0, 0, 0.1),
                mouth_position=(0, 0, 0),
                throat_radius=0.01,
                mouth_radius=0.02,
                profile="invalid",
            )

    def test_invalid_radii(self):
        """Test that invalid radii raise errors."""
        with pytest.raises(ValueError, match="must be positive"):
            Horn(
                throat_position=(0, 0, 0.1),
                mouth_position=(0, 0, 0),
                throat_radius=-0.01,
                mouth_radius=0.02,
                profile="conical",
            )

        with pytest.raises(ValueError, match="must be positive"):
            Horn(
                throat_position=(0, 0, 0.1),
                mouth_position=(0, 0, 0),
                throat_radius=0.01,
                mouth_radius=0,
                profile="conical",
            )

    def test_identical_positions(self):
        """Test that identical throat/mouth positions raise error."""
        with pytest.raises(ValueError, match="must be distinct"):
            Horn(
                throat_position=(0, 0, 0),
                mouth_position=(0, 0, 0),
                throat_radius=0.01,
                mouth_radius=0.02,
                profile="conical",
            )

    def test_radius_at_position_conical(self):
        """Test radius calculation for conical profile."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # At throat (t=0)
        r_throat = horn.radius_at_position(0.0)
        np.testing.assert_allclose(r_throat, 0.01, rtol=1e-6)

        # At mouth (t=1)
        r_mouth = horn.radius_at_position(1.0)
        np.testing.assert_allclose(r_mouth, 0.03, rtol=1e-6)

        # At midpoint (t=0.5), should be linear interpolation
        r_mid = horn.radius_at_position(0.5)
        assert r_throat < r_mid < r_mouth

    def test_radius_at_position_exponential(self):
        """Test radius calculation for exponential profile."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.04,
            profile="exponential",
        )

        # At throat (t=0)
        r_throat = horn.radius_at_position(0.0)
        np.testing.assert_allclose(r_throat, 0.01, rtol=1e-6)

        # At mouth (t=1)
        r_mouth = horn.radius_at_position(1.0)
        np.testing.assert_allclose(r_mouth, 0.04, rtol=1e-6)

        # Exponential should have increasing rate of expansion
        r_quarter = horn.radius_at_position(0.25)
        r_half = horn.radius_at_position(0.5)
        r_three_quarter = horn.radius_at_position(0.75)

        # Check exponential growth (second differences should increase)
        d1 = r_half - r_quarter
        d2 = r_three_quarter - r_half
        assert d2 > d1  # Exponential growth accelerates

    def test_radius_at_position_array(self):
        """Test radius calculation with array input."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        radii = horn.radius_at_position(t_values)

        assert radii.shape == (5,)
        np.testing.assert_allclose(radii[0], 0.01, rtol=1e-6)
        np.testing.assert_allclose(radii[-1], 0.03, rtol=1e-6)

    def test_cutoff_frequency_conical(self):
        """Test cutoff frequency calculation for conical horn."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        fc = horn.cutoff_frequency
        # fc ≈ c / (2π * L) = 343 / (2π * 0.1) ≈ 546 Hz
        expected = 343.0 / (2 * np.pi * 0.1)
        np.testing.assert_allclose(fc, expected, rtol=1e-6)

    def test_cutoff_frequency_exponential(self):
        """Test cutoff frequency calculation for exponential horn."""
        horn = Horn(
            throat_position=(0, 0, 0.2),
            mouth_position=(0, 0, 0),
            throat_radius=0.025,
            mouth_radius=0.075,
            profile="exponential",
        )

        fc = horn.cutoff_frequency
        # fc = c * alpha / (2π), alpha = ln(Am/A0) / (2*L)
        # Should be positive and reasonable (100-1000 Hz range)
        assert 50 < fc < 2000

    def test_sdf_on_axis_throat(self):
        """Test SDF for point on axis at throat."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point on axis at throat
        points = np.array([[0, 0, 0.1]])
        distances = horn.sdf(points)

        # Should be inside (negative distance of -throat_radius)
        np.testing.assert_allclose(distances[0], -0.01, rtol=1e-6)

    def test_sdf_on_surface_throat(self):
        """Test SDF for point on surface at throat."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point on surface at throat (radial distance = throat_radius)
        points = np.array([[0.01, 0, 0.1]])
        distances = horn.sdf(points)

        # Should be on or very near surface
        assert abs(distances[0]) < 1e-9

    def test_sdf_on_surface_mouth(self):
        """Test SDF for point on surface at mouth."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point on surface at mouth (radial distance = mouth_radius)
        points = np.array([[0.03, 0, 0]])
        distances = horn.sdf(points)

        # Should be on or very near surface
        assert abs(distances[0]) < 1e-9

    def test_sdf_outside_radially(self):
        """Test SDF for point outside horn radially."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point 0.01m outside mouth radially
        points = np.array([[0.04, 0, 0]])
        distances = horn.sdf(points)

        # Should be positive (outside)
        assert distances[0] > 0
        np.testing.assert_allclose(distances[0], 0.01, rtol=1e-6)

    def test_sdf_beyond_throat(self):
        """Test SDF for point beyond throat axially."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point beyond throat on axis
        points = np.array([[0, 0, 0.12]])
        distances = horn.sdf(points)

        # Should be outside (positive distance)
        assert distances[0] > 0

    def test_sdf_beyond_mouth(self):
        """Test SDF for point beyond mouth axially."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point beyond mouth on axis
        points = np.array([[0, 0, -0.02]])
        distances = horn.sdf(points)

        # Should be outside (positive distance)
        assert distances[0] > 0

    def test_sdf_all_profiles(self):
        """Test that SDF works for all profile types."""
        profiles = ["conical", "exponential", "hyperbolic", "tractrix"]

        for profile in profiles:
            horn = Horn(
                throat_position=(0, 0, 0.1),
                mouth_position=(0, 0, 0),
                throat_radius=0.02,
                mouth_radius=0.06,
                profile=profile,
            )

            # Test multiple points
            points = np.array([
                [0, 0, 0.1],  # On axis at throat
                [0, 0, 0.05],  # On axis mid-horn
                [0, 0, 0],  # On axis at mouth
                [0.02, 0, 0.1],  # On surface at throat
                [0.1, 0, 0.05],  # Outside radially
            ])

            distances = horn.sdf(points)

            # Basic sanity checks
            assert distances.shape == (5,)
            assert distances[0] < 0  # On axis should be inside
            assert distances[4] > 0  # Far outside should be positive

    def test_bounding_box(self):
        """Test bounding box computation."""
        horn = Horn(
            throat_position=(0.1, 0.1, 0.2),
            mouth_position=(0.1, 0.1, 0.0),
            throat_radius=0.025,
            mouth_radius=0.075,
            profile="exponential",
        )

        bb_min, bb_max = horn.bounding_box

        # Should contain both throat and mouth
        np.testing.assert_array_less(bb_min, horn.throat_position)
        np.testing.assert_array_less(bb_min, horn.mouth_position)
        np.testing.assert_array_less(horn.throat_position, bb_max)
        np.testing.assert_array_less(horn.mouth_position, bb_max)

        # Should account for mouth radius (larger)
        expected_min = np.array([0.1, 0.1, 0.0]) - np.array([0.075, 0.075, 0.075])
        expected_max = np.array([0.1, 0.1, 0.2]) + np.array([0.075, 0.075, 0.075])

        np.testing.assert_allclose(bb_min, expected_min, rtol=1e-6)
        np.testing.assert_allclose(bb_max, expected_max, rtol=1e-6)

    def test_contains(self):
        """Test contains() method."""
        horn = Horn(
            throat_position=(0, 0, 0.1),
            mouth_position=(0, 0, 0),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        points = np.array([
            [0, 0, 0.05],  # On axis (inside)
            [0.01, 0, 0.1],  # On surface at throat
            [0.05, 0, 0.05],  # Outside radially
        ])

        inside = horn.contains(points)

        assert inside[0]  # On axis should be inside
        assert inside[1]  # On surface counts as inside (<=0)
        assert not inside[2]  # Outside should be False

    def test_voxelize_uniform_grid(self):
        """Test voxelization on uniform grid."""
        grid = UniformGrid(shape=(50, 50, 50), resolution=2e-3)

        horn = Horn(
            throat_position=(0.05, 0.05, 0.08),
            mouth_position=(0.05, 0.05, 0.02),
            throat_radius=0.01,
            mouth_radius=0.02,
            profile="exponential",
        )

        mask = horn.voxelize(grid)

        # Should have correct shape
        assert mask.shape == (50, 50, 50)

        # Should have some True and some False
        assert mask.any()
        assert not mask.all()

    def test_arbitrary_orientation(self):
        """Test horn with arbitrary orientation."""
        # Horn along x-axis
        horn_x = Horn(
            throat_position=(0, 0.05, 0.05),
            mouth_position=(0.1, 0.05, 0.05),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        # Point on axis mid-horn
        points = np.array([[0.05, 0.05, 0.05]])
        distances = horn_x.sdf(points)

        # Should be inside
        assert distances[0] < 0

        # Horn along diagonal
        horn_diag = Horn(
            throat_position=(0, 0, 0),
            mouth_position=(0.1, 0.1, 0.1),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="exponential",
        )

        # Point on axis mid-horn
        points = np.array([[0.05, 0.05, 0.05]])
        distances = horn_diag.sdf(points)

        # Should be inside
        assert distances[0] < 0


# =============================================================================
# Horn Factory Function Tests
# =============================================================================


class TestHornFactories:
    """Tests for horn factory functions."""

    def test_conical_horn_factory(self):
        """Test conical_horn factory function."""
        horn = conical_horn((0, 0, 0.1), (0, 0, 0), 0.01, 0.02)

        assert isinstance(horn, Horn)
        assert horn.profile == "conical"
        assert horn.throat_radius == 0.01
        assert horn.mouth_radius == 0.02

    def test_exponential_horn_factory(self):
        """Test exponential_horn factory function."""
        horn = exponential_horn((0, 0, 0.1), (0, 0, 0), 0.025, 0.075)

        assert isinstance(horn, Horn)
        assert horn.profile == "exponential"
        assert horn.throat_radius == 0.025
        assert horn.mouth_radius == 0.075

    def test_hyperbolic_horn_factory(self):
        """Test hyperbolic_horn factory function."""
        horn = hyperbolic_horn((0, 0, 0.1), (0, 0, 0), 0.02, 0.06)

        assert isinstance(horn, Horn)
        assert horn.profile == "hyperbolic"
        assert horn.throat_radius == 0.02
        assert horn.mouth_radius == 0.06

    def test_tractrix_horn_factory(self):
        """Test tractrix_horn factory function."""
        horn = tractrix_horn((0, 0, 0.1), (0, 0, 0), 0.025, 0.08)

        assert isinstance(horn, Horn)
        assert horn.profile == "tractrix"
        assert horn.throat_radius == 0.025
        assert horn.mouth_radius == 0.08
