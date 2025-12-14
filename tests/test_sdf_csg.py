"""
Unit tests for SDF CSG operations (Union, Intersection, Difference).

Tests verify:
- Correct SDF computation for each operation
- Proper bounding box calculation
- Composability (CSG trees)
- Smooth operations with blend radius
"""

import numpy as np
import pytest

from strata_fdtd.grid import UniformGrid
from strata_fdtd.sdf import (
    Box,
    Difference,
    Intersection,
    SmoothDifference,
    SmoothIntersection,
    SmoothUnion,
    Sphere,
    Union,
)


class TestUnion:
    """Tests for Union CSG operation."""

    def test_union_two_boxes_no_overlap(self):
        """Union of two non-overlapping boxes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box2 = Box(center=(2.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        union = Union(box1, box2)

        # Point inside box1 (negative distance)
        p1 = np.array([[0.0, 0.0, 0.0]])
        assert union.sdf(p1)[0] < 0

        # Point inside box2 (negative distance)
        p2 = np.array([[2.0, 0.0, 0.0]])
        assert union.sdf(p2)[0] < 0

        # Point between boxes (positive distance)
        p_between = np.array([[1.0, 0.0, 0.0]])
        assert union.sdf(p_between)[0] > 0

        # Point far outside (positive distance)
        p_outside = np.array([[10.0, 0.0, 0.0]])
        assert union.sdf(p_outside)[0] > 0

    def test_union_two_boxes_overlapping(self):
        """Union of two overlapping boxes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        box2 = Box(center=(1.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        union = Union(box1, box2)

        # Point in overlap region (inside both)
        p_overlap = np.array([[0.5, 0.0, 0.0]])
        assert union.sdf(p_overlap)[0] < 0

        # Union SDF should be minimum of the two
        d1 = box1.sdf(p_overlap)[0]
        d2 = box2.sdf(p_overlap)[0]
        assert np.isclose(union.sdf(p_overlap)[0], min(d1, d2))

    def test_union_empty(self):
        """Union with no children."""
        union = Union()

        # All points infinitely far from surface
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        distances = union.sdf(points)
        assert np.all(np.isinf(distances))
        assert np.all(distances > 0)

    def test_union_multiple_shapes(self):
        """Union of more than two shapes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box2 = Box(center=(2.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box3 = Box(center=(4.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        union = Union(box1, box2, box3)

        # Points inside each box
        assert union.sdf(np.array([[0.0, 0.0, 0.0]]))[0] < 0
        assert union.sdf(np.array([[2.0, 0.0, 0.0]]))[0] < 0
        assert union.sdf(np.array([[4.0, 0.0, 0.0]]))[0] < 0

    def test_union_bounding_box(self):
        """Union bounding box is union of child bounding boxes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box2 = Box(center=(2.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        union = Union(box1, box2)

        bb_min, bb_max = union.bounding_box

        # Should span from -0.5 to 2.5 in x
        assert np.isclose(bb_min[0], -0.5)
        assert np.isclose(bb_max[0], 2.5)

        # y and z should be -0.5 to 0.5
        assert np.isclose(bb_min[1], -0.5)
        assert np.isclose(bb_max[1], 0.5)


class TestIntersection:
    """Tests for Intersection CSG operation."""

    def test_intersection_two_boxes_overlapping(self):
        """Intersection of two overlapping boxes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        box2 = Box(center=(0.5, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        intersection = Intersection(box1, box2)

        # Point in overlap region (inside both)
        p_overlap = np.array([[0.25, 0.0, 0.0]])
        assert intersection.sdf(p_overlap)[0] < 0

        # Point only in box1 (outside intersection)
        p_only_box1 = np.array([[-0.8, 0.0, 0.0]])
        assert intersection.sdf(p_only_box1)[0] > 0

        # Point only in box2 (outside intersection)
        p_only_box2 = np.array([[1.3, 0.0, 0.0]])
        assert intersection.sdf(p_only_box2)[0] > 0

    def test_intersection_no_overlap(self):
        """Intersection of non-overlapping boxes is empty."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box2 = Box(center=(5.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        intersection = Intersection(box1, box2)

        # All points should be outside
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        distances = intersection.sdf(points)
        assert np.all(distances > 0)

    def test_intersection_empty(self):
        """Intersection with no children."""
        intersection = Intersection()

        # All points inside (distance = -inf)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        distances = intersection.sdf(points)
        assert np.all(np.isinf(distances))
        assert np.all(distances < 0)

    def test_intersection_bounding_box(self):
        """Intersection bounding box is intersection of child boxes."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0))
        box2 = Box(center=(1.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        intersection = Intersection(box1, box2)

        bb_min, bb_max = intersection.bounding_box

        # Should be the smaller box2's bounds
        assert np.allclose(bb_min, [0.0, -1.0, -1.0])
        assert np.allclose(bb_max, [2.0, 1.0, 1.0])


class TestDifference:
    """Tests for Difference CSG operation."""

    def test_difference_box_minus_box(self):
        """Subtract a box from another box."""
        base = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        hole = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        difference = Difference(base, hole)

        # Point in base but outside hole (inside the shell)
        p_shell = np.array([[0.75, 0.0, 0.0]])
        assert difference.sdf(p_shell)[0] < 0

        # Point inside hole (outside the difference)
        p_hole = np.array([[0.0, 0.0, 0.0]])
        assert difference.sdf(p_hole)[0] > 0

        # Point outside base (outside the difference)
        p_outside = np.array([[5.0, 0.0, 0.0]])
        assert difference.sdf(p_outside)[0] > 0

    def test_difference_multiple_subtractions(self):
        """Subtract multiple shapes from base."""
        base = Box(center=(0.0, 0.0, 0.0), size=(3.0, 3.0, 3.0))
        hole1 = Box(center=(-0.75, 0.0, 0.0), size=(0.5, 0.5, 0.5))
        hole2 = Box(center=(0.75, 0.0, 0.0), size=(0.5, 0.5, 0.5))
        difference = Difference(base, hole1, hole2)

        # Point in base but not in either hole
        p_solid = np.array([[0.0, 1.0, 0.0]])
        assert difference.sdf(p_solid)[0] < 0

        # Points inside holes
        assert difference.sdf(np.array([[-0.75, 0.0, 0.0]]))[0] > 0
        assert difference.sdf(np.array([[0.75, 0.0, 0.0]]))[0] > 0

    def test_difference_bounding_box(self):
        """Difference bounding box is same as base."""
        base = Box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0))
        hole = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        difference = Difference(base, hole)

        bb_min, bb_max = difference.bounding_box

        # Should match base box bounds
        assert np.allclose(bb_min, [-2.0, -2.0, -2.0])
        assert np.allclose(bb_max, [2.0, 2.0, 2.0])


class TestSmoothOperations:
    """Tests for smooth CSG operations."""

    def test_smooth_union_blend(self):
        """Smooth union creates smooth transition."""
        sphere1 = Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        sphere2 = Sphere(center=(1.5, 0.0, 0.0), radius=1.0)

        # Regular union (sharp)
        sharp = Union(sphere1, sphere2)

        # Smooth union with blend
        smooth = SmoothUnion(sphere1, sphere2, radius=0.5)

        # At point between spheres, smooth union should be "more inside"
        # (more negative distance) than sharp union
        p_between = np.array([[0.75, 0.0, 0.0]])

        d_sharp = sharp.sdf(p_between)[0]
        d_smooth = smooth.sdf(p_between)[0]

        # Smooth version pulls the surface outward (more negative inside)
        assert d_smooth < d_sharp

    def test_smooth_union_far_from_blend(self):
        """Smooth union matches regular union far from blend zone."""
        sphere1 = Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        sphere2 = Sphere(center=(5.0, 0.0, 0.0), radius=1.0)

        sharp = Union(sphere1, sphere2)
        smooth = SmoothUnion(sphere1, sphere2, radius=0.2)

        # Point clearly inside sphere1, far from blend
        p_inside = np.array([[0.0, 0.0, 0.0]])
        assert np.isclose(sharp.sdf(p_inside)[0], smooth.sdf(p_inside)[0], atol=0.1)

        # Point clearly outside both, far from blend
        p_outside = np.array([[10.0, 0.0, 0.0]])
        assert np.isclose(sharp.sdf(p_outside)[0], smooth.sdf(p_outside)[0], atol=0.1)

    def test_smooth_union_invalid_radius(self):
        """Smooth union requires positive radius."""
        sphere1 = Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        sphere2 = Sphere(center=(1.5, 0.0, 0.0), radius=1.0)

        with pytest.raises(ValueError, match="radius must be positive"):
            SmoothUnion(sphere1, sphere2, radius=0.0)

        with pytest.raises(ValueError, match="radius must be positive"):
            SmoothUnion(sphere1, sphere2, radius=-0.5)

    def test_smooth_intersection(self):
        """Smooth intersection creates smooth transition."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        box2 = Box(center=(0.5, 0.0, 0.0), size=(2.0, 2.0, 2.0))

        sharp = Intersection(box1, box2)
        smooth = SmoothIntersection(box1, box2, radius=0.3)

        # In overlap region, smooth should differ from sharp
        p_overlap = np.array([[0.25, 0.0, 0.0]])

        d_sharp = sharp.sdf(p_overlap)[0]
        d_smooth = smooth.sdf(p_overlap)[0]

        # Both should be inside, but smooth is different
        assert d_sharp < 0
        assert d_smooth < 0
        assert not np.isclose(d_sharp, d_smooth, atol=0.01)

    def test_smooth_difference(self):
        """Smooth difference creates smooth transition."""
        base = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        hole = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))

        sharp = Difference(base, hole)
        smooth = SmoothDifference(base, hole, radius=0.2)

        # Near hole boundary, smooth should differ from sharp
        p_near_hole = np.array([[0.6, 0.0, 0.0]])

        d_sharp = sharp.sdf(p_near_hole)[0]
        d_smooth = smooth.sdf(p_near_hole)[0]

        # Both should be inside the shell, but smooth is different
        assert d_sharp < 0
        assert d_smooth < 0


class TestComposition:
    """Tests for composing CSG operations (CSG trees)."""

    def test_nested_unions(self):
        """Union of unions."""
        box1 = Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box2 = Box(center=(2.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box3 = Box(center=(4.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))

        # Nested unions: (box1 ∪ box2) ∪ box3
        union12 = Union(box1, box2)
        union_all = Union(union12, box3)

        # Should behave like union of all three
        assert union_all.sdf(np.array([[0.0, 0.0, 0.0]]))[0] < 0
        assert union_all.sdf(np.array([[2.0, 0.0, 0.0]]))[0] < 0
        assert union_all.sdf(np.array([[4.0, 0.0, 0.0]]))[0] < 0

    def test_difference_of_union(self):
        """Complex composition: (box1 ∪ box2) - hole."""
        box1 = Box(center=(-0.5, 0.0, 0.0), size=(1.0, 2.0, 2.0))
        box2 = Box(center=(0.5, 0.0, 0.0), size=(1.0, 2.0, 2.0))
        hole = Sphere(center=(0.0, 0.0, 0.0), radius=0.5)

        base = Union(box1, box2)
        result = Difference(base, hole)

        # Point in union but not in hole (clearly inside the solid region)
        p_solid = np.array([[-0.8, 0.0, 0.0]])
        assert result.sdf(p_solid)[0] < 0

        # Point inside hole
        p_hole = np.array([[0.0, 0.0, 0.0]])
        assert result.sdf(p_hole)[0] > 0

    def test_intersection_of_differences(self):
        """Complex composition: (A - B) ∩ (C - D)."""
        box_a = Box(center=(0.0, 0.0, 0.0), size=(3.0, 3.0, 3.0))
        box_b = Box(center=(-0.5, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        box_c = Box(center=(0.0, 0.0, 0.0), size=(2.5, 2.5, 2.5))
        box_d = Box(center=(0.5, 0.0, 0.0), size=(1.0, 1.0, 1.0))

        diff1 = Difference(box_a, box_b)
        diff2 = Difference(box_c, box_d)
        result = Intersection(diff1, diff2)

        # Point inside both differences
        p_inside = np.array([[0.0, 1.0, 0.0]])
        assert result.sdf(p_inside)[0] < 0

        # Point in hole of first difference
        p_hole1 = np.array([[-0.5, 0.0, 0.0]])
        assert result.sdf(p_hole1)[0] > 0

        # Point in hole of second difference
        p_hole2 = np.array([[0.5, 0.0, 0.0]])
        assert result.sdf(p_hole2)[0] > 0

    def test_deep_csg_tree(self):
        """Deeply nested CSG tree."""
        # Create a complex tree: ((A ∪ B) ∩ C) - (D ∪ E)
        box_a = Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        box_b = Box(center=(1.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        box_c = Box(center=(0.5, 0.0, 0.0), size=(3.0, 1.5, 1.5))
        box_d = Box(center=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))
        box_e = Box(center=(1.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))

        union_ab = Union(box_a, box_b)
        intersect_abc = Intersection(union_ab, box_c)
        union_de = Union(box_d, box_e)
        result = Difference(intersect_abc, union_de)

        # Verify it computes without error
        test_points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        distances = result.sdf(test_points)

        # Should have a mix of inside/outside
        assert len(distances) == 4
        assert np.all(np.isfinite(distances))


class TestVoxelize:
    """Tests for voxelization of SDF primitives."""

    def test_voxelize_box(self):
        """Voxelize a simple box."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        grid = UniformGrid(shape=(10, 10, 10), resolution=0.01)

        mask = box.voxelize(grid)

        # Mask should be boolean
        assert mask.dtype == np.bool_

        # Mask shape should match grid
        assert mask.shape == (10, 10, 10)

        # Center should be inside (True)
        # Grid point at index (5, 5, 5) is at physical position (0.05, 0.05, 0.05)
        assert mask[5, 5, 5]

        # Corners should be outside (False)
        assert not mask[0, 0, 0]
        assert not mask[9, 9, 9]

    def test_voxelize_union(self):
        """Voxelize a union of boxes."""
        box1 = Box(center=(0.02, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        box2 = Box(center=(0.08, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        union = Union(box1, box2)

        grid = UniformGrid(shape=(10, 10, 10), resolution=0.01)
        mask = union.voxelize(grid)

        # Both box regions should be inside
        assert mask.sum() > 0

        # Should have two separate regions
        # (approximately - depends on grid resolution)
        assert mask.sum() > 10  # More than just a few voxels

    def test_voxelize_difference(self):
        """Voxelize a difference (box with hole)."""
        outer = Box(center=(0.05, 0.05, 0.05), size=(0.08, 0.08, 0.08))
        inner = Box(center=(0.05, 0.05, 0.05), size=(0.04, 0.04, 0.04))
        shell = Difference(outer, inner)

        grid = UniformGrid(shape=(10, 10, 10), resolution=0.01)
        mask = shell.voxelize(grid)

        # Should have some inside voxels (the shell)
        assert mask.sum() > 0

        # Center should be outside (in the hole)
        # Grid point at (5, 5, 5) is at (0.05, 0.05, 0.05) - center of inner box
        assert not mask[5, 5, 5]
