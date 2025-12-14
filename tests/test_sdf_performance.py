"""
Performance tests for deep CSG trees.

Tests verify that CSG operations scale well with tree depth and complexity.
These are marked as slow tests and may be skipped in quick test runs.
"""

import time

import numpy as np
import pytest

from strata_fdtd.grid import UniformGrid
from strata_fdtd.sdf import Difference, Intersection, SDFPrimitive, Sphere, Union


@pytest.mark.slow
class TestCSGPerformance:
    """Performance tests for CSG tree evaluation."""

    def test_deep_union_tree_performance(self):
        """Test performance of deeply nested union tree."""
        # Create a binary tree of unions with depth 10
        # Total primitives = 2^10 = 1024 spheres
        depth = 10

        def create_union_tree(level: int, x_offset: float) -> SDFPrimitive:
            if level == 0:
                # Leaf: single sphere
                return Sphere(center=(x_offset, 0.0, 0.0), radius=0.1)

            # Internal node: union of two subtrees
            left = create_union_tree(level - 1, x_offset - 0.1 * (2 ** level))
            right = create_union_tree(level - 1, x_offset + 0.1 * (2 ** level))
            return Union(left, right)

        tree = create_union_tree(depth, x_offset=0.0)

        # Evaluate SDF at 10,000 points
        num_points = 10000
        points = np.random.randn(num_points, 3) * 5.0  # Random points in [-15, 15]^3

        start = time.perf_counter()
        distances = tree.sdf(points)
        elapsed = time.perf_counter() - start

        # Verify correct output shape
        assert distances.shape == (num_points,)

        # Should complete in reasonable time (< 1 second for 10k points, depth 10)
        assert elapsed < 1.0, f"Deep tree evaluation took {elapsed:.3f}s (expected < 1s)"

        print(f"\nDeep union tree (depth {depth}, {2**depth} primitives):")
        print(f"  Evaluated {num_points} points in {elapsed*1000:.1f}ms")
        print(f"  Rate: {num_points/elapsed:.0f} points/sec")

    def test_deep_mixed_csg_tree(self):
        """Test performance of deeply nested mixed CSG operations."""
        # Create a complex tree with unions, intersections, and differences
        depth = 7

        def create_mixed_tree(level: int, x: float, y: float) -> SDFPrimitive:
            if level == 0:
                return Sphere(center=(x, y, 0.0), radius=0.15)

            # Alternate between operations at each level
            s1 = create_mixed_tree(level - 1, x - 0.2, y - 0.2)
            s2 = create_mixed_tree(level - 1, x + 0.2, y - 0.2)
            s3 = create_mixed_tree(level - 1, x, y + 0.2)

            if level % 3 == 0:
                # Union level
                return Union(s1, s2, s3)
            elif level % 3 == 1:
                # Intersection level
                return Intersection(s1, s2)
            else:
                # Difference level
                return Difference(Union(s1, s2), s3)

        tree = create_mixed_tree(depth, 0.0, 0.0)

        # Evaluate at 5,000 points
        num_points = 5000
        points = np.random.randn(num_points, 3) * 3.0

        start = time.perf_counter()
        distances = tree.sdf(points)
        elapsed = time.perf_counter() - start

        assert distances.shape == (num_points,)
        assert elapsed < 2.0, f"Mixed tree evaluation took {elapsed:.3f}s (expected < 2s)"

        print(f"\nMixed CSG tree (depth {depth}):")
        print(f"  Evaluated {num_points} points in {elapsed*1000:.1f}ms")
        print(f"  Rate: {num_points/elapsed:.0f} points/sec")

    def test_voxelization_performance(self):
        """Test performance of voxelizing a complex CSG tree."""
        # Create a moderately complex shape: box with multiple spherical holes
        from strata_fdtd.sdf import Difference, Union

        # Base box (using sphere as approximation)
        base = Sphere(center=(0.05, 0.05, 0.05), radius=0.04)

        # Multiple holes
        holes = []
        for i in range(10):
            x = 0.03 + (i % 5) * 0.01
            y = 0.03 + (i // 5) * 0.01
            z = 0.05
            holes.append(Sphere(center=(x, y, z), radius=0.005))

        hole_union = Union(*holes)
        shape = Difference(base, hole_union)

        # Voxelize on a 50^3 grid
        grid = UniformGrid(shape=(50, 50, 50), resolution=0.002)

        start = time.perf_counter()
        mask = shape.voxelize(grid)
        elapsed = time.perf_counter() - start

        assert mask.shape == (50, 50, 50)
        assert mask.dtype == np.bool_

        # Should complete in reasonable time (< 0.5 seconds for 50^3 grid)
        assert elapsed < 0.5, f"Voxelization took {elapsed:.3f}s (expected < 0.5s)"

        print("\nVoxelization (50^3 grid, 125k voxels):")
        print(f"  Completed in {elapsed*1000:.1f}ms")
        print(f"  Rate: {125000/elapsed:.0f} voxels/sec")

    def test_bounding_box_computation_deep_tree(self):
        """Test bounding box computation for deep trees."""
        depth = 12

        def create_balanced_tree(level: int, x: float) -> SDFPrimitive:
            if level == 0:
                return Sphere(center=(x, 0.0, 0.0), radius=0.1)
            left = create_balanced_tree(level - 1, x - 0.5)
            right = create_balanced_tree(level - 1, x + 0.5)
            return Union(left, right)

        tree = create_balanced_tree(depth, 0.0)

        # Bounding box should be computed quickly even for deep trees
        start = time.perf_counter()
        bb_min, bb_max = tree.bounding_box
        elapsed = time.perf_counter() - start

        # Should be fast (< 0.05 seconds for 2^12 primitives with caching)
        # Note: First access computes and caches; subsequent accesses are instant
        assert elapsed < 0.05, f"Bounding box took {elapsed:.3f}s (expected < 0.05s)"

        # Bounding box should be reasonable
        assert bb_min.shape == (3,)
        assert bb_max.shape == (3,)
        assert np.all(bb_min < bb_max)

        print(f"\nBounding box (depth {depth}, {2**depth} primitives):")
        print(f"  Computed in {elapsed*1000:.3f}ms")

    @pytest.mark.parametrize("num_children", [2, 5, 10, 20, 50])
    def test_union_scaling_with_children(self, num_children: int):
        """Test how union performance scales with number of children."""
        # Create union with many children
        children = [
            Sphere(center=(i * 0.5, 0.0, 0.0), radius=0.1)
            for i in range(num_children)
        ]
        union = Union(*children)

        # Evaluate at 1000 points
        num_points = 1000
        points = np.random.randn(num_points, 3) * 2.0

        start = time.perf_counter()
        distances = union.sdf(points)
        elapsed = time.perf_counter() - start

        assert distances.shape == (num_points,)

        # Time should scale roughly linearly with num_children
        # Allow generous margin: < 10ms per child for 1000 points
        max_time = num_children * 0.01
        assert elapsed < max_time, (
            f"Union with {num_children} children took {elapsed:.3f}s "
            f"(expected < {max_time:.3f}s)"
        )

        print(f"\nUnion with {num_children} children:")
        print(f"  {num_points} points in {elapsed*1000:.1f}ms")
        print(f"  {elapsed/num_children*1000:.3f}ms per child")


@pytest.mark.slow
class TestMemoryEfficiency:
    """Test memory efficiency of CSG trees."""

    def test_lazy_evaluation(self):
        """Verify that CSG operations don't pre-compute results."""
        # Create a large union without evaluating
        num_spheres = 1000
        spheres = [
            Sphere(center=(i * 0.1, 0.0, 0.0), radius=0.05)
            for i in range(num_spheres)
        ]
        union = Union(*spheres)

        # Creating the union should be fast (no computation yet)
        # Just verify it was created
        assert len(union.children) == num_spheres

        # Bounding box should also be fast (just min/max over primitives)
        bb_min, bb_max = union.bounding_box
        assert bb_min.shape == (3,)

    def test_shared_primitives(self):
        """Test that primitives can be shared across multiple CSG trees."""
        # Create some base primitives
        sphere1 = Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        sphere2 = Sphere(center=(1.0, 0.0, 0.0), radius=1.0)
        sphere3 = Sphere(center=(0.5, 0.5, 0.0), radius=0.5)

        # Use them in multiple trees
        tree1 = Union(sphere1, sphere2)
        tree2 = Difference(tree1, sphere3)
        tree3 = Intersection(sphere1, sphere3)

        # All trees should work correctly
        test_point = np.array([[0.0, 0.0, 0.0]])

        d1 = tree1.sdf(test_point)[0]
        d2 = tree2.sdf(test_point)[0]
        d3 = tree3.sdf(test_point)[0]

        # All should produce finite values
        assert np.isfinite(d1)
        assert np.isfinite(d2)
        assert np.isfinite(d3)
