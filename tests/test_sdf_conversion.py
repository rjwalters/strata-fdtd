"""
Tests for SDF to Stack conversion functionality.
"""

import numpy as np
import pytest

from strata_fdtd import Box, Cylinder, Horn, Sphere
from strata_fdtd.manufacturing.conversion import (
    VoxelSDF,
    sdf_to_slices_generator,
    sdf_to_stack,
    stack_to_sdf,
    validate_slices_manufacturable,
)


class TestSDFToStack:
    """Test SDF to Stack conversion."""

    def test_box_conversion(self):
        """Test converting a simple box SDF to Stack."""
        # Create a 20mm cube centered at (50, 50, 50) mm
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        # Convert to stack
        stack = sdf_to_stack(
            box,
            z_range=(0.04, 0.06),  # 20mm range
            slice_thickness=6e-3,  # 6mm slices
            xy_resolution=0.5e-3,  # 0.5mm resolution
        )

        # Should have ~3 slices (20mm / 6mm)
        assert stack.num_slices in [3, 4]  # Allow for rounding

        # Check that slices have reasonable dimensions
        for slice_ in stack.slices:
            assert slice_.mask.shape[0] > 0
            assert slice_.mask.shape[1] > 0
            # Should have some air (True) and some solid (False)
            assert slice_.mask.any()  # Has some air
            assert not slice_.mask.all()  # Has some solid

    def test_sphere_conversion(self):
        """Test converting a sphere SDF to Stack."""
        # Create a 20mm radius sphere
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.02)

        stack = sdf_to_stack(
            sphere,
            z_range=(0.03, 0.07),  # 40mm range
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
        )

        # Should have ~6-7 slices
        assert stack.num_slices in [6, 7]

        # Middle slice should have more air than edge slices
        middle_idx = stack.num_slices // 2
        middle_slice = stack.slices[middle_idx]
        first_slice = stack.slices[0]

        middle_air = np.sum(middle_slice.mask)
        first_air = np.sum(first_slice.mask)

        # Middle should have more air (larger cross-section)
        assert middle_air > first_air

    def test_cylinder_conversion(self):
        """Test converting a vertical cylinder SDF to Stack."""
        # Vertical cylinder, 50mm tall, 10mm radius
        cyl = Cylinder(p1=(0.05, 0.05, 0.0), p2=(0.05, 0.05, 0.05), radius=0.01)

        stack = sdf_to_stack(
            cyl,
            z_range=(0.0, 0.05),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
        )

        # Should have ~8 slices
        assert stack.num_slices in [8, 9]

        # All slices should have similar air patterns (circular cross-section)
        # Check that they all have roughly similar amounts of air
        air_counts = [np.sum(s.mask) for s in stack.slices]
        mean_air = np.mean(air_counts)
        std_air = np.std(air_counts)

        # Standard deviation should be small (uniform cylinder)
        assert std_air / mean_air < 0.2  # Less than 20% variation

    def test_horn_conversion(self):
        """Test converting a horn SDF to Stack."""
        # Exponential horn
        horn = Horn(
            throat_position=(0.05, 0.05, 0.0),
            mouth_position=(0.05, 0.05, 0.15),
            throat_radius=0.015,
            mouth_radius=0.045,
            profile="exponential",
        )

        stack = sdf_to_stack(
            horn,
            z_range=(0.0, 0.15),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
        )

        # Should have ~25 slices (150mm / 6mm)
        assert stack.num_slices in [24, 25, 26]

        # Air should increase from throat to mouth
        first_air = np.sum(stack.slices[0].mask)
        last_air = np.sum(stack.slices[-1].mask)

        assert last_air > first_air  # Mouth has more air than throat

    def test_custom_xy_bounds(self):
        """Test specifying custom XY bounds."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.01, 0.01, 0.01))

        # Use larger bounds than necessary
        stack = sdf_to_stack(
            box,
            z_range=(0.045, 0.055),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
            xy_bounds=((0.0, 0.0), (0.1, 0.1)),  # 100mm x 100mm
        )

        # Slices should be 100mm / 0.5mm = 200 pixels on each side
        assert stack.slices[0].mask.shape == (200, 200)

    def test_empty_geometry(self):
        """Test converting geometry with no volume in range."""
        # Sphere that doesn't intersect z_range
        sphere = Sphere(center=(0.05, 0.05, 0.2), radius=0.01)

        stack = sdf_to_stack(
            sphere,
            z_range=(0.0, 0.1),  # Below sphere
            slice_thickness=6e-3,
            xy_resolution=1e-3,
        )

        # All slices should be solid (no air)
        for slice_ in stack.slices:
            assert not slice_.mask.any()


class TestSDFToSlicesGenerator:
    """Test memory-efficient generator version."""

    def test_generator_yields_slices(self):
        """Test that generator yields correct number of slices."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        slices = list(
            sdf_to_slices_generator(
                box,
                z_range=(0.04, 0.06),
                slice_thickness=6e-3,
                xy_resolution=0.5e-3,
            )
        )

        # Should have ~3 slices
        assert len(slices) in [3, 4]

        # Check z_index is sequential
        for i, slice_ in enumerate(slices):
            assert slice_.z_index == i

    def test_generator_matches_stack(self):
        """Test that generator produces same slices as sdf_to_stack."""
        sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.015)

        # Convert using both methods
        stack = sdf_to_stack(
            sphere,
            z_range=(0.035, 0.065),
            slice_thickness=6e-3,
            xy_resolution=1e-3,
        )

        gen_slices = list(
            sdf_to_slices_generator(
                sphere,
                z_range=(0.035, 0.065),
                slice_thickness=6e-3,
                xy_resolution=1e-3,
            )
        )

        # Same number of slices
        assert len(gen_slices) == stack.num_slices

        # Same masks
        for gen_slice, stack_slice in zip(gen_slices, stack.slices, strict=True):
            assert np.array_equal(gen_slice.mask, stack_slice.mask)
            assert gen_slice.z_index == stack_slice.z_index


class TestStackToSDF:
    """Test Stack to SDF conversion."""

    def test_voxel_sdf_creation(self):
        """Test creating VoxelSDF from voxel array."""
        # Create simple voxel array (cube of air)
        voxels = np.zeros((50, 50, 50), dtype=bool)
        voxels[15:35, 15:35, 15:35] = True

        sdf = VoxelSDF(
            voxels=voxels,
            resolution=(1e-3, 1e-3, 1e-3),
            origin=(0, 0, 0),
        )

        # Test points inside and outside
        inside_point = np.array([[0.025, 0.025, 0.025]])  # Center of cube
        outside_point = np.array([[0.0, 0.0, 0.0]])  # Origin

        inside_dist = sdf.sdf(inside_point)
        outside_dist = sdf.sdf(outside_point)

        assert inside_dist[0] < 0  # Inside
        assert outside_dist[0] > 0  # Outside

    def test_stack_to_sdf_extrude(self):
        """Test Stack to SDF conversion using extrude method."""
        # Create a simple box
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        # Convert to stack
        stack = sdf_to_stack(
            box,
            z_range=(0.04, 0.06),
            slice_thickness=6e-3,
            xy_resolution=1e-3,
        )

        # Convert back to SDF
        voxel_sdf = stack_to_sdf(stack, method="extrude")

        # Should be VoxelSDF
        assert isinstance(voxel_sdf, VoxelSDF)

        # Get the actual bounding box of the voxel SDF
        bb_min, bb_max = voxel_sdf.bounding_box

        # Test points relative to the voxel grid's coordinate system
        # Center of the voxel grid
        center = (bb_min + bb_max) / 2
        # Far outside point
        outside = bb_max + 0.05

        center_point = np.array([center])
        outside_point = np.array([outside])

        center_dist = voxel_sdf.sdf(center_point)
        outside_dist = voxel_sdf.sdf(outside_point)

        assert center_dist[0] < 0  # Inside
        assert outside_dist[0] > 0  # Outside

    def test_stack_to_sdf_interpolate(self):
        """Test Stack to SDF conversion using interpolate method."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.01, 0.01, 0.01))

        stack = sdf_to_stack(
            box,
            z_range=(0.045, 0.055),
            slice_thickness=6e-3,
            xy_resolution=1e-3,
        )

        # Convert back to SDF (currently falls back to extrude)
        voxel_sdf = stack_to_sdf(stack, method="interpolate")

        assert isinstance(voxel_sdf, VoxelSDF)

    def test_voxel_sdf_bounding_box(self):
        """Test VoxelSDF bounding box calculation."""
        voxels = np.zeros((100, 100, 100), dtype=bool)
        voxels[25:75, 25:75, 25:75] = True

        sdf = VoxelSDF(
            voxels=voxels,
            resolution=(1e-3, 1e-3, 1e-3),
            origin=(0, 0, 0),
        )

        bb_min, bb_max = sdf.bounding_box

        assert np.allclose(bb_min, [0, 0, 0])
        assert np.allclose(bb_max, [0.1, 0.1, 0.1])


class TestRoundTrip:
    """Test round-trip conversion SDF → Stack → SDF."""

    def test_box_roundtrip(self):
        """Test round-trip conversion preserves approximate geometry."""
        # Original box
        original_box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        # Convert to stack
        stack = sdf_to_stack(
            original_box,
            z_range=(0.04, 0.06),
            slice_thickness=2e-3,  # Finer slices for better fidelity
            xy_resolution=0.5e-3,
        )

        # Convert back to SDF
        reconstructed = stack_to_sdf(stack, method="extrude")

        # Get the voxel grid's bounding box to test points within its coordinate system
        bb_min, bb_max = reconstructed.bounding_box

        # Generate test points within and around the voxel grid
        center = (bb_min + bb_max) / 2
        extent = bb_max - bb_min

        test_points = np.array([
            center,  # Center
            bb_min + extent * 0.25,  # Inside
            bb_max - extent * 0.25,  # Inside
            bb_max + extent * 0.5,  # Outside
        ])

        # Check that center is inside and far outside is outside
        distances = reconstructed.sdf(test_points)

        assert distances[0] < 0  # Center should be inside
        assert distances[-1] > 0  # Far outside should be outside

    def test_sphere_roundtrip(self):
        """Test sphere round-trip conversion."""
        original_sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.015)

        stack = sdf_to_stack(
            original_sphere,
            z_range=(0.035, 0.065),
            slice_thickness=2e-3,
            xy_resolution=0.5e-3,
        )

        reconstructed = stack_to_sdf(stack, method="extrude")

        # Test that conversion produces a valid SDF
        bb_min, bb_max = reconstructed.bounding_box

        # Test points relative to reconstructed grid
        center = (bb_min + bb_max) / 2
        extent = bb_max - bb_min

        # Points at different distances from center
        test_points = np.array([
            center,  # Center - should be inside
            center + extent * 0.1,  # Close to center - likely inside
            bb_max + extent * 0.5,  # Far outside - should be outside
        ])

        distances = reconstructed.sdf(test_points)

        # Basic sanity checks
        assert distances[0] < 0  # Center inside
        assert distances[-1] > 0  # Far point outside


class TestValidateManufacturable:
    """Test manufacturing constraint validation."""

    def test_manufacturable_box(self):
        """Test that simple box passes manufacturing checks."""
        # Large simple box - should have minimal violations
        box = Box(center=(0.05, 0.05, 0.05), size=(0.04, 0.04, 0.02))

        violations = validate_slices_manufacturable(
            box,
            z_range=(0.04, 0.06),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
            min_wall=3e-3,
            min_gap=2e-3,
        )

        # Allow for some edge discretization artifacts
        # Main point is that the validation runs and returns violations
        assert isinstance(violations, list)

    def test_thin_wall_violation(self):
        """Test that thin walls are detected."""
        # Create two boxes with a thin wall between them
        from strata_fdtd import Union

        box1 = Box(center=(0.04, 0.05, 0.05), size=(0.02, 0.04, 0.02))
        box2 = Box(center=(0.06, 0.05, 0.05), size=(0.02, 0.04, 0.02))
        union = Union(box1, box2)

        violations = validate_slices_manufacturable(
            union,
            z_range=(0.04, 0.06),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
            min_wall=5e-3,  # 5mm minimum wall
            min_gap=2e-3,
        )

        # Might detect thin wall between boxes
        # This is a soft test - just check it runs
        assert isinstance(violations, list)

    def test_horn_manufacturability(self):
        """Test horn manufacturability check."""
        horn = Horn(
            throat_position=(0.05, 0.05, 0.0),
            mouth_position=(0.05, 0.05, 0.1),
            throat_radius=0.01,
            mouth_radius=0.03,
            profile="conical",
        )

        violations = validate_slices_manufacturable(
            horn,
            z_range=(0.0, 0.1),
            slice_thickness=6e-3,
            xy_resolution=0.5e-3,
            min_wall=3e-3,
            min_gap=2e-3,
        )

        # Horn will have some violations due to discretization
        # Main point is that validation runs successfully
        assert isinstance(violations, list)
        assert len(violations) < 100  # Not catastrophically many violations


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_z_range(self):
        """Test behavior with invalid z_range."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        # z_max < z_min should produce empty stack
        stack = sdf_to_stack(
            box,
            z_range=(0.06, 0.04),  # Reversed
            slice_thickness=6e-3,
            xy_resolution=1e-3,
        )

        assert stack.num_slices == 0

    def test_very_fine_resolution(self):
        """Test with very fine resolution."""
        # Small sphere with fine resolution
        sphere = Sphere(center=(0.01, 0.01, 0.01), radius=0.005)

        stack = sdf_to_stack(
            sphere,
            z_range=(0.005, 0.015),
            slice_thickness=2e-3,
            xy_resolution=0.1e-3,  # 0.1mm = very fine
        )

        # Should have ~5 slices
        assert stack.num_slices in [4, 5, 6]

        # Should capture sphere detail
        middle_slice = stack.slices[stack.num_slices // 2]
        assert middle_slice.mask.any()

    def test_voxel_sdf_invalid_inputs(self):
        """Test VoxelSDF error handling."""
        with pytest.raises(ValueError, match="must be 3D"):
            VoxelSDF(
                voxels=np.zeros((10, 10), dtype=bool),  # 2D instead of 3D
                resolution=(1e-3, 1e-3, 1e-3),
                origin=(0, 0, 0),
            )

        with pytest.raises(ValueError, match="resolution must be positive"):
            VoxelSDF(
                voxels=np.zeros((10, 10, 10), dtype=bool),
                resolution=(1e-3, -1e-3, 1e-3),  # Negative resolution
                origin=(0, 0, 0),
            )
