"""Tests for core geometry classes (Slice, Stack)."""

import numpy as np

from strata_fdtd.manufacturing.lamination import (
    DEFAULT_SLICE_THICKNESS,
    Slice,
    Stack,
    Violation,
    array_stack,
    difference,
    union,
)


class TestSlice:
    """Tests for the Slice class."""

    def test_create_simple_slice(self):
        """Test creating a basic slice."""
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[20:80, 20:80] = True  # Air region in center

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)

        assert slice_.z_index == 0
        assert slice_.resolution == 0.5
        assert slice_.mask.shape == (100, 100)

    def test_slice_shape_mm(self):
        """Test shape_mm property returns correct dimensions."""
        mask = np.zeros((100, 200), dtype=np.bool_)
        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)

        w, h = slice_.shape_mm
        assert w == 100.0  # 200 pixels * 0.5 mm/pixel
        assert h == 50.0   # 100 pixels * 0.5 mm/pixel

    def test_solid_mask_inverts_air_mask(self):
        """Test that solid_mask is inverse of mask."""
        mask = np.array([[True, False], [False, True]], dtype=np.bool_)
        slice_ = Slice(mask=mask, z_index=0)

        expected_solid = np.array([[False, True], [True, False]], dtype=np.bool_)
        np.testing.assert_array_equal(slice_.solid_mask, expected_solid)

    def test_is_connected_single_region(self):
        """Test connectivity detection for connected geometry."""
        # Create L-shaped solid (connected)
        mask = np.ones((50, 50), dtype=np.bool_)  # All air
        mask[10:40, 10:20] = False  # Vertical solid bar
        mask[30:40, 10:40] = False  # Horizontal solid bar (connected)

        slice_ = Slice(mask=mask, z_index=0)
        assert slice_.is_connected() is True

    def test_is_connected_disconnected_regions(self):
        """Test connectivity detection for disconnected geometry."""
        # Create two separate rectangles
        mask = np.ones((50, 50), dtype=np.bool_)  # All air
        mask[5:15, 5:15] = False   # First solid block
        mask[30:40, 30:40] = False  # Second solid block (disconnected)

        slice_ = Slice(mask=mask, z_index=0)
        assert slice_.is_connected() is False

    def test_from_rectangles_creates_air_regions(self):
        """Test creating slice from rectangle specifications."""
        slice_ = Slice.from_rectangles(
            width_mm=50.0,
            height_mm=50.0,
            z_index=0,
            air_rects=[(10.0, 10.0, 20.0, 20.0)],  # (x, y, w, h)
            resolution=1.0,
        )

        # Check dimensions
        assert slice_.mask.shape == (50, 50)

        # Check air region exists
        assert slice_.mask[15, 15]  # Inside air rect
        assert not slice_.mask[5, 5]  # Outside (solid)

    def test_boundary_polygons_returns_contours(self):
        """Test that boundary_polygons extracts cut paths."""
        # Create simple square air region
        mask = np.zeros((50, 50), dtype=np.bool_)
        mask[10:40, 10:40] = True  # Square air region

        slice_ = Slice(mask=mask, z_index=0, resolution=1.0)
        polygons = slice_.boundary_polygons()

        assert len(polygons) >= 1  # At least one boundary
        # Each polygon should have multiple points
        assert len(polygons[0]) >= 4


class TestStack:
    """Tests for the Stack class."""

    def test_create_empty_stack(self):
        """Test creating an empty stack."""
        stack = Stack()
        assert stack.num_slices == 0
        assert stack.height_mm == 0.0

    def test_create_stack_with_slices(self):
        """Test creating stack with multiple slices."""
        slices = []
        for z in range(5):
            mask = np.zeros((50, 50), dtype=np.bool_)
            slices.append(Slice(mask=mask, z_index=z))

        stack = Stack(slices=slices)

        assert stack.num_slices == 5
        assert stack.height_mm == 5 * DEFAULT_SLICE_THICKNESS

    def test_stack_sorts_slices_by_z_index(self):
        """Test that slices are automatically sorted by z_index."""
        slices = [
            Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=2),
            Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=0),
            Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=1),
        ]
        stack = Stack(slices=slices)

        assert [s.z_index for s in stack.slices] == [0, 1, 2]

    def test_add_slice_maintains_order(self):
        """Test adding slices maintains z_index order."""
        stack = Stack()
        stack.add_slice(Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=2))
        stack.add_slice(Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=0))
        stack.add_slice(Slice(mask=np.zeros((10, 10), dtype=np.bool_), z_index=1))

        assert [s.z_index for s in stack.slices] == [0, 1, 2]

    def test_get_slice_by_index(self):
        """Test retrieving slice by z_index."""
        slices = [
            Slice(mask=np.ones((10, 10), dtype=np.bool_), z_index=i)
            for i in range(3)
        ]
        stack = Stack(slices=slices)

        slice_1 = stack.get_slice(1)
        assert slice_1 is not None
        assert slice_1.z_index == 1

        slice_5 = stack.get_slice(5)
        assert slice_5 is None

    def test_bounding_box(self):
        """Test bounding box calculation."""
        mask = np.zeros((40, 80), dtype=np.bool_)  # 80 wide, 40 tall at 0.5 res
        slices = [Slice(mask=mask, z_index=i, resolution=0.5) for i in range(3)]
        stack = Stack(slices=slices, resolution_xy=0.5)

        bbox = stack.bounding_box
        assert bbox[0] == 40.0  # width = 80 * 0.5
        assert bbox[1] == 20.0  # depth = 40 * 0.5
        assert bbox[2] == 3 * DEFAULT_SLICE_THICKNESS  # height

    def test_to_3d_mask_shape(self):
        """Test 3D mask generation has correct shape."""
        mask = np.zeros((40, 60), dtype=np.bool_)
        slices = [Slice(mask=mask, z_index=i) for i in range(5)]
        stack = Stack(slices=slices)

        mask_3d = stack.to_3d_mask()

        assert mask_3d.shape == (40, 60, 5)

    def test_to_3d_mask_preserves_data(self):
        """Test that 3D mask preserves slice data correctly."""
        # Create distinct patterns per slice
        slices = []
        for z in range(3):
            mask = np.zeros((20, 20), dtype=np.bool_)
            mask[z * 5:(z + 1) * 5, :] = True  # Different rows for each slice
            slices.append(Slice(mask=mask, z_index=z))

        stack = Stack(slices=slices)
        mask_3d = stack.to_3d_mask()

        # Check each slice preserved
        for z in range(3):
            np.testing.assert_array_equal(mask_3d[:, :, z], slices[z].mask)

    def test_from_3d_mask_roundtrip(self):
        """Test roundtrip: Stack -> 3D mask -> Stack."""
        mask = np.zeros((30, 30), dtype=np.bool_)
        mask[10:20, 10:20] = True
        slices = [Slice(mask=mask.copy(), z_index=i) for i in range(4)]
        original = Stack(slices=slices)

        mask_3d = original.to_3d_mask()
        recovered = Stack.from_3d_mask(mask_3d)

        assert recovered.num_slices == original.num_slices
        for orig_slice, rec_slice in zip(original.slices, recovered.slices, strict=True):
            np.testing.assert_array_equal(orig_slice.mask, rec_slice.mask)


class TestViolation:
    """Tests for the Violation class."""

    def test_violation_str_format(self):
        """Test violation string representation."""
        v = Violation(
            constraint="min_wall_thickness",
            slice_index=2,
            location=(10.5, 20.3),
            measured=2.1,
            required=3.0,
        )

        s = str(v)
        assert "min_wall_thickness" in s
        assert "slice 2" in s
        assert "10.5" in s
        assert "20.3" in s


class TestBooleanOperations:
    """Tests for union, difference, and array operations."""

    def test_union_combines_air_regions(self):
        """Test union creates air where either input has air."""
        # Stack A: air on left half
        mask_a = np.zeros((20, 20), dtype=np.bool_)
        mask_a[:, :10] = True
        slice_a = Slice(mask=mask_a, z_index=0)
        stack_a = Stack(slices=[slice_a])

        # Stack B: air on top half
        mask_b = np.zeros((20, 20), dtype=np.bool_)
        mask_b[:10, :] = True
        slice_b = Slice(mask=mask_b, z_index=0)
        stack_b = Stack(slices=[slice_b])

        result = union(stack_a, stack_b)

        # Result should have air in L-shape (top and left)
        result_mask = result.get_slice(0).mask
        assert result_mask[5, 5]  # Top-left (both have air)
        assert result_mask[5, 15]  # Top-right (B has air)
        assert result_mask[15, 5]  # Bottom-left (A has air)
        assert not result_mask[15, 15]  # Bottom-right (neither has air)

    def test_difference_removes_air(self):
        """Test difference makes solid where B has air."""
        # Stack A: all air
        mask_a = np.ones((20, 20), dtype=np.bool_)
        slice_a = Slice(mask=mask_a, z_index=0)
        stack_a = Stack(slices=[slice_a])

        # Stack B: air in center
        mask_b = np.zeros((20, 20), dtype=np.bool_)
        mask_b[5:15, 5:15] = True
        slice_b = Slice(mask=mask_b, z_index=0)
        stack_b = Stack(slices=[slice_b])

        result = difference(stack_a, stack_b)

        result_mask = result.get_slice(0).mask
        # Outside center should still be air
        assert result_mask[0, 0]
        # Center should be solid (B had air there)
        assert not result_mask[10, 10]

    def test_array_stack_repeats_primitive(self):
        """Test array operation creates copies at positions."""
        # Simple small primitive
        mask = np.ones((10, 10), dtype=np.bool_)  # All air
        slice_ = Slice(mask=mask, z_index=0, resolution=1.0)
        primitive = Stack(slices=[slice_], resolution_xy=1.0)

        # Place at two positions
        positions = [(0.0, 0.0, 0), (20.0, 0.0, 0)]
        result = array_stack(primitive, positions)

        result_slice = result.get_slice(0)
        # Should have air at both positions
        assert result_slice.mask[5, 5]  # First copy
        assert result_slice.mask[5, 25]  # Second copy
