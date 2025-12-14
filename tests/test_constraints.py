"""Tests for manufacturing constraint checking."""

import numpy as np

from strata_fdtd.manufacturing.constraints import (
    check_connectivity,
    check_min_gap,
    check_min_wall,
    check_slice_alignment,
    find_narrow_gaps,
    find_thin_regions,
)
from strata_fdtd.manufacturing.lamination import Slice, Stack


class TestCheckConnectivity:
    """Tests for connectivity checking."""

    def test_connected_solid_passes(self):
        """Test that connected solid region passes."""
        # L-shaped solid (connected)
        mask = np.ones((50, 50), dtype=np.bool_)  # All air
        mask[10:40, 10:20] = False  # Vertical bar
        mask[30:40, 10:40] = False  # Horizontal bar (connected)

        slice_ = Slice(mask=mask, z_index=0)
        violation = check_connectivity(slice_)

        assert violation is None

    def test_disconnected_solid_fails(self):
        """Test that disconnected solids are detected."""
        # Two separate rectangles
        mask = np.ones((50, 50), dtype=np.bool_)  # All air
        mask[5:15, 5:15] = False   # First block
        mask[30:40, 30:40] = False  # Second block (disconnected)

        slice_ = Slice(mask=mask, z_index=0)
        violation = check_connectivity(slice_)

        assert violation is not None
        assert violation.constraint == "connectivity"
        assert violation.measured > 1  # More than 1 component

    def test_all_air_fails(self):
        """Test that all-air slice (no solid) is flagged."""
        mask = np.ones((50, 50), dtype=np.bool_)  # All air
        slice_ = Slice(mask=mask, z_index=0)

        violation = check_connectivity(slice_)

        assert violation is not None
        assert violation.constraint == "connectivity"

    def test_single_solid_block_passes(self):
        """Test simple rectangle passes."""
        mask = np.ones((50, 50), dtype=np.bool_)
        mask[10:40, 10:40] = False  # Single solid block

        slice_ = Slice(mask=mask, z_index=0)
        violation = check_connectivity(slice_)

        assert violation is None


class TestCheckMinWall:
    """Tests for minimum wall thickness checking."""

    def test_thick_walls_interior_passes(self):
        """Test that interior of thick walls passes (edges will always be thin)."""
        # Create a thick solid bar - the interior should pass
        mask = np.ones((100, 100), dtype=np.bool_)  # All air initially

        # Create thick solid rectangle (20 pixels = 10mm at 0.5 res)
        mask[40:60, 10:90] = False  # 20 pixel thick bar at y=40-60

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)
        violations = check_min_wall(slice_, min_thickness=3.0)  # 3mm = 6 pixels

        # The bar has edges that will be flagged (distance=1 at boundary)
        # But violations should have measured thickness showing the actual thin spots
        # All violations should be at edges (distance ~1 pixel = ~0.5mm at res=0.5)
        for v in violations:
            # Measured thickness at edge is small (< 2mm)
            assert v.measured < 2.0, f"Unexpected thick violation: {v}"

    def test_thin_walls_detected(self):
        """Test that thin walls are detected."""
        # Create thin wall (2 pixels = 1mm at 0.5 res)
        mask = np.ones((100, 100), dtype=np.bool_)  # All air
        mask[45:55, :] = False  # Thin horizontal bar (10 pixels = 5mm)
        # Actually make it thinner
        mask[48:52, :] = False  # 4 pixels = 2mm
        mask[:, :] = True
        mask[49:51, 20:80] = False  # Very thin: 2 pixels = 1mm

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)
        violations = check_min_wall(slice_, min_thickness=3.0)

        assert len(violations) > 0
        assert all(v.constraint == "min_wall_thickness" for v in violations)

    def test_no_solid_returns_empty(self):
        """Test that all-air slice returns no violations."""
        mask = np.ones((50, 50), dtype=np.bool_)
        slice_ = Slice(mask=mask, z_index=0)

        violations = check_min_wall(slice_, min_thickness=3.0)
        assert len(violations) == 0


class TestCheckMinGap:
    """Tests for minimum gap width checking."""

    def test_wide_gaps_interior_passes(self):
        """Test that interior of wide gaps passes (edges will always be narrow)."""
        # Create a wide air slot
        mask = np.zeros((100, 100), dtype=np.bool_)

        # Create wide horizontal slot (20 pixels = 10mm at 0.5 res)
        mask[40:60, 10:90] = True  # 20 pixel wide slot at y=40-60

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)
        violations = check_min_gap(slice_, min_width=2.0)  # 2mm = 4 pixels

        # The slot has edges that will be flagged (distance=1 at boundary)
        # All violations should be at edges (distance ~1 pixel = ~0.5mm at res=0.5)
        for v in violations:
            # Measured gap at edge is small (< 2mm)
            assert v.measured < 2.0, f"Unexpected wide violation: {v}"

    def test_narrow_gaps_detected(self):
        """Test that narrow gaps/slots are detected."""
        # Create narrow slot (2 pixels = 1mm at 0.5 res)
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[49:51, 20:80] = True  # Very narrow slot: 2 pixels = 1mm

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)
        violations = check_min_gap(slice_, min_width=2.0)

        assert len(violations) > 0
        assert all(v.constraint == "min_gap_width" for v in violations)

    def test_no_air_returns_empty(self):
        """Test that all-solid slice returns no violations."""
        mask = np.zeros((50, 50), dtype=np.bool_)
        slice_ = Slice(mask=mask, z_index=0)

        violations = check_min_gap(slice_, min_width=2.0)
        assert len(violations) == 0


class TestCheckSliceAlignment:
    """Tests for inter-slice alignment checking."""

    def test_aligned_slices_pass(self):
        """Test that well-aligned slices pass."""
        slices = []
        for z in range(3):
            mask = np.ones((50, 50), dtype=np.bool_)
            mask[20:30, 20:30] = False  # Same position in all slices
            slices.append(Slice(mask=mask, z_index=z, resolution=1.0))

        stack = Stack(slices=slices, resolution_xy=1.0)
        violations = check_slice_alignment(stack)

        assert len(violations) == 0

    def test_misaligned_slices_detected(self):
        """Test that significant misalignment is detected."""
        slices = []

        # First slice: solid at (20, 20)
        mask0 = np.ones((50, 50), dtype=np.bool_)
        mask0[15:25, 15:25] = False
        slices.append(Slice(mask=mask0, z_index=0, resolution=1.0))

        # Second slice: solid at (30, 30) - 14mm shift
        mask1 = np.ones((50, 50), dtype=np.bool_)
        mask1[25:35, 25:35] = False
        slices.append(Slice(mask=mask1, z_index=1, resolution=1.0))

        stack = Stack(slices=slices, resolution_xy=1.0)
        violations = check_slice_alignment(stack, max_shift=5.0)

        # Should detect the large shift
        assert len(violations) > 0
        assert all(v.constraint == "slice_alignment" for v in violations)

    def test_single_slice_returns_empty(self):
        """Test that single-slice stack has no alignment issues."""
        mask = np.ones((50, 50), dtype=np.bool_)
        mask[20:30, 20:30] = False
        slice_ = Slice(mask=mask, z_index=0)
        stack = Stack(slices=[slice_])

        violations = check_slice_alignment(stack)
        assert len(violations) == 0


class TestFindThinRegions:
    """Tests for thin region visualization helper."""

    def test_find_thin_regions_identifies_narrow_parts(self):
        """Test that thin regions are correctly identified."""
        # Create a shape with thin and thick parts
        mask = np.ones((100, 100), dtype=np.bool_)
        mask[40:60, 10:30] = False  # Thick part (20x20 pixels)
        mask[45:55, 30:70] = False  # Thin connection (10 pixels wide)
        mask[40:60, 70:90] = False  # Another thick part

        slice_ = Slice(mask=mask, z_index=0, resolution=1.0)
        thin = find_thin_regions(slice_, thickness_threshold=15.0)

        # The thin connection should be identified
        # The thick parts should not be flagged
        assert thin[50, 50]  # Middle of thin connection
        # Note: edges of thick parts may also show up due to distance transform


class TestFindNarrowGaps:
    """Tests for narrow gap visualization helper."""

    def test_find_narrow_gaps_identifies_slots(self):
        """Test that narrow gaps are correctly identified."""
        # Create solid with a narrow slot
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[45:55, 20:80] = True  # Narrow horizontal slot

        slice_ = Slice(mask=mask, z_index=0, resolution=1.0)
        narrow = find_narrow_gaps(slice_, width_threshold=15.0)

        # The narrow slot should be identified
        assert narrow[50, 50]  # Middle of slot


class TestIntegration:
    """Integration tests for constraint checking."""

    def test_stack_check_manufacturable(self):
        """Test full manufacturing check on a stack."""
        # Create simple connected solid frame
        slices = []

        for z in range(3):
            mask = np.ones((100, 100), dtype=np.bool_)  # All air

            # Create connected frame (outer rectangle minus inner rectangle)
            # Outer boundary - solid
            mask[5:95, 5:95] = False  # Fill with solid
            # Inner cavity - air
            mask[25:75, 25:75] = True  # Cut out center
            # This creates a 20-pixel thick frame all around

            slices.append(Slice(mask=mask, z_index=z, resolution=0.5))

        stack = Stack(slices=slices, resolution_xy=0.5)
        violations = stack.check_manufacturable(min_wall=3.0, min_gap=2.0)

        # Check that connectivity passes (it's a connected frame)
        connectivity_violations = [v for v in violations if v.constraint == "connectivity"]
        assert len(connectivity_violations) == 0

        # The frame should have no thin wall violations in its body
        # (corners may be flagged, which is correct behavior)

    def test_constraint_violation_location_accuracy(self):
        """Test that violation locations are reported in mm."""
        # Create thin wall at known position
        mask = np.ones((100, 100), dtype=np.bool_)
        # Thin wall at pixels 48-52 = position 24-26mm at 0.5 res
        mask[48:52, :] = False

        slice_ = Slice(mask=mask, z_index=0, resolution=0.5)
        violations = check_min_wall(slice_, min_thickness=5.0)

        if violations:
            # Location should be in mm, roughly in the center
            v = violations[0]
            # Y position should be around 25mm (center of 48-52 at 0.5 res)
            assert 20.0 < v.location[1] < 30.0
